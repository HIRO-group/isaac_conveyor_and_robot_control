"""Main control loop: steps physics, drives both pick-and-place controllers,
runs conveyor indexing at the control rate, logs, and despawns landed boxes.
"""

from __future__ import annotations

import logging
import os
import signal

import numpy as np

from cameras.frame_meta import now_us
from conveyor_indexing.protos import plc, sim_action
from sim_cell import layout, settings
from sim_cell.cell import build_cell
from sim_cell.debug import dump_tick_debug
from sim_cell.pick_dispatch import evaluate_pick_station
from sim_cell.recording import (
    BOX_EVENT_DESPAWNED,
    BOX_EVENT_SPAWNED,
    EpisodeTracker,
    build_box_states,
    build_observation_state,
)
from sim_cell.stage_setup import prepare_stage
from sim_cell.stage_setup.truck import despawn_boxes_in_truck

# Suction on + all 8 cups on - the sim's magic attach has no per-cup
# actuation, so this always toggles as one block (see sim_cell.recording's
# module docstring). Matches theia's real dio_blocks[0] bit layout: 0x10000 =
# suction, low byte = cup mask.
_DIO_HOLDING = 0x10000 | 0xFF
_DIO_EMPTY = 0

logger = logging.getLogger(__name__)


def run(simulation_app) -> None:
    stage_prep = prepare_stage()
    cell = build_cell(stage_prep)
    logger.info("pick/place controllers ready, entering main loop")

    control_period_s = 1.0 / settings.CONTROL_HZ
    last_control_time = 0.0
    camera_period_s = 1.0 / settings.CAMERA_FPS
    last_camera_time = 0.0
    render_count = 0
    tick = 0
    pick_ready = False
    pick_box_path = None
    pick_ready_2 = False
    pick_box_path_2 = None

    # Training-data recording (None unless CONVEYOR_INDEXING_RECORD=1 - see
    # sim_cell.recording). latest_plc_bytes carries the control block's most
    # recent StateConveyors serialization into the camera block's recorded
    # rows (at most one control period stale).
    recorder = cell.episode_recorder
    episode_tracker = EpisodeTracker()
    latest_plc_bytes = None

    # Episode-free ground-truth recording (None unless
    # CONVEYOR_INDEXING_RECORD_MCAP=1 - see sim_cell.recording). Independent of
    # `recorder` above - either, both, or neither can be enabled.
    mcap_recorder = cell.mcap_recorder
    camera_role_by_serial = {spec.serial: spec.role for spec in cell.camera_specs}
    box_id_to_variant = {path: variant for variant, paths in cell.pool.paths_by_variant.items() for path in paths}
    active_box_paths: set = set()
    prev_phase_1 = cell.pick_place.phase_name
    prev_phase_2 = cell.pick_place_2.phase_name

    # Batch data-collection runs (e.g. one job per Vertex AI worker) cap sim time
    # instead of running indefinitely; unset (default) preserves today's
    # run-until-closed/SIGINT behavior. Checked against sim_time, not wall clock -
    # consistent with everything else in this loop being sim-time paced.
    max_sim_seconds_env = os.environ.get("CONVEYOR_INDEXING_MAX_SIM_SECONDS")
    max_sim_seconds = float(max_sim_seconds_env) if max_sim_seconds_env else None

    # SIGTERM otherwise kills the process immediately, skipping the finally block
    # below and leaving the parquet writer's file truncated - route it through the
    # normal loop-exit path instead, same as SIGINT.
    shutdown_requested = False

    def _handle_sigterm(signum, frame):
        nonlocal shutdown_requested
        shutdown_requested = True

    signal.signal(signal.SIGTERM, _handle_sigterm)

    world = cell.world

    try:
        while simulation_app.is_running() and not shutdown_requested:
            if world.is_playing():
                # Physics-only step every iteration - the full CONTROL_HZ rate
                # pick-and-place needs for smooth convergence. Rendering (6 camera
                # render products + viewport) is decoupled below and only paid for
                # on iterations that actually need a fresh camera frame; `render=True`
                # would otherwise render at RENDERING_DT (60Hz) even though frames
                # are only ever consumed at CAMERA_FPS (30Hz) - twice the RTX work
                # this loop actually uses.
                world.step(render=False)
                sim_time = world.current_time

                # Same shutdown path as SIGTERM/SIGINT below - flushes the recorder(s)
                # cleanly via the `finally` block instead of exiting mid-frame.
                if max_sim_seconds is not None and sim_time >= max_sim_seconds:
                    logger.info("sim-time cap of %.1fs reached at tick %d - shutting down", max_sim_seconds, tick)
                    shutdown_requested = True
                    continue

                # Pick-and-place runs every physics step for smooth convergence; conveyor
                # indexing runs at the coarser control rate below.
                cell.pick_place.forward(pick_ready, pick_box_path)
                cell.pick_place_2.forward(pick_ready_2, pick_box_path_2)

                # Every physics step (not just recorded ones) so no WAITING->pick
                # edge is missed between 30Hz samples.
                if recorder is not None:
                    episode_tracker.update(cell.pick_place.phase_name, cell.pick_place_2.phase_name)

                # Same every-physics-step cadence, for the same reason - no
                # transition dropped between 30Hz camera samples.
                if mcap_recorder is not None:
                    phase_1 = cell.pick_place.phase_name
                    if phase_1 != prev_phase_1:
                        mcap_recorder.record_phase_transition(
                            sim_time, 1, prev_phase_1, phase_1, cell.pick_place.held_box_path or ""
                        )
                        prev_phase_1 = phase_1
                    phase_2 = cell.pick_place_2.phase_name
                    if phase_2 != prev_phase_2:
                        mcap_recorder.record_phase_transition(
                            sim_time, 2, prev_phase_2, phase_2, cell.pick_place_2.held_box_path or ""
                        )
                        prev_phase_2 = phase_2

                # Paced at CAMERA_FPS (30Hz). world.render() refreshes render products
                # (and the viewport) without stepping physics again - see
                # SimulationContext.render(), which disables playSimulations for the
                # duration of its app.update() call.
                if sim_time - last_camera_time >= camera_period_s:
                    world.render()
                    capture_ts_us = now_us()
                    frames = cell.camera_rig.capture_all()
                    for serial, rgb_bytes in frames.items():
                        cell.camera_publisher.publish_frame(serial, rgb_bytes, capture_ts_us)
                        if mcap_recorder is not None:
                            mcap_recorder.record_camera_frame(
                                serial,
                                camera_role_by_serial[serial],
                                sim_time,
                                capture_ts_us,
                                rgb_bytes,
                                settings.CAMERA_WIDTH,
                                settings.CAMERA_HEIGHT,
                            )
                    # Images + state sampled in the same iteration = the synchronized
                    # training rows theia's converter expects. Skipped while annotators
                    # are still warming up (partial frames) or before the first control
                    # tick has serialized conveyor state.
                    if recorder is not None and latest_plc_bytes is not None and frames.keys() == recorder.expected_serials:
                        recorder.record(
                            reference_req_id=episode_tracker.episode_id,
                            observation_state=build_observation_state(
                                cell.robot, cell.robot2, cell.pick_place.holding_box, cell.pick_place_2.holding_box
                            ),
                            frames=frames,
                            plc_state_conveyors=latest_plc_bytes,
                            tick=tick,
                            sim_time_s=sim_time,
                            phase_1=cell.pick_place.phase_name,
                            phase_2=cell.pick_place_2.phase_name,
                        )
                    last_camera_time = sim_time

                if sim_time - last_control_time >= control_period_s:
                    # One batched pose read for every box, reused below by
                    # ConveyorLineController.step, despawn_boxes_in_truck, and
                    # evaluate_pick_station - instead of each of them calling
                    # get_world_poses() per box (a GPU sync + host copy every time).
                    # orientations previously discarded here - kept (at no extra GPU
                    # sync cost, it's the same batched call) for ground-truth box
                    # recording below.
                    positions, orientations = cell.box_positions_view.get_world_poses()
                    box_positions = dict(zip(cell.box_paths_ordered, positions.numpy()))
                    box_orientations = None
                    box_linear_vel = None
                    box_angular_vel = None
                    if mcap_recorder is not None:
                        box_orientations = dict(zip(cell.box_paths_ordered, orientations.numpy()))
                        linear_vel, angular_vel = cell.box_positions_view.get_velocities()
                        box_linear_vel = dict(zip(cell.box_paths_ordered, linear_vel.numpy()))
                        box_angular_vel = dict(zip(cell.box_paths_ordered, angular_vel.numpy()))

                    state_msg = plc.StateConveyors()
                    commands_msg = sim_action.SimConveyorCommands()
                    cell.loop1.step(state_msg, commands_msg, box_positions)
                    cell.loop2.step(state_msg, commands_msg, box_positions)
                    landed_box_paths = despawn_boxes_in_truck(
                        cell.box_rigid_prims,
                        box_positions,
                        layout.TRUCK_PATH,
                        cell.truck_bed_min,
                        cell.truck_bed_max,
                    )
                    if mcap_recorder is not None:
                        for path in landed_box_paths:
                            mcap_recorder.record_box_event(
                                sim_time,
                                BOX_EVENT_DESPAWNED,
                                path,
                                box_id_to_variant.get(path, ""),
                                tuple(box_positions[path]),
                                tuple(box_orientations[path]),
                            )
                            active_box_paths.discard(path)
                    # Recycle truck-landed boxes back into the pool, then spawn a new
                    # wave if ConveyorTrack (loop1 zone 0) just emptied out - reuses
                    # the occupancy loop1.step already computed this tick.
                    cell.spawner.release(landed_box_paths)
                    spawned = cell.spawner.update(sim_time, cell.loop1.occupied[0])
                    if mcap_recorder is not None:
                        for path, variant, position, quat_wxyz in spawned:
                            mcap_recorder.record_box_event(
                                sim_time, BOX_EVENT_SPAWNED, path, variant, position, quat_wxyz
                            )
                            active_box_paths.add(path)
                            # box_positions/box_orientations were read at the top of this
                            # tick, before this box was teleported onto the belt just now -
                            # without this override, this tick's BoxStates would show the
                            # box at its stale parked pose (POOL_PARK_ORIGIN). Zero velocity
                            # is a reasonable approximation for "just placed, not yet fallen".
                            box_positions[path] = position
                            box_orientations[path] = quat_wxyz
                            box_linear_vel[path] = (0.0, 0.0, 0.0)
                            box_angular_vel[path] = (0.0, 0.0, 0.0)
                    latest_plc_bytes = state_msg.SerializeToString()
                    if mcap_recorder is not None:
                        # Same object, not re-parsed from latest_plc_bytes - state_msg
                        # is freshly built this tick and never mutated again.
                        mcap_recorder.record_state_conveyors(sim_time, state_msg)
                        mcap_recorder.record_position_status(
                            1,
                            sim_time,
                            list(np.degrees(cell.robot.get_dof_positions().numpy()[0])),
                            _DIO_HOLDING if cell.pick_place.holding_box else _DIO_EMPTY,
                        )
                        mcap_recorder.record_position_status(
                            2,
                            sim_time,
                            list(np.degrees(cell.robot2.get_dof_positions().numpy()[0])),
                            _DIO_HOLDING if cell.pick_place_2.holding_box else _DIO_EMPTY,
                        )
                        held_by_arm = {}
                        if cell.pick_place.held_box_path:
                            held_by_arm[cell.pick_place.held_box_path] = 1
                        if cell.pick_place_2.held_box_path:
                            held_by_arm[cell.pick_place_2.held_box_path] = 2
                        mcap_recorder.record_box_states(
                            sim_time,
                            build_box_states(
                                active_box_paths,
                                box_positions,
                                box_orientations,
                                box_linear_vel,
                                box_angular_vel,
                                box_id_to_variant,
                                held_by_arm,
                            ),
                        )
                    cell.tick_logger.log_tick(
                        tick=tick,
                        sim_time_s=sim_time,
                        plc_state_conveyors=latest_plc_bytes,
                        conveyor_commands=commands_msg.SerializeToString(),
                    )
                    # Only "ready" once the pick zone has settled into holding (IDLE +
                    # occupied); identify which box is actually there rather than assuming a fixed one.
                    pick_ready, pick_box_path = evaluate_pick_station(
                        cell.loop1.zones[layout.PICK_ZONE_INDEX],
                        cell.loop1.machine_states[layout.PICK_ZONE_INDEX],
                        box_positions,
                        cell.robot_xy,
                    )
                    pick_ready_2, pick_box_path_2 = evaluate_pick_station(
                        cell.loop1.zones[layout.PICK_ZONE_INDEX_2],
                        cell.loop1.machine_states[layout.PICK_ZONE_INDEX_2],
                        box_positions,
                        cell.robot_2_xy,
                    )
                    last_control_time = sim_time
                    tick += 1
                    dump_tick_debug(cell, tick, sim_time, pick_ready, pick_box_path, pick_ready_2, pick_box_path_2)
            else:
                render_count += 1
                if render_count % 60 == 1:
                    logger.info("world not playing (render_count=%d)", render_count)
                world.render()
    finally:
        if cell.episode_recorder is not None:
            cell.episode_recorder.close()
        if cell.mcap_recorder is not None:
            cell.mcap_recorder.close()
        cell.tick_logger.close()
        cell.camera_publisher.close()
        simulation_app.close()
