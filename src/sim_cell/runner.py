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
from conveyor_indexing.telemetry import resolve_override_speed_direction
from pick_and_place import apply_suction_edge
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
    resolve_arm_telemetry,
    validate_external_action_recording,
)
from sim_cell.stage_setup import prepare_stage
from sim_cell.stage_setup.truck import despawn_boxes_in_truck

# Suction on + all 8 cups on - the sim's magic attach has no per-cup
# actuation, so this always toggles as one block (see sim_cell.recording's
# module docstring). Matches theia's real dio_blocks[0] bit layout: 0x10000 =
# suction, low byte = cup mask.
_DIO_HOLDING = 0x10000 | 0xFF
_DIO_EMPTY = 0

# When set, an external controller (e.g. a trained policy, via
# sim_cell.external_command_bridge) drives both arms + both conveyors
# directly instead of the autonomous pick_and_place/conveyor_indexing control
# - see the top-level README's "Design" section. Still mutually exclusive
# with CONVEYOR_INDEXING_RECORD (the 30Hz episode/parquet recorder) - see
# validate_external_action_recording below - but CONVEYOR_INDEXING_RECORD_MCAP
# is allowed at the same time (needed for on-policy eval recording).
EXTERNAL_ACTION_ENV_VAR = "CONVEYOR_INDEXING_EXTERNAL_ACTION"

# Opt-out for camera frames specifically within CONVEYOR_INDEXING_RECORD_MCAP
# (default "1" - on, matching prior behavior; this is additive, nothing else
# changes unless set to "0"). record_camera_frame and record_position_status/
# etc. share the same EpisodeRecorder's bounded queue (mcap_recorder.py); 6
# cameras at 30Hz of full RGB frames can fill that queue faster than the
# writer thread drains it, and once full, every subsequent enqueue - including
# position_status - silently drops too. Confirmed against a real run where
# recording stopped entirely, for a task that only needs position_status,
# not camera frames.
RECORD_MCAP_CAMERAS_ENV_VAR = "CONVEYOR_INDEXING_RECORD_MCAP_CAMERAS"

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
    record_mcap_cameras = os.environ.get(RECORD_MCAP_CAMERAS_ENV_VAR, "1") == "1"

    external_action = os.environ.get(EXTERNAL_ACTION_ENV_VAR) == "1"
    # CONVEYOR_INDEXING_RECORD (30Hz episode/parquet recorder) still conflicts -
    # episode segmentation has no defined meaning once an external controller
    # owns the phase machine. CONVEYOR_INDEXING_RECORD_MCAP no longer does:
    # it's needed for on-policy eval recording (see the on-policy action-log
    # channels below); phase-transition recording is simply skipped instead
    # (the phase machine is dormant in external_action mode - see below).
    validate_external_action_recording(external_action, recorder is not None)
    # This arm's currently-held box path (None if not holding) while
    # external_action is set - MagicAttachPickPlace.holding_box/held_box_path
    # are frozen (forward() never runs), so external mode tracks its own
    # equivalent via pick_and_place.apply_suction_edge.
    held_box_path_1 = None
    held_box_path_2 = None

    if external_action:
        # cell.py registers each hold zone's overflow-readiness check against
        # pick_place.phase_name == "WAITING" (see build_cell) - correct for the
        # autonomous controller, but phase_name is frozen at its WAITING default
        # here (forward() is never called in this branch, same reason
        # holding_box is frozen above), so that check is permanently True. A
        # permanently-True "robot ready" means the zone permanently treats
        # itself as holding for this arm and NEVER overflows a box to the next
        # station's zone downstream, regardless of real activity - confirmed
        # against a real 270s recording where a box settled at the hold zone's
        # stop point and never advanced. Re-register both hold zones' checks
        # against the same live held_box_path_N signal already used for
        # holding_1/holding_2 above, so "ready" correctly means "not currently
        # holding a box" instead of an autonomous state that never updates.
        arm_holding_externally = {1: False, 2: False}
        cell.loop1.set_hold_zone_ready_check(layout.PICK_ZONE_INDEX, lambda: not arm_holding_externally[1])
        cell.loop1.set_hold_zone_ready_check(layout.PICK_ZONE_INDEX_2, lambda: not arm_holding_externally[2])

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

    # Only used in external_action mode, to route an externally-commanded
    # SimConveyorCommand to the zone it names - see the override right after
    # loop1.step/loop2.step below.
    zones_by_node_path = {zone.node_path: zone for zone in [*cell.loop1.zones, *cell.loop2.zones]}

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
                if external_action:
                    # Drive both arms directly from the latest externally-supplied command,
                    # bypassing MagicAttachPickPlace's phase state machine entirely (it's
                    # simply never called in this branch, so it stays dormant - no explicit
                    # pause needed). set_dof_position_targets is the exact same call
                    # TrajectoryDriver.drive_to() uses internally; the PD drive holds the
                    # last-set target for free on ticks where no new command has arrived yet.
                    cmd_arm1, cmd_arm2, cmd_conveyors = cell.external_command_bridge.latest()
                    if cmd_arm1 is not None:
                        cell.robot.set_dof_position_targets(
                            positions=np.asarray(cmd_arm1.joint_targets, dtype=np.float32)
                        )
                        held_box_path_1 = apply_suction_edge(
                            1, cell.pick_place, cell.box_rigid_prims, cmd_arm1.suction, held_box_path_1, pick_box_path
                        )
                        arm_holding_externally[1] = held_box_path_1 is not None
                        if mcap_recorder is not None:
                            mcap_recorder.record_arm_action_command(1, sim_time, cmd_arm1)
                    if cmd_arm2 is not None:
                        cell.robot2.set_dof_position_targets(
                            positions=np.asarray(cmd_arm2.joint_targets, dtype=np.float32)
                        )
                        held_box_path_2 = apply_suction_edge(
                            2, cell.pick_place_2, cell.box_rigid_prims, cmd_arm2.suction, held_box_path_2,
                            pick_box_path_2,
                        )
                        arm_holding_externally[2] = held_box_path_2 is not None
                        if mcap_recorder is not None:
                            mcap_recorder.record_arm_action_command(2, sim_time, cmd_arm2)
                else:
                    cell.pick_place.forward(pick_ready, pick_box_path)
                    cell.pick_place_2.forward(pick_ready_2, pick_box_path_2)

                # Every physics step (not just recorded ones) so no WAITING->pick
                # edge is missed between 30Hz samples.
                if recorder is not None:
                    episode_tracker.update(cell.pick_place.phase_name, cell.pick_place_2.phase_name)

                # Same every-physics-step cadence, for the same reason - no
                # transition dropped between 30Hz camera samples. Skipped
                # entirely in external_action mode: MagicAttachPickPlace.
                # forward() never runs there (see the external_action branch
                # above), so phase_name never changes from WAITING - recording
                # transitions would be meaningless, not just unchanging.
                if mcap_recorder is not None and not external_action:
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
                        if mcap_recorder is not None and record_mcap_cameras:
                            mcap_recorder.record_camera_frame(
                                serial,
                                camera_role_by_serial[serial],
                                sim_time,
                                capture_ts_us,
                                rgb_bytes,
                                settings.CAMERA_WIDTH,
                                settings.CAMERA_HEIGHT,
                            )
                    # Live arm-state publish, always on (like camera_publisher) regardless of
                    # control mode. In external_action mode, MagicAttachPickPlace.holding_box is
                    # frozen (forward() never runs) - held_box_path_1/2 are external mode's own
                    # equivalent, tracked by apply_suction_edge above.
                    holding_1 = held_box_path_1 is not None if external_action else cell.pick_place.holding_box
                    holding_2 = held_box_path_2 is not None if external_action else cell.pick_place_2.holding_box
                    cell.robot_state_publisher.publish_arm_state(
                        1, np.degrees(cell.robot.get_dof_positions().numpy()[0]), holding_1, capture_ts_us
                    )
                    cell.robot_state_publisher.publish_arm_state(
                        2, np.degrees(cell.robot2.get_dof_positions().numpy()[0]), holding_2, capture_ts_us
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
                    if external_action:
                        # Let step() run as normal first - it also drives occupancy/PackML
                        # bookkeeping that evaluate_pick_station() depends on for the arm
                        # box-lookup above, so skipping it would silently break arm control
                        # too. Only the belt command it just applied gets overridden here;
                        # OmniGraph only consumes belt attributes on the next physics
                        # substep, so this later same-tick override safely wins.
                        #
                        # state_msg's items were already populated by step() from its own
                        # autonomous decision, before this override - re-point Speed at what
                        # actually got commanded so theia/plc/state_conveyors (the "actual
                        # state" telemetry an external observer sees) doesn't silently report
                        # stale autonomous values while external_action owns the real belt.
                        items_by_name = {item.Name: item for item in state_msg.Conveyors}
                        _, _, cmd_conveyors = cell.external_command_bridge.latest()
                        if cmd_conveyors is not None:
                            for cmd in cmd_conveyors.commands:
                                zone = zones_by_node_path.get(cmd.conveyor_node_path)
                                if zone is not None:
                                    zone.apply_command(cmd.run, cmd.speed)
                                    item = items_by_name.get(cmd.conveyor_node_path)
                                    if item is not None:
                                        item.Speed, item.Direction = resolve_override_speed_direction(
                                            cmd.run, cmd.speed, cmd.direction
                                        )
                            if mcap_recorder is not None:
                                mcap_recorder.record_conveyor_command(sim_time, cmd_conveyors)
                        else:
                            # No external command has ever arrived yet - stop every zone rather
                            # than leaving step()'s autonomous decision in effect, so external-
                            # action mode never runs on the autonomous controller's behavior by
                            # default (see the top-level README's "Design" section).
                            for zone in zones_by_node_path.values():
                                zone.apply_command(False, 0)
                                item = items_by_name.get(zone.node_path)
                                if item is not None:
                                    item.Speed, item.Direction = resolve_override_speed_direction(False, 0, 0)
                    cell.robot_state_publisher.publish_conveyor_state(state_msg)
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
                        # holding_1/2 + held_by_arm must come from held_box_path_1/2 in
                        # external_action mode, NOT cell.pick_place(_2).holding_box/
                        # held_box_path - those are frozen (forward() never runs there),
                        # so every policy-run KPI would otherwise silently read "never
                        # holding". See sim_cell.recording.resolve_arm_telemetry.
                        holding_1, holding_2, held_by_arm = resolve_arm_telemetry(
                            external_action,
                            held_box_path_1,
                            held_box_path_2,
                            cell.pick_place.holding_box,
                            cell.pick_place_2.holding_box,
                            cell.pick_place.held_box_path,
                            cell.pick_place_2.held_box_path,
                        )
                        mcap_recorder.record_position_status(
                            1,
                            sim_time,
                            list(np.degrees(cell.robot.get_dof_positions().numpy()[0])),
                            _DIO_HOLDING if holding_1 else _DIO_EMPTY,
                        )
                        mcap_recorder.record_position_status(
                            2,
                            sim_time,
                            list(np.degrees(cell.robot2.get_dof_positions().numpy()[0])),
                            _DIO_HOLDING if holding_2 else _DIO_EMPTY,
                        )
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
                        # Independent of control mode (unlike holding_1/2 above) - the
                        # tool_prim GeomPrim tracks the arm's actual physical
                        # wrist_3_link/flange regardless of whether forward() runs.
                        # Recorded at the same 120Hz cadence as BoxStates/
                        # PositionStatus so FK is never needed downstream (see
                        # pick_and_place.controller.MagicAttachPickPlace.tool_world_pose).
                        tool_pos_1, tool_quat_1 = cell.pick_place.tool_world_pose()
                        mcap_recorder.record_tool_pose(1, sim_time, tuple(tool_pos_1), tuple(tool_quat_1))
                        tool_pos_2, tool_quat_2 = cell.pick_place_2.tool_world_pose()
                        mcap_recorder.record_tool_pose(2, sim_time, tuple(tool_pos_2), tuple(tool_quat_2))
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
        cell.robot_state_publisher.close()
        if cell.external_command_bridge is not None:
            cell.external_command_bridge.close()
        simulation_app.close()
