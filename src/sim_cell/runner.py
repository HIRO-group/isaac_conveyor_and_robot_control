"""Main control loop: steps physics, drives both pick-and-place controllers,
runs conveyor indexing at the control rate, logs, and despawns landed boxes.
"""

from __future__ import annotations

import logging
import os
import signal
import time

import numpy as np

from cameras.frame_meta import now_us
from conveyor_indexing.protos import sim_action, telemetry
from conveyor_indexing.telemetry import resolve_override_speed_direction
from pick_and_place import apply_suction_edge
from sim_cell import layout, settings
from sim_cell.cell import build_cell
from sim_cell.control import ControlChannel, ControlMode, ModeController
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
from sim_cell.stage_setup.truck import (
    despawn_boxes_below_floor,
    despawn_boxes_in_truck,
    despawn_boxes_off_belt,
    despawn_stale_boxes,
)

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

# Skip camera work entirely (render products, GPU capture, and the Zenoh
# frame publish) - default "1", i.e. unchanged behavior. Distinct from
# RECORD_MCAP_CAMERAS, which only skips WRITING frames to MCAP while still
# paying for the render and capture. For a collection run whose dataset is
# built purely from joint/box/conveyor state, that render is the single
# largest per-frame cost and buys nothing: the loop is already unthrottled
# (world.step in a tight while, no real-time pacing), so this is the main
# lever for getting sim time per wall-clock second up.
#
# The 30Hz arm-state publish shares this block and is NOT skipped - an
# external-action client polls it for live pose (see
# sim_cell.robot_state_publisher / the collectors' get_live_pose_rad), so
# dropping it would break external control outright.
CAMERAS_ENV_VAR = "CONVEYOR_INDEXING_CAMERAS"

logger = logging.getLogger(__name__)

# Task #56/#60 convergence-freeze investigation (2026-09-04): the
# capability-diffusion collection runs repeatedly showed a joint's live pose
# freeze at a fixed residual for the rest of a session, silently, with no
# exception anywhere. box_spawner.py already had to reach into
# RigidPrim._physics_rigid_body_view.wake_up() because PhysX doesn't wake a
# re-enabled body on its own; set_dof_position_targets has no equivalent
# public wake call, and neither Articulation nor RigidPrim in this Isaac Sim
# version expose is_sleeping(). This periodically logs DOF velocities
# (near-zero velocity with nonzero position error would indicate a genuine
# physics stall, not slow convergence) and defensively attempts the same
# private-view wake_up() every tick a command is applied, tolerating
# AttributeError/RuntimeError the same way box_spawner.py does. Diagnostic +
# candidate fix - not yet confirmed against a live repro.
_WAKE_DIAG_STATE: dict = {}


def _apply_external_arm_command(robot, pick_place, cmd) -> None:
    """Joint targets go straight to the PD drive (it holds the last target on its own); a tool
    target is planned and solved by the sim itself - see MagicAttachPickPlace.drive_external_tool_target.
    """
    if cmd.HasField("tool_target"):
        t = cmd.tool_target
        pick_place.drive_external_tool_target(
            cmd.seq,
            np.array([t.position.x, t.position.y, t.position.z], dtype=np.float64),
            np.array([t.orientation.w, t.orientation.x, t.orientation.y, t.orientation.z], dtype=np.float64),
        )
    else:
        robot.set_dof_position_targets(positions=np.asarray(cmd.joint_targets, dtype=np.float32))


def _wake_and_diagnose(articulation, arm: int, tick: int) -> None:
    # 2026-09-04 update: the wake_up() defensive fix was tested live (Task
    # #62) and REFUTED - the freeze reproduced even with wake_up() firing
    # every tick, so it's removed here (this function name is kept to avoid
    # re-threading a rename through both call sites for a diagnostic-only
    # change). Now purely diagnostic: dof_actuation_forces alongside
    # velocities distinguishes "drive genuinely fighting something" (forces
    # saturated near max effort - a real mechanical block/self-collision)
    # from "drive isn't even trying" (near-zero forces despite a large
    # position error - a control-path bug, not a physical obstruction).
    view = getattr(articulation, "_physics_articulation_view", None)
    if view is None:
        return
    if tick % 20 == 0:
        try:
            vel = view.get_dof_velocities()
            forces = view.get_dof_actuation_forces()
            logger.info(
                "WAKE_DIAG arm=%d tick=%d dof_velocities=%s dof_actuation_forces=%s",
                arm, tick, vel, forces,
            )
        except Exception as e:
            logger.warning("WAKE_DIAG arm=%d tick=%d diagnostic read failed: %s", arm, tick, e)


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
    cameras_enabled = os.environ.get(CAMERAS_ENV_VAR, "1") == "1"
    if not cameras_enabled and recorder is not None:
        raise SystemExit(
            f"{CAMERAS_ENV_VAR}=0 is incompatible with CONVEYOR_INDEXING_RECORD: that recorder's rows are "
            "image+state pairs, so with no frames captured it would silently record nothing."
        )

    initial_mode = ControlMode.EXTERNAL if os.environ.get(EXTERNAL_ACTION_ENV_VAR) == "1" else ControlMode.AUTONOMOUS
    # The 30Hz episode recorder's segmentation is meaningless under an external controller.
    validate_external_action_recording(initial_mode is ControlMode.EXTERNAL, recorder is not None)

    def _release(arm: int, held_box_path: str) -> None:
        pick_place = cell.pick_place if arm == 1 else cell.pick_place_2
        apply_suction_edge(arm, pick_place, cell.box_rigid_prims, False, held_box_path, None)

    # Who drives the cell; switchable at runtime over sim/control (see sim_cell.control).
    modes = ModeController(
        stations={
            1: (cell.loop1, layout.PICK_ZONE_INDEX, cell.pick_place),
            2: (cell.loop1, layout.PICK_ZONE_INDEX_2, cell.pick_place_2),
        },
        bridge=cell.external_command_bridge,
        release=_release,
        mode=initial_mode,
        block_external=lambda: "episode recorder (CONVEYOR_INDEXING_RECORD=1) is on" if recorder is not None else None,
    )
    control = ControlChannel(initial_mode)

    camera_role_by_serial = {spec.serial: spec.role for spec in cell.camera_specs}
    box_id_to_variant = {path: variant for variant, paths in cell.pool.paths_by_variant.items() for path in paths}
    active_box_paths: set = set()
    # 2026-09-02: box-age gate for despawn_boxes_below_floor (see below) - a
    # freshly-spawned box's collider isn't guaranteed active on its very
    # first tick(s), so it can free-fall through the belt surface briefly
    # before physics catches it. Confirmed live: on a fresh sim launch,
    # despawn_boxes_below_floor without this gate wrongly caught ~20 boxes
    # (nearly the whole initial pool) within seconds of spawn - a real,
    # observed false-positive, not a hypothetical. FLOOR_DESPAWN_MIN_AGE_S
    # gives every box a grace window to settle before it's eligible.
    box_first_seen_time: dict = {}
    FLOOR_DESPAWN_MIN_AGE_S = 3.0
    # 2026-09-02 (capability-diffusion's Stage 7c investigation): the sim's
    # own authoritative count of GENUINE truck arrivals, distinct from
    # floor- and stale-despawns - see publish_box_states's own docstring
    # for why a client can't reconstruct this from BoxStates.boxes alone.
    truck_deliveries_count = 0
    # 2026-09-02 (fourth Stage 7c root-cause pass): a box an external-action
    # client gives up on (repeated ik_unreachable/attach_failed) never lands
    # in the truck bed or falls below the floor, so it sits at proper belt
    # height forever - which permanently keeps BoxSpawner's trigger zone
    # "occupied" and starves every future wave. STALE_BOX_MAX_AGE_S is set
    # well above a real end-to-end pick+place cycle's typical duration
    # (observed ~10-20s/example when a box is found promptly) so a working
    # pipeline never triggers this - see despawn_stale_boxes's own docstring.
    # Overridable (2026-09-09): a policy evaluation runs short trials with
    # long between-trial waits, so boxes legitimately sit on the belt far
    # longer than in a collection run and were being removed right after a
    # release. Held boxes are always exempt (see stale_check_positions).
    STALE_BOX_MAX_AGE_S = float(os.environ.get("CONVEYOR_INDEXING_STALE_BOX_MAX_AGE_S", "90.0"))
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

    # sim/clock: sim time plus the recent realtime factor.
    clock_period_s = 0.1
    last_clock_sim_s = 0.0
    last_clock_wall_s = time.monotonic()

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

                pending_mode = control.take_pending()
                if pending_mode is not None and modes.apply(pending_mode):
                    control.publish_status(modes.mode, int(sim_time * 1e6))

                # Same shutdown path as SIGTERM/SIGINT below - flushes the recorder(s)
                # cleanly via the `finally` block instead of exiting mid-frame.
                if max_sim_seconds is not None and sim_time >= max_sim_seconds:
                    logger.info("sim-time cap of %.1fs reached at tick %d - shutting down", max_sim_seconds, tick)
                    shutdown_requested = True
                    continue

                # Pick-and-place runs every physics step for smooth convergence; conveyor
                # indexing runs at the coarser control rate below.
                if modes.external:
                    # Drive both arms directly from the latest externally-supplied command,
                    # bypassing MagicAttachPickPlace's phase state machine entirely (it's
                    # simply never called in this branch, so it stays dormant - no explicit
                    # pause needed). set_dof_position_targets is the exact same call
                    # TrajectoryDriver.drive_to() uses internally; the PD drive holds the
                    # last-set target for free on ticks where no new command has arrived yet.
                    cmd_arm1, cmd_arm2, cmd_conveyors = cell.external_command_bridge.latest()
                    if cmd_arm1 is not None:
                        _apply_external_arm_command(cell.robot, cell.pick_place, cmd_arm1)
                        _wake_and_diagnose(cell.robot, 1, tick)
                        modes.held[1] = apply_suction_edge(
                            1, cell.pick_place, cell.box_rigid_prims, cmd_arm1.suction, modes.held[1], pick_box_path
                        )
                        modes.holding[1] = modes.held[1] is not None
                        if mcap_recorder is not None:
                            mcap_recorder.record_arm_action_command(1, sim_time, cmd_arm1)
                    if cmd_arm2 is not None:
                        _apply_external_arm_command(cell.robot2, cell.pick_place_2, cmd_arm2)
                        _wake_and_diagnose(cell.robot2, 2, tick)
                        modes.held[2] = apply_suction_edge(
                            2, cell.pick_place_2, cell.box_rigid_prims, cmd_arm2.suction, modes.held[2],
                            pick_box_path_2,
                        )
                        modes.holding[2] = modes.held[2] is not None
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
                if mcap_recorder is not None and not modes.external:
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
                    capture_ts_us = now_us()
                    # CONVEYOR_INDEXING_CAMERAS=0 skips the render/capture/publish
                    # entirely (the dominant per-frame cost) while leaving this
                    # block's 30Hz arm-state publish below intact - see
                    # CAMERAS_ENV_VAR. `frames` stays an empty dict, which the
                    # frame-consuming recorder branch below already guards on.
                    # world.render() is NOT optional and NOT purely visual: it
                    # calls app.update(), which is what evaluates OmniGraph -
                    # and the belts are driven by OgnIsaacConveyor inside each
                    # ConveyorBeltGraph. Skipping it stops every conveyor dead
                    # (measured 2026-09-09: boxes moved +0.000m across a full
                    # 120 sim-second run, 0 truck deliveries, while the sim
                    # "ran" at 2.9x realtime doing nothing). Only the GPU
                    # capture and frame publish are genuinely optional.
                    world.render()
                    frames = cell.camera_rig.capture_all() if cameras_enabled else {}
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
                    # Images + state sampled in the same iteration = the synchronized
                    # training rows the converters expect. Skipped while annotators
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
                    # In external-action mode the phase machine is dormant; holding comes from the suction edge.
                    holding_1 = modes.held[1] is not None if modes.external else cell.pick_place.holding_box
                    holding_2 = modes.held[2] is not None if modes.external else cell.pick_place_2.holding_box
                    # One batched pose read for every box, reused below by
                    # ConveyorLineController.step, despawn_boxes_in_truck, and
                    # evaluate_pick_station - instead of each of them calling
                    # get_world_poses() per box (a GPU sync + host copy every time).
                    positions, orientations = cell.box_positions_view.get_world_poses()
                    box_positions = dict(zip(cell.box_paths_ordered, positions.numpy()))
                    # orientations dict-ified unconditionally (Stage 5b, docs/progress-
                    # tracker.md) - free, it's the same batched get_world_poses() call above
                    # regardless of recording mode, and the live box-state publish below
                    # needs it too, not just MCAP.
                    box_orientations = dict(zip(cell.box_paths_ordered, orientations.numpy()))
                    box_linear_vel = {}
                    box_angular_vel = {}
                    if mcap_recorder is not None or modes.external:
                        # Velocity needs its own PhysX sync (get_velocities()). MCAP
                        # recording always pays it. External-action mode pays it too
                        # (2026-09-09): the live BoxStates publish is the external
                        # client's only view of the belts, and without this branch it
                        # carried a hard-coded 0.0 velocity for every box (the
                        # _ZERO_VEC fallback in build_box_states) - so a client that
                        # selects "settled" boxes, or measures whether a commanded
                        # belt is actually moving, was reading a constant, not the
                        # world. That is exactly how the scripted belt check reported
                        # 0.0 m/s under a 55 % command that was in fact running.
                        linear_vel, angular_vel = cell.box_positions_view.get_velocities()
                        box_linear_vel = dict(zip(cell.box_paths_ordered, linear_vel.numpy()))
                        box_angular_vel = dict(zip(cell.box_paths_ordered, angular_vel.numpy()))

                    state_msg = telemetry.SimConveyorStates(sim_time_us=int(sim_time * 1e6))
                    commands_msg = sim_action.SimConveyorCommands()
                    cell.loop1.step(state_msg, commands_msg, box_positions)
                    cell.loop2.step(state_msg, commands_msg, box_positions)
                    if modes.external:
                        # The state machine's own decision, before the override
                        # below re-points the telemetry at the external command.
                        cell.robot_state_publisher.publish_autonomous_decision(commands_msg)
                        # Let step() run as normal first - it also drives occupancy/PackML
                        # bookkeeping that evaluate_pick_station() depends on for the arm
                        # box-lookup above, so skipping it would silently break arm control
                        # too. Only the belt command it just applied gets overridden here;
                        # OmniGraph only consumes belt attributes on the next physics
                        # substep, so this later same-tick override safely wins.
                        #
                        # state_msg's items were already populated by step() from its own
                        # autonomous decision, before this override - re-point Speed at what
                        # actually got commanded so sim/conveyor/state (the "actual
                        # state" telemetry an external observer sees) doesn't silently report
                        # stale autonomous values while external_action owns the real belt.
                        items_by_name = {item.name: item for item in state_msg.conveyors}
                        _, _, cmd_conveyors = cell.external_command_bridge.latest()
                        if cmd_conveyors is not None:
                            for cmd in cmd_conveyors.commands:
                                zone = zones_by_node_path.get(cmd.conveyor_node_path)
                                if zone is not None:
                                    zone.apply_command(cmd.run, cmd.speed)
                                    item = items_by_name.get(cmd.conveyor_node_path)
                                    if item is not None:
                                        item.run = cmd.run
                                        item.speed, item.direction = resolve_override_speed_direction(
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
                                    item.run = False
                                    item.speed, item.direction = resolve_override_speed_direction(False, 0, 0)
                    cell.robot_state_publisher.publish_conveyor_state(state_msg)
                    # Exclude any box currently held by an arm from EVERY despawn check
                    # below (truck/floor/stale) - none of the three has any way to know a
                    # box is mid-transport (still rigidly FixedJoint-attached to a wrist),
                    # only the caller can. Previously only despawn_stale_boxes's own call
                    # site filtered this (see its docstring); despawn_boxes_in_truck/
                    # despawn_boxes_below_floor had no exclusion at all, so a held box
                    # whose live position transiently satisfied either check (e.g. a
                    # real momentary collision/contact jitter while carried) got
                    # despawned - parked, hidden, disabled - out from under the arm's
                    # own FixedJoint, permanently vanishing from published box state even
                    # though the client never released it. That surfaced downstream as a
                    # spurious `dropped_in_transit` (a telemetry/despawn bug, not an
                    # actual grip failure) rather than a real physical drop.
                    held_box_paths = {p for p in modes.held.values() if p is not None}
                    truck_check_positions = {
                        path: pos for path, pos in box_positions.items() if path not in held_box_paths
                    }
                    landed_box_paths = despawn_boxes_in_truck(
                        cell.box_rigid_prims,
                        truck_check_positions,
                        layout.TRUCK_PATH,
                        cell.truck_bed_min,
                        cell.truck_bed_max,
                    )
                    truck_deliveries_count += len(landed_box_paths)
                    # 2026-09-02: a box knocked off a belt onto the floor (a real,
                    # observed failure mode under external-action control - see
                    # capability-diffusion's Stage 7c investigation) never lands in
                    # the truck bed, so despawn_boxes_in_truck alone leaves it on
                    # the floor forever, permanently shrinking the usable box pool.
                    # Mirrors the truck-despawn handling exactly, just gated on
                    # FLOOR_Z_THRESHOLD instead of the truck bed AABB - restricted
                    # to boxes past FLOOR_DESPAWN_MIN_AGE_S (see box_first_seen_time's
                    # own comment above for why: a just-spawned box's collider isn't
                    # guaranteed active on its first tick(s)).
                    floor_check_positions = {
                        path: pos
                        for path, pos in box_positions.items()
                        if path not in held_box_paths
                        and sim_time - box_first_seen_time.get(path, sim_time) >= FLOOR_DESPAWN_MIN_AGE_S
                    }
                    grounded_box_paths = despawn_boxes_below_floor(
                        cell.box_rigid_prims,
                        floor_check_positions,
                        settings.FLOOR_Z_THRESHOLD,
                    )
                    # 2026-09-10: a box already falling off a belt anywhere but
                    # into the truck vanishes now, not after the floor bounce
                    # (see despawn_boxes_off_belt). Same age gate and
                    # held-box exclusion as the floor check; a box the floor
                    # check just caught is not checked again.
                    off_belt_box_paths = despawn_boxes_off_belt(
                        cell.box_rigid_prims,
                        {p: pos for p, pos in floor_check_positions.items() if p not in grounded_box_paths},
                        settings.OFF_BELT_Z_THRESHOLD,
                        cell.truck_bed_min,
                        cell.truck_bed_max,
                        settings.OFF_BELT_TRUCK_XY_MARGIN_M,
                    )
                    grounded_box_paths = grounded_box_paths + off_belt_box_paths
                    stale_check_positions = {
                        path: pos for path, pos in box_positions.items() if path not in held_box_paths
                    }
                    stale_box_ages_s = {
                        path: sim_time - box_first_seen_time.get(path, sim_time) for path in stale_check_positions
                    }
                    stale_box_paths = despawn_stale_boxes(
                        stale_check_positions,
                        stale_box_ages_s,
                        STALE_BOX_MAX_AGE_S,
                        cell.box_rigid_prims,
                    )
                    despawned_box_paths = landed_box_paths + grounded_box_paths + stale_box_paths
                    for path in despawned_box_paths:
                        if mcap_recorder is not None:
                            mcap_recorder.record_box_event(
                                sim_time,
                                BOX_EVENT_DESPAWNED,
                                path,
                                box_id_to_variant.get(path, ""),
                                tuple(box_positions[path]),
                                tuple(box_orientations[path]),
                            )
                        active_box_paths.discard(path)
                        box_first_seen_time.pop(path, None)
                    # Recycle truck-landed and floor-grounded boxes back into the
                    # pool, then spawn a new wave if ConveyorTrack (loop1 zone 0)
                    # just emptied out - reuses the occupancy loop1.step already
                    # computed this tick.
                    cell.spawner.release(despawned_box_paths)
                    # Scripted placement (2026-09-10, decision trials): drain
                    # this tick's box commands from the external client - pause
                    # or resume the automatic waves, clear every box not held,
                    # or place one box at an exact world pose. Cleared boxes get
                    # a DESPAWNED event and go back to the pool; placed boxes
                    # join `spawned` below and get the same SPAWNED bookkeeping
                    # a wave does.
                    spawned = []
                    if cell.external_command_bridge is not None:
                        for cmd in cell.external_command_bridge.drain_box_commands():
                            op = cmd.get("op")
                            if op == "auto":
                                cell.spawner.auto_waves = bool(cmd.get("enabled", True))
                                logger.info("box command seq=%s: automatic waves %s", cmd.get("seq"), "on" if cell.spawner.auto_waves else "off")
                            elif op == "clear":
                                to_clear = [p for p in active_box_paths if p not in held_box_paths]
                                for path in cell.spawner.despawn(to_clear):
                                    if mcap_recorder is not None:
                                        mcap_recorder.record_box_event(
                                            sim_time, BOX_EVENT_DESPAWNED, path, box_id_to_variant.get(path, ""),
                                            tuple(box_positions[path]), tuple(box_orientations[path]),
                                        )
                                    active_box_paths.discard(path)
                                    box_first_seen_time.pop(path, None)
                                logger.info("box command seq=%s: cleared %d box(es)", cmd.get("seq"), len(to_clear))
                            elif op == "spawn":
                                spawned.extend(cell.spawner.spawn_at(
                                    sim_time, float(cmd["x"]), float(cmd["y"]), float(cmd.get("yaw", 0.0)), cmd.get("variant"),
                                ))
                                logger.info("box command seq=%s: spawn at (%.2f, %.2f) -> %s", cmd.get("seq"), float(cmd["x"]), float(cmd["y"]),
                                            [s[0] for s in spawned])
                    spawned = spawned + cell.spawner.update(sim_time, cell.loop1.occupied[0])
                    for path, variant, position, quat_wxyz in spawned:
                        if mcap_recorder is not None:
                            mcap_recorder.record_box_event(
                                sim_time, BOX_EVENT_SPAWNED, path, variant, position, quat_wxyz
                            )
                        active_box_paths.add(path)
                        box_first_seen_time[path] = sim_time
                        # box_positions/box_orientations were read at the top of this
                        # tick, before this box was teleported onto the belt just now -
                        # without this override, this tick's BoxStates (recorded AND
                        # live-published, Stage 5b) would show the box at its stale parked
                        # pose (POOL_PARK_ORIGIN). Zero velocity is a reasonable
                        # approximation for "just placed, not yet fallen". Unconditional now
                        # - the live publish below needs it too, not just MCAP.
                        box_positions[path] = position
                        box_orientations[path] = quat_wxyz
                        box_linear_vel[path] = (0.0, 0.0, 0.0)
                        box_angular_vel[path] = (0.0, 0.0, 0.0)
                    latest_plc_bytes = state_msg.SerializeToString()

                    # holding_1/2 + held_by_arm must come from held_box_path_1/2 in
                    # external_action mode, NOT cell.pick_place(_2).holding_box/
                    # held_box_path - those are frozen (forward() never runs there),
                    # so every policy-run KPI would otherwise silently read "never
                    # holding". See sim_cell.recording.resolve_arm_telemetry.
                    # Unconditional now (Stage 5b) - the live box-state publish below
                    # needs held_by_arm too, not just MCAP.
                    holding_1, holding_2, held_by_arm = resolve_arm_telemetry(
                        modes.external,
                        modes.held[1],
                        modes.held[2],
                        cell.pick_place.holding_box,
                        cell.pick_place_2.holding_box,
                        cell.pick_place.held_box_path,
                        cell.pick_place_2.held_box_path,
                    )
                    box_states = build_box_states(
                        active_box_paths,
                        box_positions,
                        box_orientations,
                        box_linear_vel,
                        box_angular_vel,
                        box_id_to_variant,
                        held_by_arm,
                    )
                    cell.robot_state_publisher.publish_box_states(sim_time, box_states, truck_deliveries_count)
                    if sim_time - last_clock_sim_s >= clock_period_s:
                        wall_now = time.monotonic()
                        realtime_factor = (sim_time - last_clock_sim_s) / max(wall_now - last_clock_wall_s, 1e-9)
                        cell.robot_state_publisher.publish_clock(int(sim_time * 1e6), realtime_factor)
                        control.publish_status(modes.mode, int(sim_time * 1e6), realtime_factor=realtime_factor)
                        last_clock_sim_s, last_clock_wall_s = sim_time, wall_now
                    # Live flange pose, both arms, every tick, any control mode -
                    # the same read the MCAP block below records, now also on
                    # the wire so an external policy can measure the exact
                    # distance the attach gate measures (2026-09-09).
                    tool_pos_1, tool_quat_1 = cell.pick_place.tool_world_pose()
                    tool_pos_2, tool_quat_2 = cell.pick_place_2.tool_world_pose()
                    cell.robot_state_publisher.publish_tool_pose(1, sim_time, tool_pos_1, tool_quat_1)
                    cell.robot_state_publisher.publish_tool_pose(2, sim_time, tool_pos_2, tool_quat_2)
                    sim_time_us = int(sim_time * 1e6)
                    for arm, robot, holding, tool_pos, tool_quat in (
                        (1, cell.robot, holding_1, tool_pos_1, tool_quat_1),
                        (2, cell.robot2, holding_2, tool_pos_2, tool_quat_2),
                    ):
                        joint_pos = robot.get_dof_positions().numpy()[0]
                        joint_vel = robot.get_dof_velocities().numpy()[0]
                        cell.robot_state_publisher.publish_arm_state(
                            arm, joint_pos, joint_vel, holding, tool_pos, tool_quat, sim_time_us
                        )
                        if mcap_recorder is not None:
                            mcap_recorder.record_arm_state(arm, sim_time, joint_pos, joint_vel, holding, tool_pos, tool_quat)

                    if mcap_recorder is not None:
                        # Same object, not re-parsed from latest_plc_bytes - state_msg
                        # is freshly built this tick and never mutated again.
                        mcap_recorder.record_conveyor_states(sim_time, state_msg)
                        mcap_recorder.record_box_states(sim_time, box_states, truck_deliveries_count)
                        # Independent of control mode (unlike holding_1/2 above) - the
                        # tool_prim GeomPrim tracks the arm's actual physical
                        # wrist_3_link/flange regardless of whether forward() runs.
                        # Recorded at the same 120Hz cadence as BoxStates/
                        # PositionStatus so FK is never needed downstream (see
                        # pick_and_place.controller.MagicAttachPickPlace.tool_world_pose).
                        # tool_pos/quat read once above, for the live publish.
                        mcap_recorder.record_tool_pose(1, sim_time, tuple(tool_pos_1), tuple(tool_quat_1))
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
        control.close()
        if cell.external_command_bridge is not None:
            cell.external_command_bridge.close()
        simulation_app.close()
