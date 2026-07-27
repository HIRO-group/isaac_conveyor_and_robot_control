"""Main control loop: steps physics, drives both pick-and-place controllers,
runs conveyor indexing at the control rate, logs, and despawns landed boxes.
"""

from __future__ import annotations

import logging
import signal

from conveyor_indexing.protos import plc, sim_action
from sim_cell import layout, settings
from sim_cell.cell import build_cell
from sim_cell.debug import dump_tick_debug
from sim_cell.pick_dispatch import evaluate_pick_station
from sim_cell.stage_setup import prepare_stage
from sim_cell.stage_setup.truck import despawn_boxes_in_truck

logger = logging.getLogger(__name__)


def run(simulation_app) -> None:
    stage_prep = prepare_stage()
    cell = build_cell(stage_prep)
    logger.info("pick/place controllers ready, entering main loop")

    control_period_s = 1.0 / settings.CONTROL_HZ
    last_control_time = 0.0
    sim_time = 0.0
    render_count = 0
    tick = 0
    pick_ready = False
    pick_box_path = None
    pick_ready_2 = False
    pick_box_path_2 = None

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
                world.step(render=True)
                sim_time += world.get_physics_dt()

                # Pick-and-place runs every physics step for smooth convergence; conveyor
                # indexing runs at the coarser control rate below.
                cell.pick_place.forward(pick_ready, pick_box_path)
                cell.pick_place_2.forward(pick_ready_2, pick_box_path_2)

                if sim_time - last_control_time >= control_period_s:
                    state_msg = plc.StateConveyors()
                    commands_msg = sim_action.SimConveyorCommands()
                    cell.loop1.step(state_msg, commands_msg)
                    cell.loop2.step(state_msg, commands_msg)
                    despawn_boxes_in_truck(cell.box_rigid_prims, layout.TRUCK_PATH, cell.truck_bed_min, cell.truck_bed_max)
                    cell.tick_logger.log_tick(
                        tick=tick,
                        sim_time_s=sim_time,
                        plc_state_conveyors=state_msg.SerializeToString(),
                        conveyor_commands=commands_msg.SerializeToString(),
                    )
                    # Only "ready" once the pick zone has settled into holding (IDLE +
                    # occupied); identify which box is actually there rather than assuming a fixed one.
                    pick_ready, pick_box_path = evaluate_pick_station(
                        cell.loop1.zones[layout.PICK_ZONE_INDEX],
                        cell.loop1.machine_states[layout.PICK_ZONE_INDEX],
                        cell.box_rigid_prims,
                        cell.robot_xy,
                    )
                    pick_ready_2, pick_box_path_2 = evaluate_pick_station(
                        cell.loop1.zones[layout.PICK_ZONE_INDEX_2],
                        cell.loop1.machine_states[layout.PICK_ZONE_INDEX_2],
                        cell.box_rigid_prims,
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
        cell.tick_logger.close()
        simulation_app.close()
