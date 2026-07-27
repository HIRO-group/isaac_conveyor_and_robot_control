"""Periodic diagnostic dumps for the main control loop - pick-zone centering,
zone-0 direction/enabled/velocity, and every box's position. Emitted once
every 3 control ticks, and only when this logger is at DEBUG.
"""

from __future__ import annotations

import logging

from sim_cell import layout

logger = logging.getLogger(__name__)


def dump_tick_debug(
    cell,
    tick: int,
    sim_time: float,
    pick_ready: bool,
    pick_box_path: str | None,
    pick_ready_2: bool,
    pick_box_path_2: str | None,
) -> None:
    if tick % 3 != 0 or not logger.isEnabledFor(logging.DEBUG):
        return

    if pick_box_path is not None:
        pick_zone = cell.loop1.zones[layout.PICK_ZONE_INDEX]
        box_pos, _ = cell.box_rigid_prims[pick_box_path].get_world_poses()
        box_x = box_pos.numpy()[0][0]
        logger.debug(
            "pick zone centering: box=%s box_x=%.3f target_x=%.3f machine=%s",
            pick_box_path, box_x, pick_zone.bbox_center[0], cell.loop1.machine_states[layout.PICK_ZONE_INDEX],
        )

    logger.debug(
        "tick=%d sim_time=%.2f pick_phase=%s pick_ready=%s pick_phase_2=%s pick_ready_2=%s",
        tick, sim_time, cell.pick_place.phase_name, pick_ready, cell.pick_place_2.phase_name, pick_ready_2,
    )
    zone0 = cell.loop1.zones[0]
    z0_dir = zone0.direction_attr.Get()
    z0_enabled = zone0.node_prim.GetAttribute("inputs:enabled").Get()
    z0_vel = zone0.velocity_var_attr.Get()
    logger.debug(
        "ConveyorTrack (zone0) direction=%s enabled=%s velvar=%s machine=%s",
        tuple(z0_dir), z0_enabled, z0_vel, cell.loop1.machine_states[0],
    )
    for box_path, rigid_prim in cell.box_rigid_prims.items():
        pos, _ = rigid_prim.get_world_poses()
        p = pos.numpy().tolist()[0]
        logger.debug("%s pos=(%.3f,%.3f,%.3f)", box_path, p[0], p[1], p[2])
