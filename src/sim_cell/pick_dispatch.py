"""Per-station pick readiness + candidate box selection."""

from __future__ import annotations

from conveyor_indexing.protos import plc
from pick_and_place import rank_pick_zone_hit_paths

Machine = plc.ConveyorStateMachineCode


def evaluate_pick_station(zone, machine_state, box_rigid_prims: dict, robot_xy: tuple) -> tuple:
    """Only "ready" once the pick zone has settled into holding (IDLE + occupied);
    identify which box is actually there rather than assuming a fixed one.

    Returns (pick_ready, pick_box_path).
    """
    hits = zone.get_occupying_prim_paths()
    ready = bool(hits) and machine_state == Machine.CONVEYOR_STATE_MACHINE_IDLE
    box_path = rank_pick_zone_hit_paths(hits, box_rigid_prims, robot_xy)
    return ready, box_path
