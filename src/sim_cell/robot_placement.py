"""Derives station 2's robot placement from actual zone geometry rather than a
hardcoded position - ConveyorTrack_02/_10 aren't guaranteed to line up in X.

`derive_station_2_geometry` is pure geometry over each zone's
`bbox_center`/`bbox_half_extent` - no USD calls, so unit-testable without
Isaac Sim. `belt_top_z`/`zone_geometry_inputs` do call into USD
(`UsdGeom.BBoxCache` via `conveyor_indexing.belt_geometry.compute_belt_bounds`)
- testable with just `pxr` (e.g. the `usd-core` PyPI package) against a real
or in-memory stage, without needing full Isaac Sim/omni/carb.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

from conveyor_indexing.belt_geometry import compute_belt_bounds
from conveyor_indexing.state_machine import HOLD_ZONE_STOP_FRACTION
from sim_cell.recording import ZoneGeometryInput

logger = logging.getLogger(__name__)


@dataclass
class StationGeometry:
    robot_position: tuple
    place_xy: tuple


def derive_station_2_geometry(pick_zone_2, place_zone_2) -> StationGeometry:
    """Reach-balanced position for the second robot, derived from actual zone
    geometry rather than hardcoded - ConveyorTrack_02/_10 aren't guaranteed to
    line up in X.
    """
    robot_2_x = (pick_zone_2.bbox_center[0] + place_zone_2.bbox_center[0]) / 2.0
    robot_2_y = (
        pick_zone_2.bbox_center[1] + pick_zone_2.bbox_half_extent[1]
        + place_zone_2.bbox_center[1] - place_zone_2.bbox_half_extent[1]
    ) / 2.0
    robot_position = (robot_2_x, robot_2_y, 0.0)
    place_xy = (place_zone_2.bbox_center[0], place_zone_2.bbox_center[1])
    logger.debug(
        "robot 2 geometry: robot_position=%s place_xy=%s reach_to_pick=%.3fm reach_to_place=%.3fm (UR20 spec ~1.75m)",
        robot_position, place_xy,
        math.dist((robot_2_x, robot_2_y), pick_zone_2.bbox_center[:2]),
        math.dist((robot_2_x, robot_2_y), place_zone_2.bbox_center[:2]),
    )
    return StationGeometry(robot_position=robot_position, place_xy=place_xy)


def belt_top_z(belt_prim) -> float:
    """Place target Z depends on box height, computed per-cycle inside
    MagicAttachPickPlace; only the belt-top Z is fixed here.
    """
    return compute_belt_bounds(belt_prim).belt_top_z


def zone_geometry_inputs(
    zones: list, speed_m_per_s: float, line_id: int, hold_zone_indices: frozenset
) -> list[ZoneGeometryInput]:
    """RunMetadata.zone_geometry entries for one loop's zones, in belt order -
    see sim_cell.recording.ZoneGeometryInput. Shared by sim_cell.cell (live
    per-run RunMetadata) and scripts/export_zone_geometry.py (one-time
    committed snapshot) so both read zone geometry the same way.

    `hold_zone_indices` here is the metadata notion of "hold zone"
    (sim_cell.layout's PICK_ZONE_INDEX/PICK_ZONE_INDEX_2/PLACE_ZONE_INDEX/
    PLACE_ZONE_INDEX_2), not ConveyorLineController.hold_zone_indices (which
    is narrower - only loop1's pick stations need robot-busy overflow
    handling; loop2's place zones don't, but are still "hold zones" for
    metadata/eval purposes).

    `speed_m_per_s` is this line's configured run speed while running (see
    conveyor_indexing.zone.ConveyorZone.ZONE_RUN_VELOCITY * this line's
    run_speed_pct / 100) - passed in already computed rather than derived
    here, so this module (otherwise pxr-only) doesn't need
    conveyor_indexing.zone's carb dependency.
    """
    inputs = []
    for i, zone in enumerate(zones):
        is_hold_zone = i in hold_zone_indices
        travel = zone.world_travel_direction
        travel_tuple = (float(travel[0]), float(travel[1]), float(travel[2])) if travel is not None else (0.0, 0.0, 0.0)
        inputs.append(
            ZoneGeometryInput(
                node_path=zone.node_path,
                bbox_center=tuple(float(v) for v in zone.bbox_center),
                bbox_half_extent=tuple(float(v) for v in zone.bbox_half_extent),
                belt_top_z=belt_top_z(zone.belt_prim),
                travel_direction=travel_tuple,
                stop_fraction=HOLD_ZONE_STOP_FRACTION if is_hold_zone else 0.0,
                speed_m_per_s=speed_m_per_s,
                is_hold_zone=is_hold_zone,
                line_id=line_id,
            )
        )
    return inputs
