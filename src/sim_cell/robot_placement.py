"""Derives station 2's robot placement from actual zone geometry rather than a
hardcoded position - ConveyorTrack_02/_10 aren't guaranteed to line up in X.
Pure geometry over each zone's `bbox_center`/`bbox_half_extent` - no USD calls,
so unit-testable without Isaac Sim.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

from conveyor_indexing.belt_geometry import compute_belt_bounds

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
