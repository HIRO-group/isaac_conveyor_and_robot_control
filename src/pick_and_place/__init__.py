"""UR20 + cuMotion pick-and-place: magic-attach phase state machine moving a
box from a pick-zone conveyor to a place-zone conveyor on another loop.

`compat` (the NumPy `reshape(shape=...)` shim) must be imported here, first,
before anything in this package - or anywhere else - imports `cumotion`.
"""

from __future__ import annotations

from pick_and_place import compat  # noqa: F401  (import order matters - see module docstring)
from pick_and_place.controller import MagicAttachPickPlace
from pick_and_place.robot_setup import create_pedestal_and_robot
from pick_and_place.selection import PICK_MAX_REACH_M, rank_pick_zone_hit_paths
from pick_and_place.ur20 import UR20_PRE_PLACE_JOINT_POSITIONS_AWAY

__all__ = [
    "MagicAttachPickPlace",
    "create_pedestal_and_robot",
    "rank_pick_zone_hit_paths",
    "PICK_MAX_REACH_M",
    "UR20_PRE_PLACE_JOINT_POSITIONS_AWAY",
]
