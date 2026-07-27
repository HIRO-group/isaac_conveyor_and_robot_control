"""Privileged pose/geometry queries against a box prim - no perception, just
ground-truth USD/PhysX state.
"""

from __future__ import annotations

import numpy as np
from pxr import Usd, UsdGeom


def measure_box_half_height(box_path: str) -> float:
    """Privileged, one-time query of a box prim's half-height via its world bbox."""
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(box_path)
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    aligned_range = bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange()
    return aligned_range.GetSize()[2] / 2.0


def box_top_center(rigid_prim, half_height: float) -> np.ndarray:
    """The box's current world-space top-face center.

    get_world_poses() returns the box's BOTTOM-face origin, not its center, so the
    full height (not half) must be added to reach the top - same convention
    MagicAttachPickPlace.place_position uses.
    """
    position, _ = rigid_prim.get_world_poses()
    bottom_center = position.numpy()[0]
    return bottom_center + np.array([0.0, 0.0, 2.0 * half_height])
