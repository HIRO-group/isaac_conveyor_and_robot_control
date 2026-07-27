"""Belt-top world-space geometry, shared by occupancy queries and direction
correction (see ``conveyor_indexing.occupancy`` / ``conveyor_indexing.directions``).
"""

from __future__ import annotations

from dataclasses import dataclass

from pxr import Usd, UsdGeom

# Belt mesh has near-zero Z extent; query a column above it instead so
# boxes resting on top are actually caught by the occupancy check.
# Meters; tallest known box is ~0.42m.
OCCUPANCY_QUERY_HALF_HEIGHT = 0.5


@dataclass
class BeltBounds:
    """World-space belt-top bounding box for one conveyor zone."""

    bbox_center: list[float]
    bbox_half_extent: list[float]
    belt_top_z: float


def compute_belt_bounds(belt_prim: Usd.Prim) -> BeltBounds:
    """Compute belt_prim's world AABB, projected up into a column above the
    belt top (see ``OCCUPANCY_QUERY_HALF_HEIGHT``) rather than the belt
    mesh's own near-zero-height bounds.
    """
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    world_bound = bbox_cache.ComputeWorldBound(belt_prim)
    aligned_range = world_bound.ComputeAlignedRange()
    size = aligned_range.GetSize()
    belt_top_z = aligned_range.GetMax()[2]
    return BeltBounds(
        bbox_center=[
            aligned_range.GetMidpoint()[0],
            aligned_range.GetMidpoint()[1],
            belt_top_z + OCCUPANCY_QUERY_HALF_HEIGHT,
        ],
        bbox_half_extent=[size[0] * 0.5, size[1] * 0.5, OCCUPANCY_QUERY_HALF_HEIGHT],
        belt_top_z=belt_top_z,
    )
