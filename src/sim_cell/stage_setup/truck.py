"""Truck collision, world bounds, and box despawn-on-landing."""

from __future__ import annotations

import logging

from pxr import Gf, Usd, UsdGeom, UsdPhysics

logger = logging.getLogger(__name__)

_TRUCK_BODY_SUBPATH = "sm_steelboxtruck_a01_body_01"

# Parking spot for despawned boxes, far from any geometry so its AABB never
# matches truck_bed_min/max again.
DESPAWNED_BOX_PARK_POSITION = (100.0, 100.0, -100.0)


def apply_truck_collision(stage: Usd.Stage, truck_path: str) -> None:
    """Add a static collider to the truck body mesh (a pure visual payload with no
    physics) so falling boxes land in the bed instead of clipping through.
    """
    body_prim = stage.GetPrimAtPath(f"{truck_path}/{_TRUCK_BODY_SUBPATH}")
    if not body_prim.IsValid():
        raise RuntimeError(f"Expected truck body mesh not found under {truck_path}")
    UsdPhysics.CollisionAPI.Apply(body_prim)
    logger.info("added static collision to %s", body_prim.GetPath())


def truck_body_world_bounds(stage: Usd.Stage, truck_path: str) -> tuple:
    """World-space AABB of the truck body mesh, computed once and reused every tick
    to test whether a box has landed inside it (see despawn_boxes_in_truck).
    """
    body_prim = stage.GetPrimAtPath(f"{truck_path}/{_TRUCK_BODY_SUBPATH}")
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    aligned_range = bbox_cache.ComputeWorldBound(body_prim).ComputeAlignedRange()
    return aligned_range.GetMin(), aligned_range.GetMax()


def despawn_boxes_in_truck(
    box_rigid_prims: dict,
    box_positions: dict,
    truck_path: str,
    truck_bed_min: "Gf.Vec3d",
    truck_bed_max: "Gf.Vec3d",
) -> None:
    """Disable, hide, and park any box that's fallen inside the truck bed AABB.

    `box_positions` is {path: (x, y, z)}, precomputed once per control tick from a
    single batched RigidPrim read (see sim_cell.runner) rather than queried per-box
    here; `box_rigid_prims` is still needed for the per-box writes below.

    Deleting the prim outright crashes the app: RigidPrim's shared PhysX tensor
    view gets invalidated for every other tracked box too.
    """
    landed_paths = []
    for box_path, (x, y, z) in box_positions.items():
        in_x = truck_bed_min[0] <= x <= truck_bed_max[0]
        in_y = truck_bed_min[1] <= y <= truck_bed_max[1]
        if in_x and in_y and z <= truck_bed_max[2]:
            landed_paths.append(box_path)
    for box_path in landed_paths:
        rigid_prim = box_rigid_prims[box_path]
        rigid_prim.set_enabled_rigid_bodies([False])
        rigid_prim.set_visibilities([False])
        rigid_prim.set_world_poses(positions=[DESPAWNED_BOX_PARK_POSITION])
        logger.info("despawned %s - landed in %s", box_path, truck_path)
