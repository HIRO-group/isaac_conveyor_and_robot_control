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
) -> list:
    """Disable, hide, and park any box that's fallen inside the truck bed AABB.
    Returns the paths that were despawned this call, so sim_cell.box_spawner can
    recycle them back into its pool (see BoxSpawner.release).

    `box_positions` is {path: (x, y, z)}, precomputed once per control tick from a
    single batched RigidPrim read (see sim_cell.runner) rather than queried per-box
    here; `box_rigid_prims` is still needed for the per-box writes below.

    There is no supported way to actually remove the prim from the simulation
    while the timeline runs - this parked/disabled/hidden state is the real
    despawn. Deleting the prim (or Usd.Prim.SetActive(False), which Isaac's own
    USD notice listener treats identically) destroys its PhysX actor, and the
    tensors backend responds by invalidating the single process-wide
    SimulationView every tracked RigidPrim, both UR20 Articulations, and
    cuMotion depend on - not just the other boxes. A disabled body still
    participates in scene queries (occupancy's overlap_box would still see it),
    which is why the teleport below is load-bearing and not just cosmetic.
    """
    landed_paths = []
    for box_path, (x, y, z) in box_positions.items():
        in_x = truck_bed_min[0] <= x <= truck_bed_max[0]
        in_y = truck_bed_min[1] <= y <= truck_bed_max[1]
        in_z = truck_bed_min[2] <= z <= truck_bed_max[2]
        if in_x and in_y and in_z:
            landed_paths.append(box_path)
    for box_path in landed_paths:
        rigid_prim = box_rigid_prims[box_path]
        rigid_prim.set_enabled_rigid_bodies([False])
        rigid_prim.set_visibilities([False])
        rigid_prim.set_world_poses(positions=[DESPAWNED_BOX_PARK_POSITION])
        logger.info("despawned %s - landed in %s", box_path, truck_path)
    return landed_paths
