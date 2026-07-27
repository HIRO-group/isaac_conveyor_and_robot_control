"""Box discovery, pre-physics overlap resolution, and rigid-body physics
application for the pre-authored CubeBox_* pallet.
"""

from __future__ import annotations

import logging

from pxr import PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics

logger = logging.getLogger(__name__)

BOX_DENSITY_KG_PER_M3 = 150.0  # plausible ballpark for a packed shipping box; not measured

# Caps how fast PhysX may push overlapping bodies apart per step. Some boxes start out
# interpenetrating (pallet Z-layer spacing is tighter than the tallest box variant);
# without this cap PhysX's depenetration impulse flung boxes clean off the belt.
BOX_MAX_DEPENETRATION_VELOCITY = 0.5  # m/s

# Iteration cap for resolve_box_overlaps's separation passes, and the extra
# gap (beyond just-touching) left between two boxes once separated - small
# enough to be visually unnoticeable, large enough that the two don't start
# back in (near-)contact and immediately re-trigger PhysX's own depenetration
# push once physics starts.
BOX_OVERLAP_RESOLVE_MAX_PASSES = 8
BOX_OVERLAP_CLEARANCE = 0.002  # meters


def discover_box_prim_paths(stage: Usd.Stage, name_prefix: str) -> list:
    """Find every pre-authored CubeBox_* top-level prim (direct child of the stage
    root or /World) - excludes child meshes that share the naming prefix.
    """
    paths = []
    for prim in stage.Traverse():
        if not prim.GetName().startswith(name_prefix):
            continue
        parent = prim.GetParent()
        if parent.IsPseudoRoot() or parent.GetPath() == Sdf.Path("/World"):
            paths.append(str(prim.GetPath()))
    return sorted(paths)


def resolve_box_overlaps(stage: Usd.Stage, box_paths: list) -> None:
    """Nudge apart any box prims whose world AABBs actually overlap, before physics
    ever runs - the pre-authored pallet has some pairs uncomfortably tight (~2cm),
    which otherwise makes PhysX's depenetration fling boxes off the belt on tick one.
    Pushes along the smallest-overlap axis (a standard MTV separation), to a fixed pass cap.
    """

    def _translate_op(prim: Usd.Prim) -> UsdGeom.XformOp:
        for op in UsdGeom.Xformable(prim).GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                return op
        raise RuntimeError(f"{prim.GetPath()} has no xformOp:translate")

    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    prims = [stage.GetPrimAtPath(path) for path in box_paths]
    translate_ops = [_translate_op(prim) for prim in prims]

    total_nudges = 0
    for _ in range(BOX_OVERLAP_RESOLVE_MAX_PASSES):
        bounds = [bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange() for prim in prims]
        any_overlap = False
        for i in range(len(prims)):
            for j in range(i + 1, len(prims)):
                min_i, max_i = bounds[i].GetMin(), bounds[i].GetMax()
                min_j, max_j = bounds[j].GetMin(), bounds[j].GetMax()
                overlap = [min(max_i[axis], max_j[axis]) - max(min_i[axis], min_j[axis]) for axis in range(3)]
                if not all(depth > 0.0 for depth in overlap):
                    continue
                any_overlap = True
                axis = min(range(3), key=lambda a: overlap[a])
                push = overlap[axis] / 2.0 + BOX_OVERLAP_CLEARANCE
                sign = 1.0 if (min_j[axis] + max_j[axis]) >= (min_i[axis] + max_i[axis]) else -1.0
                delta = [0.0, 0.0, 0.0]
                delta[axis] = sign * push
                current = translate_ops[j].Get()
                translate_ops[j].Set(type(current)(current[0] + delta[0], current[1] + delta[1], current[2] + delta[2]))
                bounds[j] = bbox_cache.ComputeWorldBound(prims[j]).ComputeAlignedRange()
                total_nudges += 1
        if not any_overlap:
            break
    logger.info("resolved box overlaps with %d nudge(s)", total_nudges)


def apply_box_physics(stage: Usd.Stage, box_paths: list) -> None:
    """Add RigidBodyAPI + convex-hull CollisionAPI + mass to every box prim (ships
    with no physics schemas at all). Convex-hull since these are dynamic bodies in
    contact with each other and the moving belt.
    """
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    for path in box_paths:
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            raise RuntimeError(f"Expected box prim not found at {path}")
        UsdPhysics.RigidBodyAPI.Apply(prim)
        UsdPhysics.CollisionAPI.Apply(prim)
        UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr().Set("convexHull")
        PhysxSchema.PhysxRigidBodyAPI.Apply(prim).CreateMaxDepenetrationVelocityAttr().Set(
            BOX_MAX_DEPENETRATION_VELOCITY
        )

        size = bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange().GetSize()
        mass_kg = size[0] * size[1] * size[2] * BOX_DENSITY_KG_PER_M3
        UsdPhysics.MassAPI.Apply(prim).CreateMassAttr().Set(mass_kg)
    logger.info("applied rigid-body physics to %d boxes", len(box_paths))
