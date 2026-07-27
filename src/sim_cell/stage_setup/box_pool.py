"""Authors a fixed-size pool of parked CubeBox prims into the stage at prep
time (`5_conv_env_empty.usd` ships none). Runtime spawning
(`sim_cell.box_spawner`) teleports pool prims onto the belt and re-enables
them rather than creating/deleting prims mid-run - deleting (or
`SetActive(False)`-ing) a tensorized rigid prim invalidates the shared PhysX
SimulationView for every other tracked box, both UR20 Articulations, and
cuMotion (see stage_setup.truck's despawn comment).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from pxr import Gf, Usd, UsdGeom

from sim_cell.asset_paths import LOCAL_ASSET_ROOT, TOP_LEVEL_URLS, local_path_for

logger = logging.getLogger(__name__)


def _asset_url(name_fragment: str) -> str:
    matches = [url for url in TOP_LEVEL_URLS if name_fragment in url]
    assert len(matches) == 1, f"expected exactly one asset URL containing {name_fragment!r}, got {matches}"
    return matches[0]


# Variant key -> box asset URL. Only the two sizes to randomize between - the
# 42cm variant present in the original pallet is deliberately excluded.
POOL_VARIANT_URLS = {
    "CubeBox_A03_21cm_PR_NVD": _asset_url("CubeBox_A03_21cm_PR_NVD_01.usd"),
    "CubeBox_A04_26cm_PR_NVD": _asset_url("CubeBox_A04_26cm_PR_NVD_01.usd"),
}

# 24 boxes total. Worst-case in-flight census across both loops: up to 5 on
# ConveyorTrack (a wave never spawns onto an occupied belt) + boxes queued
# across the two downstream hold zones + one per robot gripper + boxes
# mid-fall/awaiting truck despawn - comfortably under the original scene's
# proven 32-box scale for physics/render load.
POOL_COUNT_PER_VARIANT = 12

# Distinct, far-apart park slots - not load-bearing (nothing overlaps way out
# here), just keeps intent obvious next to stage_setup.truck's park position.
POOL_PARK_ORIGIN = (100.0, 100.0, -100.0)
POOL_PARK_SPACING_M = 1.0

# CubeBox_* assets are authored in centimeters; the original pallet's box
# prims carry this same value as `xformOp:scale:unitsResolve` (see
# pick_and_place/attachment.py's non-unity-scale comment) to bring them into
# the stage's meters-based units.
UNITS_RESOLVE_SCALE = (0.01, 0.01, 0.01)


@dataclass
class BoxPool:
    """Parked pool prim paths and per-variant world-space half-extents."""

    paths_by_variant: dict
    half_extents_by_variant: dict

    def all_paths(self) -> list:
        return [path for paths in self.paths_by_variant.values() for path in paths]


def _payload_asset_path(url: str) -> str:
    # Mirrors localize_asset_references' fallback contract: prefer the local
    # mirror if one exists, otherwise fall back to fetching over the network.
    if os.path.isdir(LOCAL_ASSET_ROOT):
        return local_path_for(url)
    return url


def author_box_pool(stage: Usd.Stage) -> BoxPool:
    """Define POOL_COUNT_PER_VARIANT prims per variant in POOL_VARIANT_URLS, each
    payloaded to its box asset and parked at a distinct, far-away slot. Named and
    parented exactly like the original pallet's boxes (CubeBox_* under /World) so
    stage_setup.boxes.discover_box_prim_paths picks them up unchanged.
    """
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    paths_by_variant: dict = {}
    half_extents_by_variant: dict = {}

    slot = 0
    for variant, url in POOL_VARIANT_URLS.items():
        payload_path = _payload_asset_path(url)
        paths = []
        for i in range(POOL_COUNT_PER_VARIANT):
            prim_path = f"/World/{variant}_P{i:02d}"
            park_position = Gf.Vec3d(
                POOL_PARK_ORIGIN[0] + slot * POOL_PARK_SPACING_M,
                POOL_PARK_ORIGIN[1],
                POOL_PARK_ORIGIN[2],
            )
            slot += 1

            prim = stage.DefinePrim(prim_path, "Xform")
            prim.GetPayloads().AddPayload(payload_path)
            xformable = UsdGeom.Xformable(prim)
            xformable.ClearXformOpOrder()
            xformable.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(park_position)
            xformable.AddOrientOp().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            xformable.AddScaleOp().Set(Gf.Vec3f(1.0, 1.0, 1.0))
            xformable.AddScaleOp(opSuffix="unitsResolve").Set(Gf.Vec3f(*UNITS_RESOLVE_SCALE))
            paths.append(prim_path)

        # Geometry is identical across every instance of a variant - compute once.
        size = bbox_cache.ComputeWorldBound(stage.GetPrimAtPath(paths[0])).ComputeAlignedRange().GetSize()
        half_extents_by_variant[variant] = (size[0] / 2.0, size[1] / 2.0, size[2] / 2.0)
        paths_by_variant[variant] = paths

    pool = BoxPool(paths_by_variant=paths_by_variant, half_extents_by_variant=half_extents_by_variant)
    logger.info("authored box pool: %d prims across %d variant(s)", len(pool.all_paths()), len(paths_by_variant))
    return pool


# Deliberately left physics-enabled (default) here, unlike the parked state
# BoxSpawner puts pool boxes into at runtime: authoring rigidBodyEnabled=False
# before the stage is ever played means PhysX never creates an actor for the
# prim at all, so RigidPrim's tensor view construction fails during
# world.reset() ("Pattern '...' did not match any rigid bodies") - confirmed by
# running the sim with an earlier version of this function that did exactly
# that. Disabling only works as a *runtime* toggle on an actor that already
# exists (same mechanism stage_setup.truck.despawn_boxes_in_truck uses) - see
# BoxSpawner.__init__, which parks every pool box right after world.reset().
