"""PhysX box-overlap occupancy queries and leading-occupant ranking.

Occupancy sensing is a PhysX box-overlap query (``overlap_box``) against each
zone's belt bounding box, computed once at startup (belts are static). Hits
whose rigid body path falls under a known structure root (conveyor track,
robot, pedestal, truck - see ``sim_cell.layout.EXCLUDED_STRUCTURE_ROOTS``) are
structure, not items, and are excluded.
"""

from __future__ import annotations

import logging

import carb
from omni.physics.core import get_physics_scene_query_interface
from pxr import Gf, PhysicsSchemaTools

logger = logging.getLogger(__name__)


def overlap_box_prim_paths(
    half_extent: list,
    center: list,
    quat: "carb.Float4",
    excluded_roots: tuple,
    zone_name: str = "",
) -> list:
    """Return paths of every rigid body overlapping the given box query,
    excluding anything under `excluded_roots` (belt/structure/robot/truck
    geometry, not a transported item).
    """
    hits = []

    def report_hit(hit) -> bool:
        path = str(PhysicsSchemaTools.intToSdfPath(hit.rigid_body))
        if not path.startswith(excluded_roots):
            hits.append(path)
            logger.debug("occupancy hit: zone=%s hit_path=%s", zone_name, path)
        return True

    get_physics_scene_query_interface().overlap_box(
        carb.Float3(*half_extent),
        carb.Float3(*center),
        quat,
        report_hit,
    )
    return hits


def leading_occupant_path(
    travel_direction: "Gf.Vec3f | None",
    hit_paths: list,
    box_positions: dict,
    zone_name: str = "",
) -> str | None:
    """Pick whichever occupying box is furthest downstream - hit_paths isn't
    in spatial order, so hit_paths[0] could be a trailing box instead.

    `box_positions` is {path: (x, y, z)}, precomputed once per control tick
    from a single batched RigidPrim read (see sim_cell.runner) rather than
    queried per-box here.
    """
    if not hit_paths:
        return None

    if travel_direction is None:
        raise RuntimeError(f"{zone_name} has no world_travel_direction set")

    def _downstream(path: str) -> float:
        pos = box_positions[path]
        return float(
            pos[0] * travel_direction[0] + pos[1] * travel_direction[1] + pos[2] * travel_direction[2]
        )

    return max(hit_paths, key=_downstream)
