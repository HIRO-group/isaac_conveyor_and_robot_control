"""Belt-direction geometry correction.

Overwrites every zone's ``inputs:direction`` from actual belt geometry - the
baked values aren't reliable (see ``conveyor_indexing.zone.ConveyorZone``).

Straight zones: unit vector from this zone's bbox_center toward the next
zone's, snapped to the belt's long axis, authored as-is in world space (no
per-body flip negation - confirmed empirically the wrong way round). Curved
zones: angular-velocity magnitude rederived from the curve's own radius (the
baked value is far too large for this scaffold's velocity scaling); sign
rederived from whether the curve's entry-side neighbor is itself flipped
(copying the baked sign directly is wrong at one end of each loop, since the
two curves are mirror-image ends of the same track).
"""

from __future__ import annotations

import logging
import math

from pxr import Gf, Usd, UsdGeom

logger = logging.getLogger(__name__)


def is_body_flipped(belt_prim: Usd.Prim) -> bool:
    """True if belt_prim's world rotation is ~180deg about some axis (i.e. its
    body frame is flipped relative to world space).
    """
    world_rotation = UsdGeom.Xformable(belt_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractRotation()
    return abs(world_rotation.GetQuat().GetReal()) < 0.5


def fix_zone_directions(zones: list, closed_loop: bool) -> None:
    """Rederive and author every zone's `inputs:direction` in `zones` (in
    belt order); also sets each straight zone's `world_travel_direction`.
    """
    n = len(zones)
    for i, zone in enumerate(zones):
        if not zone.is_straight:
            baked = zone.direction_attr.Get()
            prev_zone = zones[(i - 1) % n]
            next_center = zones[(i + 1) % n].bbox_center
            radius = 0.5 * math.dist(prev_zone.bbox_center, next_center)
            baked_sign = -1.0 if baked[2] < 0.0 else 1.0
            sign = baked_sign if is_body_flipped(prev_zone.belt_prim) else -baked_sign
            corrected = Gf.Vec3f(0.0, 0.0, sign / radius)
            if Gf.Vec3f(baked) != corrected:
                logger.info(
                    "correcting %s inputs:direction %s -> %s (radius=%.3fm)",
                    zone.node_path, tuple(baked), tuple(corrected), radius,
                )
                zone.direction_attr.Set(corrected)
            continue

        if i + 1 < n:
            next_center = zones[i + 1].bbox_center
        elif closed_loop:
            next_center = zones[0].bbox_center
        else:
            # Last zone of an open line: extrapolate from the previous zone's
            # center through this one, rather than wrapping back to zone 0.
            prev_center = zones[i - 1].bbox_center
            next_center = (
                2 * zone.bbox_center[0] - prev_center[0],
                2 * zone.bbox_center[1] - prev_center[1],
            )
        dx = next_center[0] - zone.bbox_center[0]
        dy = next_center[1] - zone.bbox_center[1]
        corrected = Gf.Vec3f(1.0, 0.0, 0.0) if abs(dx) >= abs(dy) else Gf.Vec3f(0.0, 1.0, 0.0)
        if (dx if abs(dx) >= abs(dy) else dy) < 0.0:
            corrected = -corrected
        # World-space travel direction - used by is_past_center, which
        # only cares about actual world-space geometry and must NOT be
        # negated below (unlike what gets authored as inputs:direction).
        zone.world_travel_direction = Gf.Vec3f(corrected)

        # The authored inputs:direction (unlike world_travel_direction above) DOES need
        # a per-body sign flip here: every track is uniformly ~180deg-rotated about Z,
        # and the un-negated vector drove boxes the wrong way off the belt's open edge.
        authored = -corrected if is_body_flipped(zone.belt_prim) else corrected
        baked = zone.direction_attr.Get()
        if Gf.Vec3f(baked) != authored:
            logger.info(
                "correcting %s inputs:direction %s -> %s",
                zone.node_path, tuple(baked), tuple(authored),
            )
            zone.direction_attr.Set(authored)
