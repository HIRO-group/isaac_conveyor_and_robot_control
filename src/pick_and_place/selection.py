"""Pick-candidate ranking: which occupying box should this robot actually pick."""

from __future__ import annotations

import math

PICK_MAX_REACH_M = 1.75  # UR20 spec reach; pick targets farther than this are dropped


def rank_pick_zone_hit_paths(
    hit_paths: list,
    box_positions: dict,
    robot_xy: tuple,
    max_reach_m: float = PICK_MAX_REACH_M,
) -> str | None:
    """Pick the closest reachable box to the robot (height as tie-breaker); drops
    anything farther than max_reach_m rather than ranking it.

    `box_positions` is {path: (x, y, z)}, precomputed once per control tick from a
    single batched RigidPrim read (see sim_cell.runner) rather than queried per-box.
    """

    def _distance(path: str) -> float:
        return float(math.dist(box_positions[path][:2], robot_xy))

    reachable = [path for path in hit_paths if _distance(path) <= max_reach_m]
    if not reachable:
        return None

    def _score(path: str) -> tuple:
        pos = box_positions[path]
        return (_distance(path), -float(pos[2]), path)

    return min(reachable, key=_score)
