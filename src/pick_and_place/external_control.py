"""External-control-mode suction handling: attach/detach driven by an
externally-supplied suction bit instead of MagicAttachPickPlace's ATTACH/DETACH
phases. Reuses the same standalone attach_box/detach_box functions the phase
machine itself calls, so both control paths converge on identical physics.
"""

from __future__ import annotations

import logging

import numpy as np

from pick_and_place.attachment import attach_box, detach_box
from pick_and_place.box_queries import box_top_center, measure_box_half_height

logger = logging.getLogger(__name__)

# Deliberately looser than the autonomous controller's ATTACH_MAX_DISTANCE (see
# phases.py, 0.005m) - a separate, tunable gate for external-control testing,
# not a claim that 0.20m is physically close enough to actually pick up a box.
#
# Bumped 0.35 -> 0.5 (2026-09-01, one-off diagnostic: real rollout-eval data
# showed F-D's arm1/arm2 settling within 0.48-0.65m of a real box but never
# crossing 0.35m). Bumped again 0.5 -> 1.0 (2026-09-02, capability-diffusion's
# Stage 7c real-pick work, user-authorized): with the commanding client's
# pick-target height AND orientation both confirmed correct (independently
# verified - FK-matched target computation, live visual confirmation of the
# approach motion), live proximity-check readings still ran 0.4-0.9m -
# widened here to unblock data collection rather than leave it stuck on a
# residual gap that's tracked separately (see capability-diffusion's own
# docs/progress-tracker.md, Stage 7c section, for the investigation this
# value change is part of). Revert or re-tighten once that residual gap is
# itself root-caused and closed.
EXTERNAL_ATTACH_MAX_DISTANCE = 1.0  # meters


def apply_suction_edge(arm: int, pick_place, box_rigid_prims: dict, suction: bool, held_box_path, candidate_box_path):
    """Call once per tick per arm in external-control mode. `held_box_path` is
    this arm's current held box path (None if not holding) - the caller's own
    state, threaded through each call (holding iff held_box_path is not None,
    so no separate "previous suction" flag is needed). Returns the updated
    held_box_path.

    Only reacts on suction *edges* (attach_box/detach_box are one-shot calls,
    not idempotent per-tick), and only attaches when `candidate_box_path`
    (from sim_cell.pick_dispatch.evaluate_pick_station, computed independent
    of the phase machine) is within EXTERNAL_ATTACH_MAX_DISTANCE of the tool
    tip - the policy decides *when* to attach, this just guards against
    attaching to a box that isn't actually in reach. `arm` is only used to
    tag log lines - both arms otherwise run identical logic.
    """
    currently_holding = held_box_path is not None

    if suction and not currently_holding:
        if candidate_box_path is None:
            logger.warning("arm%d: suction commanded with no candidate box in range; ignoring", arm)
            return held_box_path
        box = box_rigid_prims[candidate_box_path]
        half_height = measure_box_half_height(candidate_box_path)
        pick_point = box_top_center(box, half_height)
        distance = float(np.linalg.norm(pick_place.tool_world_position() - pick_point))
        if distance > EXTERNAL_ATTACH_MAX_DISTANCE:
            logger.warning(
                "arm%d: suction commanded %.4fm from box %s (max %.4fm); ignoring",
                arm, distance, candidate_box_path, EXTERNAL_ATTACH_MAX_DISTANCE,
            )
            return held_box_path
        attach_box(box, pick_place.wrist_link_path, pick_place.attach_joint_path)
        return candidate_box_path

    if not suction and currently_holding:
        detach_box(pick_place.attach_joint_path)
        return None

    return held_box_path
