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
# value change is part of).
#
# RE-TIGHTENED 1.0 -> 0.05 (2026-09-03, user-authorized): 1.0m was masking a
# real gap instead of fixing it - the commanding client's own attach-confirm
# retrack loop (capability-diffusion's vm_collect_stage7c_real_pick_coupling.py)
# re-solves and publishes a fresh descend target every 0.1s without ever
# waiting for the arm to actually converge on it before this proximity check
# runs, so suction gets evaluated mid-transit toward a just-updated target -
# a timing gap, not a geometry gap. 0.05m is deliberately still 10x looser
# than the autonomous controller's 0.005m (this control path's IK/interpolation
# has real, if now much smaller, settling imperfections of its own) but is a
# real "surface must actually be close to the box" gate again, not a
# meters-wide one. Re-tighten toward 0.005m once the commanding client's own
# retrack-convergence gap (tracked separately) is closed.
EXTERNAL_ATTACH_MAX_DISTANCE = 0.05  # meters


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
        # Attach whatever box is actually UNDER THE TOOL, not the zone's
        # ranked candidate (2026-09-09 fix). `candidate_box_path` comes from
        # pick_dispatch.evaluate_pick_station, which returns the box nearest
        # THE ROBOT among those occupying the pick zone. An external client
        # commits to one box and then spends seconds driving to it, during
        # which a different box can become "nearest" - most often for arm 2,
        # whose zone is downstream and accumulates a queue. The proximity
        # check then measured tool -> some OTHER box and rejected a grab whose
        # tool was visibly touching its real target: live misses came back
        # quantized at box spacing (0.41 / 0.62 / 1.13 / 2.37m, each clustered
        # to ~1mm across hundreds of samples - a static mismatch, not motion),
        # and arm 2 collapsed to 12% success while arm 1 ran at 89%.
        #
        # Ranking by distance to the tool makes the check measure the thing it
        # is actually gating: is the suction cup on a box? The tight
        # EXTERNAL_ATTACH_MAX_DISTANCE gate is what keeps this honest - only a
        # box essentially at the cup qualifies, so this cannot silently grab
        # something across the cell.
        tool_position = pick_place.tool_world_position()
        nearest_path, nearest_box, nearest_distance = None, None, None
        for box_path, box_prim in box_rigid_prims.items():
            pick_point = box_top_center(box_prim, measure_box_half_height(box_path))
            candidate_distance = float(np.linalg.norm(tool_position - pick_point))
            if nearest_distance is None or candidate_distance < nearest_distance:
                nearest_path, nearest_box, nearest_distance = box_path, box_prim, candidate_distance
        if nearest_path is None:
            logger.warning("arm%d: suction commanded with no box in the scene at all; ignoring", arm)
            return held_box_path
        if nearest_distance > EXTERNAL_ATTACH_MAX_DISTANCE:
            logger.warning(
                "arm%d: suction commanded %.4fm from nearest box %s (max %.4fm, zone candidate was %s); ignoring",
                arm, nearest_distance, nearest_path, EXTERNAL_ATTACH_MAX_DISTANCE, candidate_box_path,
            )
            return held_box_path
        attach_box(nearest_box, pick_place.wrist_link_path, pick_place.attach_joint_path)
        return nearest_path

    if not suction and currently_holding:
        detach_box(pick_place.attach_joint_path)
        return None

    return held_box_path
