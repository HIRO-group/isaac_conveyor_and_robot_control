"""Phase enum, per-phase tick budgets, and orientation/tolerance constants for
the pick-and-place cycle.
"""

from __future__ import annotations

from enum import IntEnum

import numpy as np
from scipy.spatial.transform import Rotation

from pick_and_place.transforms import quat_wxyz_to_xyzw, quat_xyzw_to_wxyz

ATTACH_MAX_DISTANCE = 0.005  # meters (0.5 cm)
PICK_SETTLE_LINEAR_SPEED = 0.02  # m/s
# Down is weird
DOWN_ORIENTATION = np.array([-0.5, 0.5, -0.5, 0.5])

# Extra clearance above the tallest neighboring box's top, beyond just-clear,
# for LIFT_CLEAR (see Phase.LIFT_CLEAR) - tunable if boxes still get clipped.
LIFT_CLEAR_MARGIN = 0.03  # meters

# Place orientation: DOWN_ORIENTATION rotated 180 deg about the world Z axis
PLACE_ORIENTATION = quat_xyzw_to_wxyz(
    (
        Rotation.from_euler("z", 180, degrees=True) * Rotation.from_quat(quat_wxyz_to_xyzw(DOWN_ORIENTATION))
    ).as_quat()
)

# Per-phase tick budget
PHASE_TICKS = {
    "STAGE_FOR_PICK": 600,
    "DESCEND_TO_PICK": 1200,
    "LIFT_CLEAR": 600,
    "STAGE_FOR_PLACE": 600,
    "DESCEND_TO_PLACE": 1000,
}


class Phase(IntEnum):
    WAITING = 0
    STAGE_FOR_PICK = 1
    DESCEND_TO_PICK = 2
    ATTACH = 3
    LIFT_CLEAR = 4
    STAGE_FOR_PLACE = 5
    DESCEND_TO_PLACE = 6
    DETACH = 7
