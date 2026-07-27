"""UR20-specific constants: cuMotion config location, tool frame, joint
positions, and planning/trajectory tuning.
"""

from __future__ import annotations

import math

from pick_and_place.paths import ROBOT_CONFIGS_DIR

UR20_CONFIG_DIR = str(ROBOT_CONFIGS_DIR / "ur20")
TOOL_FRAME_NAME = "tool0"
TOOL_FRAME_LIVE_PRIM_SUBPATH = "wrist_3_link/flange"
UR20_DEFAULT_JOINT_POSITIONS = [1.5708, -1.5708, -1.5708, -1.5708, 1.5708, 3.1415]
UR20_PRE_PLACE_JOINT_POSITIONS = [-1.5708, -1.5708, -1.5708, -1.5708, 1.5708, 3.1415]

# Same pose as UR20_PRE_PLACE_JOINT_POSITIONS but reached via +180deg instead of
# -90deg, so the STAGE_FOR_PICK<->STAGE_FOR_PLACE swing arcs toward +X instead of -X.
# Use for a robot with a neighbor on its -X side.
UR20_PRE_PLACE_JOINT_POSITIONS_AWAY = [
    UR20_DEFAULT_JOINT_POSITIONS[0] + math.pi
] + UR20_PRE_PLACE_JOINT_POSITIONS[1:]

MAX_JOINT_VELOCITIES = [2.0, 2.0, 2.0, 2.0, 2.0, 2.0]
MAX_JOINT_ACCELERATIONS = [10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
PLANNING_RETRIES = 3
IK_CSPACE_LIMIT_BIASING_WEIGHT = 1.0  # relative weight; see IkConfig docs
