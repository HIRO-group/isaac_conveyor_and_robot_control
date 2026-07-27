"""Single-configuration IK solve via cuMotion."""

from __future__ import annotations

import cumotion
import numpy as np
from isaacsim.robot_motion.cumotion.impl.utils import isaac_sim_to_cumotion_pose

from pick_and_place.ur20 import IK_CSPACE_LIMIT_BIASING_WEIGHT, TOOL_FRAME_NAME


def solve_ik_target(
    cumotion_robot,
    world_interface,
    target_position: np.ndarray,
    orientation: np.ndarray,
    q_initial: np.ndarray,
) -> np.ndarray:
    """Solve a single joint configuration reaching (target_position, orientation), seeded
    at q_initial, so plan_to_cspace_target has a concrete destination instead of letting
    plan_to_pose_target's JtRRT settle on a possibly contorted pose-satisfying configuration.
    """
    position_world_to_base, quaternion_world_to_base = world_interface.get_world_to_robot_base_transform()
    target_pose_base = isaac_sim_to_cumotion_pose(
        position_world_to_target=target_position,
        orientation_world_to_target=orientation,
        position_world_to_base=position_world_to_base,
        orientation_world_to_base=quaternion_world_to_base,
    )

    ik_config = cumotion.IkConfig()
    ik_config.bfgs_cspace_limit_biasing = cumotion.IkConfig.CSpaceLimitBiasing.ENABLE
    ik_config.bfgs_cspace_limit_biasing_weight = IK_CSPACE_LIMIT_BIASING_WEIGHT
    ik_config.cspace_seeds = [q_initial]

    result = cumotion.solve_ik(
        kinematics=cumotion_robot.kinematics,
        target_pose=target_pose_base,
        target_frame=TOOL_FRAME_NAME,
        config=ik_config,
    )
    if not result.success:
        raise RuntimeError(
            f"cumotion.solve_ik found no joint configuration for "
            f"target_position={target_position} orientation={orientation} "
            f"(seeded at q_initial={q_initial})"
        )
    return result.cspace_position
