"""cuMotion GraphBasedMotionPlanner construction for the UR20.

Only the Belt mesh (not the frame/posts mesh) fails cuMotion's obstacle scan;
`build_motion_planner`'s own retry loop already excludes exactly the failing
prims one at a time, so no exclusion list is passed in by callers for that.
Synthetic capsule obstacles were tried and rejected: CollisionAPI makes a
real PhysX collider, not just a planning hint, so they physically shoved the
real boxes on the belt.
"""

from __future__ import annotations

import logging
import re

import isaacsim.core.experimental.utils.stage as stage_utils
import isaacsim.robot_motion.experimental.motion_generation as mg
from isaacsim.core.experimental.objects import Cone, Cylinder, Mesh
from isaacsim.core.experimental.prims import Articulation
from isaacsim.robot_motion.cumotion import (
    CumotionRobot,
    CumotionWorldInterface,
    GraphBasedMotionPlanner,
    load_cumotion_robot,
)

from pick_and_place.obstacle_guard import preserve_obstacle_rotations
from pick_and_place.ur20 import TOOL_FRAME_NAME, UR20_CONFIG_DIR

logger = logging.getLogger(__name__)

_MAX_ATTEMPTS = 40
_AABB_SEARCH_HALF_EXTENT_M = 10.0  # generous box around the robot; not a tuned value


def build_motion_planner(
    robot: Articulation,
    robot_path: str,
    exclude_obstacle_paths: list,
    disable_obstacle_tracking: bool = False,
) -> tuple[GraphBasedMotionPlanner, mg.WorldBinding, CumotionRobot]:
    """Load the generated UR20 cuMotion config and wire up a collision-aware GraphBasedMotionPlanner.

    Args:
        exclude_obstacle_paths: Prim paths to exclude from the obstacle set - the
            robot itself and every box that could ever be a pick/place target.
        disable_obstacle_tracking: Debug escape hatch - skips the AABB obstacle
            scan entirely (planner sees an empty world). Off in normal use.
    """
    cumotion_robot = load_cumotion_robot(directory=UR20_CONFIG_DIR)
    tool_frames = cumotion_robot.robot_description.tool_frame_names()
    if TOOL_FRAME_NAME not in tool_frames:
        raise RuntimeError(f"Expected tool frame '{TOOL_FRAME_NAME}' not found in generated XRDF: {tool_frames}")

    robot_pos, robot_ori = robot.get_world_poses()

    obstacle_strategy = mg.ObstacleStrategy()
    obstacle_strategy.set_default_configuration(Mesh, mg.ObstacleConfiguration("triangulated_mesh", 0.005))
    obstacle_strategy.set_default_configuration(Cone, mg.ObstacleConfiguration("obb", 0.0))
    obstacle_strategy.set_default_configuration(Cylinder, mg.ObstacleConfiguration("obb", 0.0))
    exclude_paths = list(exclude_obstacle_paths)

    def _scan_aabb(paths: list) -> list:
        if disable_obstacle_tracking:
            return []
        half = _AABB_SEARCH_HALF_EXTENT_M
        return mg.SceneQuery().get_prims_in_aabb(
            search_box_origin=robot_pos.numpy()[0],
            search_box_minimum=[-half, -half, -half],
            search_box_maximum=[half, half, half],
            tracked_api=mg.TrackableApi.PHYSICS_COLLISION,
            exclude_prim_paths=paths,
        )

    # Guard against the WorldBinding.initialize() rotation-reset bug (see
    # pick_and_place.obstacle_guard) around every prim tracked by the initial
    # scan - fixed for the duration of the retry loop below, independent of how
    # exclude_paths grows as attempts fail.
    with preserve_obstacle_rotations(_scan_aabb(exclude_paths)):
        world_binding = None
        for attempt in range(_MAX_ATTEMPTS):
            tracked_prims = _scan_aabb(exclude_paths)
            world_binding = mg.WorldBinding(
                world_interface=CumotionWorldInterface(),
                obstacle_strategy=obstacle_strategy,
                tracked_prims=tracked_prims,
                tracked_collision_api=mg.TrackableApi.PHYSICS_COLLISION,
            )
            try:
                world_binding.initialize()
                break
            except (AssertionError, RuntimeError) as exc:
                if attempt >= _MAX_ATTEMPTS - 1:
                    raise
                message = str(exc)
                if "non-unity scaling" in message:
                    offending_paths = re.findall(r"'(/[^']+)'", message)
                elif "does not point to a supported shape type" in message:
                    offending_paths = re.findall(r"Prim path (\S+) does not point", message)
                else:
                    raise
                if not offending_paths:
                    raise
                logger.warning(
                    "excluding from the planner's obstacle set "
                    "(pre-existing conveyor_setup.usd quirk, see module docstring): %s",
                    offending_paths,
                )
                exclude_paths = exclude_paths + offending_paths

    world_binding.get_world_interface().update_world_to_robot_root_transforms(poses=(robot_pos, robot_ori))
    world_binding.synchronize_transforms()

    planner = GraphBasedMotionPlanner(
        cumotion_robot=cumotion_robot,
        cumotion_world_interface=world_binding.get_world_interface(),
        tool_frame=TOOL_FRAME_NAME,
    )
    return planner, world_binding, cumotion_robot
