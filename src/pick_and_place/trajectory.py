"""Plan-once-on-entry, open-loop trajectory playback against a per-phase tick budget."""

from __future__ import annotations

import logging

import numpy as np
from isaacsim.core.experimental.prims import Articulation
from isaacsim.robot_motion.cumotion import CumotionRobot, GraphBasedMotionPlanner

from pick_and_place.ik import solve_ik_target
from pick_and_place.phases import DOWN_ORIENTATION, PHASE_TICKS
from pick_and_place.ur20 import MAX_JOINT_ACCELERATIONS, MAX_JOINT_VELOCITIES, PLANNING_RETRIES

logger = logging.getLogger(__name__)


class TrajectoryDriver:
    """Plans a fresh collision-free path to a target on phase entry, then plays back
    the resulting trajectory open-loop over subsequent calls to `drive_to` with the
    same `phase_name`; returns True once that phase should end.
    """

    def __init__(
        self,
        robot: Articulation,
        planner: GraphBasedMotionPlanner,
        world_binding,
        cumotion_robot: CumotionRobot,
        physics_dt: float,
    ) -> None:
        self.robot = robot
        self.planner = planner
        self.world_binding = world_binding
        self._cumotion_robot = cumotion_robot
        self._physics_dt = physics_dt
        self._trajectory = None
        self._t = 0.0
        self._step = 0

    def drive_to(
        self,
        target_position: np.ndarray | None,
        phase_name: str,
        orientation: np.ndarray = DOWN_ORIENTATION,
        use_ik_cspace_target: bool = False,
        cspace_target: np.ndarray | list[float] | None = None,
    ) -> bool:
        """Precedence: cspace_target plans directly to a known joint configuration;
        use_ik_cspace_target solves IK first then plans to that joint target; otherwise
        plans directly to (target_position, orientation) via task-space JtRRT.
        """
        if self._step == 0:
            self.world_binding.get_world_interface().update_world_to_robot_root_transforms(
                poses=self.robot.get_world_poses()
            )
            self.world_binding.synchronize_transforms()

            q_initial = self.robot.get_dof_positions().numpy()[0].astype(np.float64)
            path = None
            if cspace_target is not None:
                q_target = np.asarray(cspace_target, dtype=np.float64)
            elif use_ik_cspace_target:
                q_target = solve_ik_target(
                    self._cumotion_robot, self.world_binding.get_world_interface(), target_position, orientation, q_initial
                )
            else:
                q_target = None

            if q_target is not None:
                for attempt in range(PLANNING_RETRIES):
                    path = self.planner.plan_to_cspace_target(q_initial=q_initial, q_final=q_target)
                    if path is not None:
                        break
                    logger.warning(
                        "cspace planning attempt %d/%d failed entering %s (q_target=%s), retrying",
                        attempt + 1, PLANNING_RETRIES, phase_name, q_target,
                    )
            else:
                for attempt in range(PLANNING_RETRIES):
                    path = self.planner.plan_to_pose_target(
                        q_initial=q_initial, position=target_position, orientation=orientation
                    )
                    if path is not None:
                        break
                    logger.warning(
                        "planning attempt %d/%d failed entering %s (target=%s), retrying",
                        attempt + 1, PLANNING_RETRIES, phase_name, target_position,
                    )
            if path is None:
                raise RuntimeError(
                    f"GraphBasedMotionPlanner found no collision-free path entering {phase_name} "
                    f"(target={target_position if q_target is None else q_target}) "
                    f"after {PLANNING_RETRIES} attempts"
                )

            self._trajectory = path.to_minimal_time_joint_trajectory(
                max_velocities=MAX_JOINT_VELOCITIES,
                max_accelerations=MAX_JOINT_ACCELERATIONS,
                robot_joint_space=list(self.robot.dof_names),
                active_joints=self._cumotion_robot.controlled_joint_names,
            )
            self._t = 0.0

        target_state = self._trajectory.get_target_state(self._t)
        if target_state is not None and target_state.joints.positions is not None:
            self.robot.set_dof_position_targets(
                positions=target_state.joints.positions, dof_indices=target_state.joints.position_indices
            )

        self._t += self._physics_dt
        self._step += 1

        finished = self._t >= self._trajectory.duration
        timed_out = self._step >= PHASE_TICKS[phase_name]
        if timed_out and not finished:
            logger.warning(
                "%s exceeded its %d-tick budget before the planned trajectory (duration=%.2fs) finished",
                phase_name, PHASE_TICKS[phase_name], self._trajectory.duration,
            )
        if finished or timed_out:
            self._step = 0
            return True
        return False
