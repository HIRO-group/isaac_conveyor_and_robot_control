"""Magic-attach pick-and-place phase state machine: pick a box off the pick-zone
belt, place it on the place-zone belt, via NVIDIA cuMotion's collision-aware
GraphBasedMotionPlanner rather than raw per-tick differential IK.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np
from isaacsim.core.experimental.prims import Articulation, GeomPrim, RigidPrim

from pick_and_place.attachment import attach_box, detach_box
from pick_and_place.box_queries import box_top_center, measure_box_half_height
from pick_and_place.motion_planner import build_motion_planner
from pick_and_place.phases import (
    ATTACH_MAX_DISTANCE,
    LIFT_CLEAR_MARGIN,
    PICK_SETTLE_LINEAR_SPEED,
    PLACE_ORIENTATION,
    Phase,
)
from pick_and_place.transforms import local_z_axis_in_world
from pick_and_place.trajectory import TrajectoryDriver
from pick_and_place.ur20 import (
    TOOL_FRAME_LIVE_PRIM_SUBPATH,
    UR20_DEFAULT_JOINT_POSITIONS,
    UR20_PRE_PLACE_JOINT_POSITIONS,
)

logger = logging.getLogger(__name__)


class MagicAttachPickPlace:
    """Phase state machine: pick a box off the pick-zone belt, place it on the place-zone belt.

    Call ``forward(pick_ready, box_path)`` once per physics step; both args are only
    consulted while WAITING. `box_rigid_prims` must contain a `RigidPrim` pre-constructed
    for every box path up front (before world.reset()) - building one fresh mid-simulation
    was observed to return stale get_world_poses() data on subsequent calls.
    """

    def __init__(
        self,
        robot: Articulation,
        robot_path: str,
        place_xy: tuple,
        place_belt_top_z: float,
        box_rigid_prims: dict,
        physics_dt: float,
        get_pick_zone_occupant_paths: Callable[[], list],
        extra_exclude_obstacle_paths: list = (),
        default_joint_positions: list = UR20_DEFAULT_JOINT_POSITIONS,
        pre_place_joint_positions: list = UR20_PRE_PLACE_JOINT_POSITIONS,
        disable_obstacle_tracking: bool = False,
        hold_fault=None,
        reach_fault=None,
        hesitation=None,
    ) -> None:
        """`hold_fault` (sim_cell.faults.HoldFault, optional) realises kappa_h: it
        decides at each attach whether this hold drops, and when. `reach_fault`
        (sim_cell.faults.ReachFault, optional; also settable later as
        `.reach_fault`) realises kappa_r physically: a descent that takes the
        tool past its radius stalls, retreats and is retried. `hesitation`
        (sim_cell.faults.Hesitation, optional) rate-limits cycles and freezes
        descents so the arm reads as a hesitant policy."""
        self.robot = robot
        self._hold_fault = hold_fault
        self.reach_fault = reach_fault
        self._hesitation = hesitation
        self._tick = 0  # physics steps seen by forward(); the hesitation's clock
        self.place_xy = place_xy
        self.place_belt_top_z = place_belt_top_z
        self.box_rigid_prims = box_rigid_prims
        # Used by LIFT_CLEAR to find which other boxes share the pick zone,
        # since the carried box isn't a tracked collision object (see below).
        self._get_pick_zone_occupant_paths = get_pick_zone_occupant_paths
        # See UR20_PRE_PLACE_JOINT_POSITIONS_AWAY for when to override these.
        self._default_joint_positions = default_joint_positions
        self._pre_place_joint_positions = pre_place_joint_positions

        planner, world_binding, cumotion_robot = build_motion_planner(
            robot,
            robot_path,
            exclude_obstacle_paths=[robot_path, *box_rigid_prims.keys(), *extra_exclude_obstacle_paths],
            disable_obstacle_tracking=disable_obstacle_tracking,
        )
        self._cumotion_robot = cumotion_robot
        self._trajectory_driver = TrajectoryDriver(robot, planner, world_binding, cumotion_robot, physics_dt)

        wrist_link_name, flange_subprim_name = TOOL_FRAME_LIVE_PRIM_SUBPATH.split("/")
        link_names = list(robot.link_names)
        self._wrist_link_path = robot.link_paths[0][link_names.index(wrist_link_name)]
        self._tool_prim = GeomPrim(paths=f"{self._wrist_link_path}/{flange_subprim_name}")
        # Created fresh at ATTACH, deleted at DETACH - only ever one box attached at a time.
        self._attach_joint_path = f"{self._wrist_link_path}/BoxAttachJoint"

        self.place_position: np.ndarray | None = None  # set per-cycle below; depends on box height

        self.box: RigidPrim | None = None
        self._box_path: str | None = None
        self._box_half_height: float | None = None

        self._phase = Phase.WAITING
        self._pick_point: np.ndarray | None = None
        self._descend_target: np.ndarray | None = None  # the pick point, or reach_fault's clip of it
        self._holding_box = False  # True from ATTACH until DETACH; see forward()
        self._retreating = False  # a failed reach: STAGE_FOR_PICK returns to WAITING, not DESCEND

    def _lift_clear_target_z(self) -> float:
        """Top-center Z the held box must reach to clear every other box still in this
        pick zone by LIFT_CLEAR_MARGIN; returns the box's current Z if the zone is otherwise empty.
        """
        neighbor_paths = [p for p in self._get_pick_zone_occupant_paths() if p != self._box_path]
        if not neighbor_paths:
            return self._pick_point[2]
        neighbor_top_zs = [
            self.box_rigid_prims[p].get_world_poses()[0].numpy()[0][2] + 2.0 * measure_box_half_height(p)
            for p in neighbor_paths
        ]
        return max(neighbor_top_zs) + LIFT_CLEAR_MARGIN + 2.0 * self._box_half_height

    def _tool_world_position(self) -> np.ndarray:
        return self._tool_prim.get_world_poses()[0].numpy()[0]

    def forward(self, pick_ready: bool, box_path: str | None) -> None:
        # Once attach_box() creates the FixedJoint, PhysX keeps the box rigidly
        # following wrist_3_link on its own - no per-tick following code needed here.
        self._tick += 1

        # kappa_h: a decided drop fires part-way through the swing to the place side.
        if (
            self._holding_box
            and self._phase == Phase.STAGE_FOR_PLACE
            and self._hold_fault is not None
            and self._hold_fault.should_drop(self._trajectory_driver.progress)
        ):
            self._drop_held_box()
            return

        if self._phase == Phase.WAITING:
            if pick_ready and box_path is not None:
                # Track whichever box triggered pick_ready; only commits once it's
                # settled (see PICK_SETTLE_LINEAR_SPEED).
                candidate_box = self.box_rigid_prims[box_path]
                linear_velocity, _ = candidate_box.get_velocities()
                speed = float(np.linalg.norm(linear_velocity.numpy()[0]))
                if speed >= PICK_SETTLE_LINEAR_SPEED:
                    return
                if self._hesitation is not None:
                    if not self._hesitation.can_start(self._tick):
                        return  # rate-limited: wait at the staging pose
                    self._hesitation.start_cycle(self._tick)
                self.box = candidate_box
                self._box_path = box_path
                self._box_half_height = measure_box_half_height(box_path)
                self._pick_point = box_top_center(self.box, self._box_half_height)
                self._descend_target = self._pick_point
                if self.reach_fault is not None and self.reach_fault.beyond(self._pick_point):
                    # kappa_r (reach mode "attempt"): go as far toward it as the arm reaches
                    self._descend_target = np.array(self.reach_fault.clip(self._pick_point))
                # Disable the box's rigid body now (not at ATTACH) - it must never be
                # physically contactable by the approaching arm.
                self.box.set_enabled_rigid_bodies([False])
                logger.debug(
                    "WAITING->STAGE_FOR_PICK: box_path=%s pick_point=%s box_half_height=%s settled_speed=%s",
                    box_path, self._pick_point, self._box_half_height, speed,
                )
                # Recomputed here since this box's height may differ from the last one placed.
                self.place_position = np.array(
                    [self.place_xy[0], self.place_xy[1], self.place_belt_top_z + 2 * self._box_half_height]
                )
                self._phase = Phase.STAGE_FOR_PICK

        elif self._phase == Phase.STAGE_FOR_PICK:
            # Visited twice per cycle: as the staging point before DESCEND_TO_PICK, and
            # again right after ATTACH to lift the carried box back up through the same pose.
            # After a failed reach it is the retreat, and the cycle starts over from WAITING.
            if self._trajectory_driver.drive_to(None, "STAGE_FOR_PICK", cspace_target=self._default_joint_positions):
                if self._retreating:
                    self._retreating = False
                    self._phase = Phase.WAITING
                else:
                    self._phase = Phase.STAGE_FOR_PLACE if self._holding_box else Phase.DESCEND_TO_PICK

        elif self._phase == Phase.DESCEND_TO_PICK:
            if self._hesitation is not None and self._hesitation.frozen(self._trajectory_driver.progress):
                return  # holding still mid-descent; the last targets keep the arm where it is
            finished = self._trajectory_driver.drive_to(self._descend_target, "DESCEND_TO_PICK")
            if self.reach_fault is not None and (
                self.reach_fault.beyond(self._tool_world_position())
                or (finished and self._descend_target is not self._pick_point)
            ):
                self._stall_at_reach_limit()
            elif finished:
                logger.debug(
                    "DESCEND_TO_PICK end: ee_pos=%s pick_point=%s box=%s tool_z_axis_world=%s",
                    self._tool_world_position(), self._pick_point, self.box.paths,
                    local_z_axis_in_world(self._tool_prim.get_world_poses()[1].numpy()[0]),
                )
                self._phase = Phase.ATTACH

        elif self._phase == Phase.REACH_STALL:
            # Straining at the reach limit; then give the box up and retreat.
            if not self.reach_fault.stalling():
                self._abandon_reach()

        elif self._phase == Phase.ATTACH:
            # Gated on real proximity, not a tick count - DESCEND_TO_PICK finishing is a
            # strong signal but not a guarantee.
            distance = float(np.linalg.norm(self._tool_world_position() - self._pick_point))
            if distance <= ATTACH_MAX_DISTANCE:
                attach_box(self.box, self._wrist_link_path, self._attach_joint_path)
                self._holding_box = True
                self._phase = Phase.LIFT_CLEAR
                if self._hold_fault is not None and self._hold_fault.on_attach():
                    logger.info("kappa_h: this hold of %s will drop", self._box_path)

        elif self._phase == Phase.LIFT_CLEAR:
            # Straight up first: the carried box isn't a tracked collision object, so this
            # clears it past other boxes in the pick zone by hand rather than via the planner.
            target_z = self._lift_clear_target_z()
            if target_z <= self._pick_point[2]:
                self._phase = Phase.STAGE_FOR_PICK
            else:
                lift_target = np.array([self._pick_point[0], self._pick_point[1], target_z])
                if self._trajectory_driver.drive_to(lift_target, "LIFT_CLEAR", use_ik_cspace_target=True):
                    self._phase = Phase.STAGE_FOR_PICK

        elif self._phase == Phase.STAGE_FOR_PLACE:
            if self._trajectory_driver.drive_to(None, "STAGE_FOR_PLACE", cspace_target=self._pre_place_joint_positions):
                self._phase = Phase.DESCEND_TO_PLACE if self._holding_box else Phase.WAITING

        elif self._phase == Phase.DESCEND_TO_PLACE:
            if self._trajectory_driver.drive_to(
                self.place_position, "DESCEND_TO_PLACE", orientation=PLACE_ORIENTATION, use_ik_cspace_target=True
            ):
                self._phase = Phase.DETACH if self._holding_box else Phase.STAGE_FOR_PLACE

        elif self._phase == Phase.DETACH:
            box_pos_before, _ = self.box.get_world_poses()
            logger.debug(
                "detaching: box_pos=%s ee_pos=%s place_target=%s",
                box_pos_before.numpy()[0], self._tool_world_position(), self.place_position,
            )
            detach_box(self._attach_joint_path)
            self._holding_box = False
            self._phase = Phase.STAGE_FOR_PLACE

    def _drop_held_box(self) -> None:
        """A grip failure (kappa_h): release the box mid-swing and finish the cycle
        empty-handed - the in-flight swing is abandoned and re-planned to the pre-place
        pose, from which STAGE_FOR_PLACE returns to WAITING as after a place.
        """
        logger.info(
            "kappa_h: dropped %s at %.0f%% of the swing", self._box_path, 100 * self._trajectory_driver.progress
        )
        detach_box(self._attach_joint_path)
        self._holding_box = False
        self._trajectory_driver.abort()
        self._phase = Phase.STAGE_FOR_PLACE

    def _stall_at_reach_limit(self) -> None:
        """kappa_r (reach mode "attempt"): the descent has reached the arm's limit
        short of the box. Freeze where it is - the in-flight plan is dropped and
        the current joint positions become the targets - and strain there a moment.
        """
        fault = self.reach_fault
        tool = self._tool_world_position()
        attempt = fault.begin_stall(tool)
        logger.info(
            "kappa_r: attempt %d on %s stalled %.2f m from the base at z=%.2f (limit %.2f m, box at %.2f m, top z=%.2f)",
            attempt, self._box_path, fault.last_distance_m, tool[2], fault.max_reach_m,
            fault.distance_m(self._pick_point), self._pick_point[2],
        )
        self._trajectory_driver.abort()
        self.robot.set_dof_position_targets(positions=self.robot.get_dof_positions())
        self._phase = Phase.REACH_STALL

    def _abandon_reach(self) -> None:
        """Give the unreachable box back to the belt (its rigid body was disabled at
        selection) and retreat to the staging pose; WAITING then selects it again,
        so the arm keeps trying for as long as the box is there.
        """
        logger.info("kappa_r: giving up on %s - retreating to try again", self._box_path)
        self.box.set_enabled_rigid_bodies([True])
        self.box = None
        self._box_path = None
        self._pick_point = None
        self._descend_target = None
        self._retreating = True
        self._phase = Phase.STAGE_FOR_PICK

    def reset(self) -> None:
        """Abandon the current cycle: drop a held box and return to WAITING."""
        if self._holding_box:
            detach_box(self._attach_joint_path)
        elif self.box is not None and self._phase in (Phase.DESCEND_TO_PICK, Phase.REACH_STALL, Phase.ATTACH):
            self.box.set_enabled_rigid_bodies([True])  # disabled at selection, never attached
        if self._hold_fault is not None:
            self._hold_fault.reset()
        if self.reach_fault is not None:
            self.reach_fault.reset()
        if self._hesitation is not None:
            self._hesitation.reset()
        self._retreating = False
        self._trajectory_driver.abort()
        self._holding_box = False
        self.box = None
        self._box_path = None
        self._pick_point = None
        self._phase = Phase.WAITING

    @property
    def wrist_link_path(self) -> str:
        """USD path of the wrist link the box attaches to - see attachment.attach_box.
        Exposed for CONVEYOR_INDEXING_EXTERNAL_ACTION mode (sim_cell.runner), which
        drives attach/detach directly instead of through this class's phase machine.
        """
        return self._wrist_link_path

    @property
    def attach_joint_path(self) -> str:
        """USD path of the (possibly not-yet-created) FixedJoint - see attachment.py."""
        return self._attach_joint_path

    def tool_world_position(self) -> np.ndarray:
        """Public wrapper around _tool_world_position(), for external-control-mode
        proximity checks (see pick_and_place.external_control.apply_suction_edge)."""
        return self._tool_world_position()

    def tool_world_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """(position, orientation_wxyz) of the tool frame in world space -
        the position-and-orientation counterpart of tool_world_position().
        Used by ground-truth recording (conveyor_indexing.mcap_recorder's
        sim/arm/{n}/tool_pose channel) - independent of the phase state
        machine (this GeomPrim tracks the arm's actual physical
        wrist_3_link/flange, so it stays live in
        CONVEYOR_INDEXING_EXTERNAL_ACTION mode too, unlike
        holding_box/held_box_path above).
        """
        positions, orientations = self._tool_prim.get_world_poses()
        return positions.numpy()[0], orientations.numpy()[0]

    @property
    def phase_name(self) -> str:
        return self._phase.name

    @property
    def holding_box(self) -> bool:
        """True from ATTACH until DETACH - the sim's stand-in for suction/cup DIO state."""
        return self._holding_box

    @property
    def held_box_path(self) -> str | None:
        """The box prim path this arm is currently carrying, else None - same
        ATTACH..DETACH window as holding_box. Ground-truth recording
        (sim_cell.recording) uses this for BoxState.held_by_arm.
        """
        return self._box_path if self._holding_box else None
