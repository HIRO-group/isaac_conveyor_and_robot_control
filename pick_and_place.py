"""Magic-attach pick-and-place: UR10 moves a box from a pick-zone conveyor to a
place-zone conveyor on another loop.

No real grasp physics: the box is "magic attached" by disabling its rigid body
and teleporting it to follow the end effector at a fixed offset, then
re-enabled at the place location. The box's pose is queried directly
(privileged / ground truth) - no perception involved. See conveyor_indexer.py
for how the pick zone's belt is held stopped while occupied, and how it
resumes once the robot has removed the box ("starved" -> next box advances).
"""

from __future__ import annotations

import math
from enum import IntEnum

import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdPhysics

import isaacsim.core.experimental.utils.stage as stage_utils
from isaacsim.core.experimental.objects import Cylinder
from isaacsim.core.experimental.prims import RigidPrim
from isaacsim.robot.experimental.manipulators.examples.universal_robots.ur10 import UR10
from isaacsim.storage.native import get_assets_root_path

# Replicated from UR10.__init__'s create_robot=True path (ur10.py) - a
# reasonable non-singular starting configuration. Needed here because we
# construct UR10 with create_robot=False (see create_pedestal_and_robot),
# which skips that block.
UR10_DEFAULT_JOINT_POSITIONS = [-math.pi / 2, -math.pi / 2, -math.pi / 2, -math.pi / 2, math.pi / 2, 0.0]

APPROACH_HEIGHT = 0.15  # meters above the grasp/place point while transiting
IK_METHOD = "damped-least-squares"

# A zone reporting Machine=IDLE (held) means the belt has stopped commanding
# motion, but a box arriving via INDUCTING still carries real momentum and
# was observed coasting/sliding several meters in the ~1s it takes to
# capture a pick point and descend - the belt stopping doesn't brake it.
# Require linear speed below this before treating a box as an actual pick
# target, not just "the zone reports occupied+idle."
PICK_SETTLE_LINEAR_SPEED = 0.02  # m/s

# Fixed tick counts per phase, forward() called once per physics step (120 Hz
# in conveyor_indexer.py) - same pattern as FrankaPickPlace in this codebase.
PHASE_TICKS = {
    "MOVE_ABOVE_PICK": 90,
    "DESCEND_TO_PICK": 60,
    "ATTACH": 5,
    "LIFT": 60,
    "MOVE_ABOVE_PLACE": 150,
    "DESCEND_TO_PLACE": 60,
    "DETACH": 5,
    "RETRACT": 60,
}


class Phase(IntEnum):
    WAITING = 0
    MOVE_ABOVE_PICK = 1
    DESCEND_TO_PICK = 2
    ATTACH = 3
    LIFT = 4
    MOVE_ABOVE_PLACE = 5
    DESCEND_TO_PLACE = 6
    DETACH = 7
    RETRACT = 8


def measure_box_half_height(box_path: str) -> float:
    """Privileged, one-time query of a box prim's half-height via its world bbox."""
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(box_path)
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    aligned_range = bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange()
    return aligned_range.GetSize()[2] / 2.0


def create_pedestal_and_robot(
    stage: Usd.Stage,
    robot_path: str,
    pedestal_path: str,
    position: tuple,
    pedestal_height: float,
    pedestal_radius: float = 0.15,
) -> UR10:
    """Create a simple static cylindrical pedestal and a UR10 on top of it.

    Args:
        position: (x, y, z) of the pedestal's base (ground contact point).
        pedestal_height: Pedestal column height; the robot is placed at
            z = position[2] + pedestal_height.
    """
    px, py, pz = position
    pedestal = Cylinder(
        paths=pedestal_path,
        positions=[px, py, pz + pedestal_height / 2.0],
        radii=pedestal_radius,
        heights=pedestal_height,
        colors="gray",
    )
    # Static collider only - no RigidBodyAPI, so it doesn't fall under gravity
    # and isn't mistaken for a kinematic/dynamic body by anything else.
    UsdPhysics.CollisionAPI.Apply(pedestal.prims[0])

    # Position the robot via plain USD BEFORE wrapping it in the UR10/
    # Articulation controller class, rather than via UR10(create_robot=True)
    # + Articulation.set_world_poses() afterward. The latter was verified
    # (via get_world_poses() reading back (0, 0, 0) both immediately after
    # the call and again after world.reset()) to silently not reposition the
    # robot at this stage of initialization - Articulation.set_world_poses()
    # targets the articulation's root_joint frame, which apparently doesn't
    # take a plain Xform write at this point. Setting the reference prim's
    # own transform directly, before any Articulation/PhysX wrapping exists,
    # sidesteps that entirely.
    usd_path = get_assets_root_path() + "/Isaac/Robots/UniversalRobots/ur10/ur10.usd"
    robot_prim = stage_utils.add_reference_to_stage(usd_path=usd_path, path=robot_path, variants=[])
    xformable = UsdGeom.Xformable(robot_prim)
    xformable.ClearXformOpOrder()
    xformable.AddTranslateOp().Set(Gf.Vec3d(px, py, pz + pedestal_height))

    robot = UR10(robot_path=robot_path, create_robot=False, attach_gripper=False)
    # create_robot=False skips UR10.__init__'s default-joint-state block too;
    # replicate it so the arm starts from a sane, non-singular configuration.
    robot.set_default_state(dof_positions=UR10_DEFAULT_JOINT_POSITIONS)
    return robot


class MagicAttachPickPlace:
    """Phase state machine: pick a box off the pick-zone belt, place it on the place-zone belt.

    Call ``forward(pick_ready, box_path)`` once per physics step. There are
    two real physics-enabled boxes in the scene (see README), so the box to
    track isn't fixed - `box_path` should be whichever prim's rigid body was
    actually found occupying the pick zone (conveyor_indexer.py resolves this
    via ConveyorZone.get_occupying_prim_paths()). Both args are only
    consulted while WAITING. `box_rigid_prims` must contain a `RigidPrim`
    pre-constructed for every box path that might ever appear, built once at
    startup (before world.reset()) - constructing a fresh `RigidPrim` for an
    arbitrary path mid-simulation was observed to return stale/wrong
    get_world_poses() data on subsequent calls (same object, stationary box,
    inconsistent readings a second apart), vs. reusing one built up front.
    """

    def __init__(
        self,
        robot: UR10,
        place_xy: tuple,
        place_belt_top_z: float,
        box_rigid_prims: dict,
    ) -> None:
        self.robot = robot
        self.place_xy = place_xy
        self.place_belt_top_z = place_belt_top_z
        self.box_rigid_prims = box_rigid_prims
        # Depends on which box is being carried (set per-cycle below, since
        # the two real boxes in the scene aren't necessarily the same size) -
        # box bottom flush on the belt means top-center (what the end
        # effector targets, same convention as _pick_point) sits at
        # belt_top_z + 2*box_half_height.
        self.place_position: np.ndarray | None = None

        self.box: RigidPrim | None = None
        self._box_half_height: float | None = None

        self._phase = Phase.WAITING
        self._step = 0
        self._pick_point: np.ndarray | None = None
        self._attach_offset: np.ndarray | None = None
        self._attach_orientation: np.ndarray | None = None

    def _box_top_center(self) -> np.ndarray:
        """Privileged query: the box's current world-space top-face center."""
        position, _ = self.box.get_world_poses()
        center = position.numpy()[0]
        return center + np.array([0.0, 0.0, self._box_half_height])

    def _advance(self, phase_name: str) -> bool:
        """Increment the phase step counter; return True once the phase's duration elapses."""
        self._step += 1
        if self._step >= PHASE_TICKS[phase_name]:
            self._step = 0
            return True
        return False

    def forward(self, pick_ready: bool, box_path: str | None) -> None:
        goal_orientation = self.robot.get_downward_orientation()
        _, current_ee_position, _ = self.robot.get_current_state()
        current_ee_position = current_ee_position[0]

        # While holding the box, keep it rigidly offset from the end effector -
        # read BEFORE issuing this tick's motion command so it tracks the
        # actually-reached pose rather than lagging the aspirational target.
        if self._phase in (Phase.LIFT, Phase.MOVE_ABOVE_PLACE, Phase.DESCEND_TO_PLACE):
            self.box.set_world_poses(
                positions=current_ee_position + self._attach_offset,
                orientations=self._attach_orientation,
            )

        if self._phase == Phase.WAITING:
            if pick_ready and box_path is not None:
                # Track whichever box actually triggered pick_ready - not
                # necessarily the same one as last cycle (two real boxes can
                # occupy this zone, see README). Re-checked every physics
                # tick while pick_ready holds, so this only actually commits
                # once the box has settled (see PICK_SETTLE_LINEAR_SPEED).
                candidate_box = self.box_rigid_prims[box_path]
                linear_velocity, _ = candidate_box.get_velocities()
                speed = float(np.linalg.norm(linear_velocity.numpy()[0]))
                if speed >= PICK_SETTLE_LINEAR_SPEED:
                    return
                self.box = candidate_box
                self._box_half_height = measure_box_half_height(box_path)
                self._pick_point = self._box_top_center()
                print(
                    f"[pick_and_place] DEBUG WAITING->MOVE_ABOVE_PICK: box_path={box_path} "
                    f"pick_point={self._pick_point} box_half_height={self._box_half_height} settled_speed={speed}",
                    flush=True,
                )
                # Same top-center convention as _pick_point, recomputed here
                # since this box's height may differ from the last one placed.
                self.place_position = np.array(
                    [self.place_xy[0], self.place_xy[1], self.place_belt_top_z + 2 * self._box_half_height]
                )
                self._phase = Phase.MOVE_ABOVE_PICK
                self._step = 0

        elif self._phase == Phase.MOVE_ABOVE_PICK:
            goal = self._pick_point + np.array([0.0, 0.0, APPROACH_HEIGHT])
            self.robot.set_end_effector_pose(position=goal, orientation=goal_orientation, ik_method=IK_METHOD)
            if self._advance("MOVE_ABOVE_PICK"):
                self._phase = Phase.DESCEND_TO_PICK

        elif self._phase == Phase.DESCEND_TO_PICK:
            self.robot.set_end_effector_pose(position=self._pick_point, orientation=goal_orientation, ik_method=IK_METHOD)
            if self._advance("DESCEND_TO_PICK"):
                print(
                    f"[pick_and_place] DEBUG DESCEND_TO_PICK end: ee_pos={current_ee_position} "
                    f"pick_point={self._pick_point} box={self.box.paths}",
                    flush=True,
                )
                self._phase = Phase.ATTACH

        elif self._phase == Phase.ATTACH:
            if self._step == 0:
                box_position, box_orientation = self.box.get_world_poses()
                self._attach_offset = box_position.numpy()[0] - current_ee_position
                self._attach_orientation = box_orientation.numpy()[0]
                print(
                    f"[pick_and_place] DEBUG ATTACH: ee_pos={current_ee_position} "
                    f"box_pos={box_position.numpy()[0]} attach_offset={self._attach_offset}",
                    flush=True,
                )
                self.box.set_enabled_rigid_bodies([False])
            if self._advance("ATTACH"):
                self._phase = Phase.LIFT

        elif self._phase == Phase.LIFT:
            goal = self._pick_point + np.array([0.0, 0.0, APPROACH_HEIGHT])
            self.robot.set_end_effector_pose(position=goal, orientation=goal_orientation, ik_method=IK_METHOD)
            if self._advance("LIFT"):
                self._phase = Phase.MOVE_ABOVE_PLACE

        elif self._phase == Phase.MOVE_ABOVE_PLACE:
            goal = self.place_position + np.array([0.0, 0.0, APPROACH_HEIGHT])
            self.robot.set_end_effector_pose(position=goal, orientation=goal_orientation, ik_method=IK_METHOD)
            if self._step == 0:
                print(f"[pick_and_place] DEBUG MOVE_ABOVE_PLACE start: ee_pos={current_ee_position} goal={goal}", flush=True)
            if self._advance("MOVE_ABOVE_PLACE"):
                print(f"[pick_and_place] DEBUG MOVE_ABOVE_PLACE end: ee_pos={current_ee_position} goal={goal}", flush=True)
                self._phase = Phase.DESCEND_TO_PLACE

        elif self._phase == Phase.DESCEND_TO_PLACE:
            self.robot.set_end_effector_pose(position=self.place_position, orientation=goal_orientation, ik_method=IK_METHOD)
            if self._advance("DESCEND_TO_PLACE"):
                self._phase = Phase.DETACH

        elif self._phase == Phase.DETACH:
            if self._step == 0:
                box_pos_before, _ = self.box.get_world_poses()
                print(
                    f"[pick_and_place] DEBUG detaching: box_pos={box_pos_before.numpy()[0]} "
                    f"ee_pos={current_ee_position} place_target={self.place_position} "
                    f"attach_offset={self._attach_offset}",
                    flush=True,
                )
                self.box.set_enabled_rigid_bodies([True])
            if self._advance("DETACH"):
                self._phase = Phase.RETRACT

        elif self._phase == Phase.RETRACT:
            goal = self.place_position + np.array([0.0, 0.0, APPROACH_HEIGHT])
            self.robot.set_end_effector_pose(position=goal, orientation=goal_orientation, ik_method=IK_METHOD)
            if self._advance("RETRACT"):
                self._phase = Phase.WAITING

    @property
    def phase_name(self) -> str:
        return self._phase.name
