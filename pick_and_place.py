"""Magic-attach pick-and-place: UR20 moves a box from a pick-zone conveyor to a
place-zone conveyor on another loop, driven by NVIDIA cuMotion's GPU RMPflow
motion planner (isaacsim.robot_motion.cumotion) rather than raw per-tick
differential IK - smooth, collision-aware motion instead of a stateless
one-shot IK solve every physics tick (see README's "Known gaps" history for
why that mattered: the previous UR10 + raw-IK version produced visibly jerky
motion and let the approaching arm physically shove the box out of place
before "attaching" it).

No real grasp physics: the box is "magic attached" by disabling its rigid
body and teleporting it to follow the end effector at a fixed offset, then
re-enabled at the place location. Unlike the previous version, the box's
rigid body is disabled the MOMENT it's selected as the pick target (on the
WAITING -> MOVE_ABOVE_PICK transition), not at the ATTACH phase - so the
approaching arm can never physically collide with and displace it first
(previously observed: the box ending up ~0.34 m from the assumed pick point
by the time ATTACH ran, because the arm's own collision geometry nudged it
during the approach while it was still a live rigid body). The box's pose is
queried directly (privileged / ground truth) - no perception involved. See
conveyor_indexer.py for how the pick zone's belt is held stopped while
occupied, and how it resumes once the robot has removed the box ("starved" ->
next box advances).

Phase transitions are success-gated (converged end-effector pose), not
fixed-duration: each phase requires both a small minimum-step floor (avoids a
one-tick convergence fluke) and the tool frame's world position landing
within EE_POSITION_THRESHOLD of that phase's target: falls back to the old
fixed tick budget only as a logged timeout safety net if convergence never
happens. This mirrors the pattern used by Isaac Sim's own cuMotion
pick-and-place tutorial
(standalone_examples/tutorials/manipulation/tutorial_9_pick_place_cumotion.py).
"""

from __future__ import annotations

import re
from enum import IntEnum

import numpy as np
import warp as wp
from pxr import Gf, Usd, UsdGeom, UsdPhysics

import isaacsim.core.experimental.utils.stage as stage_utils
import isaacsim.robot_motion.experimental.motion_generation as mg
from isaacsim.core.experimental.objects import Cone, Cylinder, Mesh
from isaacsim.core.experimental.prims import Articulation, GeomPrim, RigidPrim
from isaacsim.robot_motion.cumotion import CumotionWorldInterface, RmpFlowController, load_cumotion_robot
from isaacsim.storage.native import get_assets_root_path

# isaacsim.robot_motion.cumotion's transforms.py / cumotion_world_interface.py
# call np.reshape(arr, shape=[...]) (confirmed via grep, 6 call sites across
# those 2 files) - the `shape=` keyword to np.reshape only exists from NumPy
# 2.1 onward, but this Isaac Sim install's own bundled NumPy is 1.26.4
# (confirmed), so every one of those calls raises `TypeError: reshape() got
# an unexpected keyword argument 'shape'`. This is a genuine version mismatch
# in the shipped extension code, not specific to this robot - even the
# bundled/officially-supported UR10 cuMotion example hits the identical
# error. Patched here only in this process (never touches the shared Isaac
# Sim installation) as a small, reversible compatibility shim.
_np_reshape = np.reshape


def _reshape_shape_kwarg_compat(a, *args, **kwargs):
    if "shape" in kwargs:
        kwargs["newshape"] = kwargs.pop("shape")
    return _np_reshape(a, *args, **kwargs)


np.reshape = _reshape_shape_kwarg_compat

# Diagnostic test: with this forced True, the arm converged to 0.95 m from
# the pick target (WORSE than the 0.54 m with obstacles tracked normally) -
# conclusively ruling out collision-avoidance repulsion as the cause of the
# stuck-short-of-target behavior. Left False; obstacle tracking is correct
# and helpful, just not the actual blocker.
_DEBUG_DISABLE_OBSTACLE_TRACKING = False

UR20_CONFIG_DIR = "/home/ubuntu/conveyor_indexing/robot_configs/ur20"
TOOL_FRAME_NAME = "tool0"
# The real USD prim mirroring tool0's transform exactly (see
# robot_configs/generate_ur20_urdf.py: tool0 was added to the URDF/XRDF as a
# fixed frame off this prim's actual authored transform, but "tool0" itself
# is not a real prim in ur20.usd - this "flange" Xform is, so it's used here
# to query the tool frame's live world pose from the running simulation).
TOOL_FRAME_LIVE_PRIM_SUBPATH = "wrist_3_link/flange"

# "Ready" home pose: tool0 directly above the robot's own base (0.5 m below
# it) with the SAME DOWN_ORIENTATION used for pick/place targets - not
# UR10's arbitrary bent-elbow seed reused unchanged (which, empirically
# checked, left the flange's Z axis pointing horizontally rather than down,
# meaning every pick/place target demanded a large simultaneous
# reorientation + 1.2 m translation from a cold start; the arm was observed
# getting stuck well short of the target). Derived by literally running
# RmpFlowController to convergence against this target in an empty,
# obstacle-free scene (matching smoke_test_ur20_rmpflow.py's setup) and
# reading back the converged joint angles - not hand-picked. In
# cspace.joint_names order (shoulder_pan, shoulder_lift, elbow, wrist_1,
# wrist_2, wrist_3).
UR20_DEFAULT_JOINT_POSITIONS = [2.583766, -0.523898, -0.007470, -0.872193, 1.125949, 0.148515]


def _local_z_axis_in_world(orientation_wxyz: np.ndarray) -> np.ndarray:
    """Diagnostic helper: where does this quaternion's local +Z axis point in world frame?

    Should read close to (0, 0, -1) if tool0 is actually oriented "straight
    down" as intended - direct numeric confirmation rather than inferring
    it from a screenshot or from position-error convergence alone.
    """
    w, x, y, z = orientation_wxyz
    return np.array(
        [
            2 * (x * z + y * w),
            2 * (y * z - x * w),
            1 - 2 * (x * x + y * y),
        ]
    )


APPROACH_HEIGHT = 0.15  # meters above the grasp/place point while transiting

# A zone reporting Machine=IDLE (held) means the belt has stopped commanding
# motion, but a box arriving via INDUCTING still carries real momentum and
# was observed coasting/sliding several meters in the ~1s it takes to
# capture a pick point and descend - the belt stopping doesn't brake it.
# Require linear speed below this before treating a box as an actual pick
# target, not just "the zone reports occupied+idle."
PICK_SETTLE_LINEAR_SPEED = 0.02  # m/s

# Downward-facing tool0 orientation (w, x, y, z): 180 deg about world X, so
# tool0's local Z axis (the approach axis, per the UR flange convention)
# points straight down (world -Z) and tool0's local X axis stays aligned
# with world X - "upright w.r.t. X/Y, fixed yaw" per the intended approach:
# move to a pose above the package with this orientation already set, then
# descend straight down without changing it (mirrored for place).
DOWN_ORIENTATION = np.array([0.0, 1.0, 0.0, 0.0])

# EE-position convergence tolerance for phase advancement, and the minimum
# tick floor before convergence is even checked (avoids a one-tick
# convergence fluke) - same values used by Isaac Sim's own cuMotion
# pick-and-place tutorial (_EE_THRESHOLD / _MIN_STEPS in
# tutorial_9_pick_place_cumotion.py).
EE_POSITION_THRESHOLD = 0.02  # meters
MIN_STEPS_PER_PHASE = 15  # ticks, at 120 Hz physics (~0.125 s)

# Per-phase tick budget - now a TIMEOUT SAFETY NET (logged if hit), not the
# primary advance signal; convergence via EE_POSITION_THRESHOLD is. The old
# fixed-duration values (60-150 ticks) were tuned for the previous raw-IK
# approach (an instantaneous per-tick snap toward a linearly-interpolated
# target) and were confirmed far too short for RMPflow's genuinely
# dynamics/jerk-limited motion - every phase was hitting its timeout without
# converging, the end effector still well short of the target. Rescaled
# instead from Isaac Sim's own cuMotion pick-and-place tutorial's
# `events_dt` (`tutorial_9_pick_place_cumotion.py`, `[250, 150, 100, 50, 150,
# 100, 100, 100]` ticks at its 60 Hz physics rate) to this repo's 120 Hz
# physics rate (x2), mapped onto our 8 phases (ATTACH/DETACH have no arm
# motion, same as the tutorial's gripper phases, so kept short).
# Bumped further from the tutorial-derived starting point above: after
# fixing the position/orientation gain imbalance, X/Y position was
# confirmed converging within 3 cm, but Z and full orientation convergence
# were still in progress (not stuck - genuinely still closing) when these
# budgets were hit. Doubled to give that real, ongoing convergence room to
# finish rather than cutting it off early.
PHASE_TICKS = {
    "MOVE_ABOVE_PICK": 1000,
    "DESCEND_TO_PICK": 600,
    "ATTACH": 5,
    "LIFT": 200,
    "MOVE_ABOVE_PLACE": 600,
    "DESCEND_TO_PLACE": 400,
    "DETACH": 5,
    "RETRACT": 400,
}

# Phases where a fresh RMPflow leg begins (controller.reset()) rather than
# continuing the previous leg's trajectory with just an updated setpoint -
# mirrors the cuMotion tutorial's reset points (start of pre-grasp, start of
# lift after grasp, start of retract after release): each is the first
# motion phase following either the initial WAITING state or a discrete,
# non-RMPflow-controlled event (attach/detach).
RESET_ON_ENTRY_PHASES = frozenset({"MOVE_ABOVE_PICK", "LIFT", "RETRACT"})


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
) -> Articulation:
    """Create a simple static cylindrical pedestal and a UR20 on top of it.

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

    # Position the robot via plain USD BEFORE wrapping it in the Articulation
    # class, rather than via Articulation.set_world_poses() afterward. The
    # latter was verified (via get_world_poses() reading back (0, 0, 0) both
    # immediately after the call and again after world.reset()) to silently
    # not reposition the robot at this stage of initialization -
    # Articulation.set_world_poses() targets the articulation's root_joint
    # frame, which apparently doesn't take a plain Xform write at this point.
    # Setting the reference prim's own transform directly, before any
    # Articulation/PhysX wrapping exists, sidesteps that entirely.
    usd_path = get_assets_root_path() + "/Isaac/Robots/UniversalRobots/ur20/ur20.usd"
    robot_prim = stage_utils.add_reference_to_stage(usd_path=usd_path, path=robot_path, variants=[])
    xformable = UsdGeom.Xformable(robot_prim)
    xformable.ClearXformOpOrder()
    xformable.AddTranslateOp().Set(Gf.Vec3d(px, py, pz + pedestal_height))

    robot = Articulation(robot_path)
    robot.set_default_state(dof_positions=UR20_DEFAULT_JOINT_POSITIONS)
    return robot


def _build_rmpflow_controller(
    robot: Articulation, robot_path: str, exclude_obstacle_paths: list[str]
) -> tuple[RmpFlowController, mg.WorldBinding]:
    """Load the generated UR20 cuMotion config and wire up collision-aware RMPflow.

    Args:
        exclude_obstacle_paths: Prim paths to exclude from the obstacle set
            scanned for collision avoidance - the robot itself and every box
            that could ever be the pick/place target (a carried or
            about-to-be-picked box must never register as an obstacle to
            dodge, same as Isaac Sim's own cuMotion pick-and-place tutorial
            excludes the cube it manipulates).
    """
    cumotion_robot = load_cumotion_robot(directory=UR20_CONFIG_DIR)
    tool_frames = cumotion_robot.robot_description.tool_frame_names()
    if TOOL_FRAME_NAME not in tool_frames:
        raise RuntimeError(f"Expected tool frame '{TOOL_FRAME_NAME}' not found in generated XRDF: {tool_frames}")

    robot_pos, robot_ori = robot.get_world_poses()

    obstacle_strategy = mg.ObstacleStrategy()
    for prim_type in (Mesh, Cone, Cylinder):
        obstacle_strategy.set_default_configuration(prim_type, mg.ObstacleConfiguration("obb", 0.01))

    # conveyor_setup.usd trips WorldBinding.initialize() two different ways,
    # both pre-existing scene-authoring quirks (same category as the stray
    # DistantLight ConveyorNode documented in the README - not something
    # this scaffold edits into the source scene), not bugs in this code:
    #   - some ConveyorTrack prims have a non-unity-scaled ancestor
    #     (AssertionError: "non-unity scaling").
    #   - the belt geometry itself isn't one of WorldBinding's supported
    #     obstacle shape types - Sphere/Cube/Cone/Plane/Capsule/Cylinder/Mesh
    #     (RuntimeError: "does not point to a supported shape type"),
    #     observed on every ConveyorTrack*/Belt prim tried so far, so
    #     potentially all 16 belt segments hit this one at a time.
    # Both name the exact offending prim path in their message; drop it and
    # retry rather than crash. Those specific prims just don't participate
    # in RMPflow's obstacle avoidance - a completeness gap (this many belt
    # segments not treated as obstacles), not a correctness bug (ordinary
    # PhysX collision is unaffected). Capped generously (not just 1-2
    # retries) since potentially every one of the 16 tracks' belt prims
    # needs excluding this way, one at a time.
    exclude_paths = list(exclude_obstacle_paths)
    max_attempts = 40
    for attempt in range(max_attempts):
        tracked_prims = (
            []
            if _DEBUG_DISABLE_OBSTACLE_TRACKING
            else mg.SceneQuery().get_prims_in_aabb(
                search_box_origin=robot_pos.numpy()[0],
                search_box_minimum=[-10.0, -10.0, -10.0],
                search_box_maximum=[10.0, 10.0, 10.0],
                tracked_api=mg.TrackableApi.PHYSICS_COLLISION,
                exclude_prim_paths=exclude_paths,
            )
        )
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
            if attempt >= max_attempts - 1:
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
            print(
                f"[pick_and_place] WARNING: excluding from RMPflow's obstacle set "
                f"(pre-existing conveyor_setup.usd quirk, see comment above): {offending_paths}",
                flush=True,
            )
            exclude_paths = exclude_paths + offending_paths
    world_binding.get_world_interface().update_world_to_robot_root_transforms(poses=(robot_pos, robot_ori))
    world_binding.synchronize_transforms()

    controller = RmpFlowController(
        cumotion_robot=cumotion_robot,
        cumotion_world_interface=world_binding.get_world_interface(),
        robot_joint_space=list(robot.dof_names),
        robot_site_space=tool_frames,
        tool_frame=TOOL_FRAME_NAME,
    )
    return controller, world_binding


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
        robot: Articulation,
        robot_path: str,
        place_xy: tuple,
        place_belt_top_z: float,
        box_rigid_prims: dict,
        physics_dt: float,
        extra_exclude_obstacle_paths: list[str] = (),
    ) -> None:
        self.robot = robot
        self.place_xy = place_xy
        self.place_belt_top_z = place_belt_top_z
        self.box_rigid_prims = box_rigid_prims
        self._physics_dt = physics_dt

        self.controller, self.world_binding = _build_rmpflow_controller(
            robot,
            robot_path,
            exclude_obstacle_paths=[robot_path, *box_rigid_prims.keys(), *extra_exclude_obstacle_paths],
        )
        wrist_link_name, flange_subprim_name = TOOL_FRAME_LIVE_PRIM_SUBPATH.split("/")
        link_names = list(robot.link_names)
        wrist_link_path = robot.link_paths[0][link_names.index(wrist_link_name)]
        self._tool_prim = GeomPrim(paths=f"{wrist_link_path}/{flange_subprim_name}")

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
        self._t = 0.0
        self._pick_point: np.ndarray | None = None
        self._attach_offset: np.ndarray | None = None
        self._attach_orientation: np.ndarray | None = None

    def _box_top_center(self) -> np.ndarray:
        """Privileged query: the box's current world-space top-face center."""
        position, _ = self.box.get_world_poses()
        center = position.numpy()[0]
        return center + np.array([0.0, 0.0, self._box_half_height])

    def _tool_world_position(self) -> np.ndarray:
        return self._tool_prim.get_world_poses()[0].numpy()[0]

    def _estimated_state(self) -> "mg.RobotState":
        names = list(self.robot.dof_names)
        return mg.RobotState(
            joints=mg.JointState.from_name(
                robot_joint_space=names,
                positions=(names, self.robot.get_dof_positions()),
                velocities=(names, self.robot.get_dof_velocities()),
            )
        )

    def _setpoint_state(self, target_position: np.ndarray) -> "mg.RobotState":
        return mg.RobotState(
            sites=mg.SpatialState.from_name(
                spatial_space=[TOOL_FRAME_NAME],
                positions=([TOOL_FRAME_NAME], wp.array([target_position.tolist()], dtype=wp.float32)),
                orientations=([TOOL_FRAME_NAME], wp.array([DOWN_ORIENTATION.tolist()], dtype=wp.float32)),
            ),
        )

    def _drive_to(self, target_position: np.ndarray, phase_name: str, reset_leg: bool) -> bool:
        """Advance RMPflow toward target_position; return True once this phase should end.

        Success-gated: requires both MIN_STEPS_PER_PHASE and the tool frame
        landing within EE_POSITION_THRESHOLD of target_position: falls back
        to PHASE_TICKS[phase_name] as a logged timeout otherwise.
        """
        if reset_leg and self._step == 0:
            if not self.controller.reset(self._estimated_state(), self._setpoint_state(target_position), t=0.0):
                raise RuntimeError(f"RmpFlowController.reset() failed entering {phase_name}")
            self._t = 0.0

        self.world_binding.get_world_interface().update_world_to_robot_root_transforms(
            poses=self.robot.get_world_poses()
        )
        self.world_binding.synchronize_transforms()

        desired = self.controller.forward(self._estimated_state(), self._setpoint_state(target_position), self._t)
        if desired is not None and desired.joints.positions is not None:
            self.robot.set_dof_position_targets(positions=desired.joints.positions, dof_indices=desired.joints.position_indices)

        self._t += self._physics_dt
        self._step += 1

        converged = self._step >= MIN_STEPS_PER_PHASE and float(
            np.linalg.norm(self._tool_world_position() - target_position)
        ) < EE_POSITION_THRESHOLD
        timed_out = self._step >= PHASE_TICKS[phase_name]
        if timed_out and not converged:
            print(f"[pick_and_place] {phase_name} timed out after {PHASE_TICKS[phase_name]} ticks without converging", flush=True)
        if converged or timed_out:
            self._step = 0
            return True
        return False

    def forward(self, pick_ready: bool, box_path: str | None) -> None:
        # While holding the box, keep it rigidly offset from the end effector -
        # read BEFORE issuing this tick's motion command so it tracks the
        # actually-reached pose rather than lagging the aspirational target.
        if self._phase in (Phase.LIFT, Phase.MOVE_ABOVE_PLACE, Phase.DESCEND_TO_PLACE):
            self.box.set_world_poses(
                positions=self._tool_world_position() + self._attach_offset,
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
                # Disable the box's rigid body NOW, not at ATTACH - it must
                # never be physically contactable by the approaching arm
                # (see module docstring: this is what fixed the ~0.34 m
                # attach-offset bug). It stays a frozen kinematic body,
                # exactly at _pick_point, all the way through ATTACH.
                self.box.set_enabled_rigid_bodies([False])
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
            if self._drive_to(goal, "MOVE_ABOVE_PICK", reset_leg="MOVE_ABOVE_PICK" in RESET_ON_ENTRY_PHASES):
                self._phase = Phase.DESCEND_TO_PICK

        elif self._phase == Phase.DESCEND_TO_PICK:
            if self._drive_to(self._pick_point, "DESCEND_TO_PICK", reset_leg=False):
                print(
                    f"[pick_and_place] DEBUG DESCEND_TO_PICK end: ee_pos={self._tool_world_position()} "
                    f"pick_point={self._pick_point} box={self.box.paths} "
                    f"tool_z_axis_world={_local_z_axis_in_world(self._tool_prim.get_world_poses()[1].numpy()[0])}",
                    flush=True,
                )
                self._phase = Phase.ATTACH

        elif self._phase == Phase.ATTACH:
            if self._step == 0:
                box_position, box_orientation = self.box.get_world_poses()
                self._attach_offset = box_position.numpy()[0] - self._tool_world_position()
                self._attach_orientation = box_orientation.numpy()[0]
                print(
                    f"[pick_and_place] DEBUG ATTACH: ee_pos={self._tool_world_position()} "
                    f"box_pos={box_position.numpy()[0]} attach_offset={self._attach_offset}",
                    flush=True,
                )
            self._step += 1
            if self._step >= PHASE_TICKS["ATTACH"]:
                self._step = 0
                self._phase = Phase.LIFT

        elif self._phase == Phase.LIFT:
            goal = self._pick_point + np.array([0.0, 0.0, APPROACH_HEIGHT])
            if self._drive_to(goal, "LIFT", reset_leg="LIFT" in RESET_ON_ENTRY_PHASES):
                self._phase = Phase.MOVE_ABOVE_PLACE

        elif self._phase == Phase.MOVE_ABOVE_PLACE:
            goal = self.place_position + np.array([0.0, 0.0, APPROACH_HEIGHT])
            if self._step == 0:
                print(f"[pick_and_place] DEBUG MOVE_ABOVE_PLACE start: ee_pos={self._tool_world_position()} goal={goal}", flush=True)
            if self._drive_to(goal, "MOVE_ABOVE_PLACE", reset_leg=False):
                print(f"[pick_and_place] DEBUG MOVE_ABOVE_PLACE end: ee_pos={self._tool_world_position()} goal={goal}", flush=True)
                self._phase = Phase.DESCEND_TO_PLACE

        elif self._phase == Phase.DESCEND_TO_PLACE:
            if self._drive_to(self.place_position, "DESCEND_TO_PLACE", reset_leg=False):
                self._phase = Phase.DETACH

        elif self._phase == Phase.DETACH:
            if self._step == 0:
                box_pos_before, _ = self.box.get_world_poses()
                print(
                    f"[pick_and_place] DEBUG detaching: box_pos={box_pos_before.numpy()[0]} "
                    f"ee_pos={self._tool_world_position()} place_target={self.place_position} "
                    f"attach_offset={self._attach_offset}",
                    flush=True,
                )
                self.box.set_enabled_rigid_bodies([True])
            self._step += 1
            if self._step >= PHASE_TICKS["DETACH"]:
                self._step = 0
                self._phase = Phase.RETRACT

        elif self._phase == Phase.RETRACT:
            goal = self.place_position + np.array([0.0, 0.0, APPROACH_HEIGHT])
            if self._drive_to(goal, "RETRACT", reset_leg="RETRACT" in RESET_ON_ENTRY_PHASES):
                self._phase = Phase.WAITING

    @property
    def phase_name(self) -> str:
        return self._phase.name
