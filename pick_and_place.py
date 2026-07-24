"""Magic-attach pick-and-place: UR20 moves a box from a pick-zone conveyor to a
place-zone conveyor on another loop, driven by NVIDIA cuMotion's
GraphBasedMotionPlanner (isaacsim.robot_motion.cumotion) rather than raw
per-tick differential IK or a reactive controller - collision-aware motion
instead of a stateless one-shot IK solve every physics tick (see README's
"Known gaps" history for why that mattered: the previous UR10 + raw-IK
version produced visibly jerky motion and let the approaching arm physically
shove the box out of place before "attaching" it).
"""

from __future__ import annotations

import math
import os
import re
from enum import IntEnum

import cumotion
import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics
from scipy.spatial.transform import Rotation

import isaacsim.core.experimental.utils.stage as stage_utils
import isaacsim.core.experimental.utils.xform as xform_utils
import isaacsim.robot_motion.experimental.motion_generation as mg
from isaacsim.core.experimental.objects import Cone, Cylinder, Mesh
from isaacsim.core.experimental.prims import Articulation, GeomPrim, RigidPrim
from isaacsim.robot_motion.cumotion import (
    CumotionRobot,
    CumotionWorldInterface,
    GraphBasedMotionPlanner,
    load_cumotion_robot,
)

from isaacsim.robot_motion.cumotion.impl.utils import isaac_sim_to_cumotion_pose
from isaacsim.storage.native import get_assets_root_path

# isaacsim.robot_motion.cumotion's shipped code calls np.reshape(arr,
# shape=[...]) - the `shape=` keyword only exists from NumPy 2.1 onward, but
# this Isaac Sim install's bundled NumPy is 1.26.4, so every such call raises
# TypeError. Patched here, in this process only (never touches the shared
# Isaac Sim installation), rather than editing the vendored file. Must
# capture the real np.reshape BEFORE reassigning np.reshape below - calling
# `np.reshape` from inside the wrapper itself (instead of this captured
# reference) recurses infinitely, since by then `np.reshape` IS this wrapper
# (confirmed: this exact mistake was in place here and crashed with
# `RecursionError` on the very first obstacle add call).
_np_reshape = np.reshape


def _reshape_shape_kwarg_compat(a, *args, **kwargs):
    if "shape" in kwargs:
        kwargs["newshape"] = kwargs.pop("shape")
    return _np_reshape(a, *args, **kwargs)


np.reshape = _reshape_shape_kwarg_compat

_DEBUG_DISABLE_OBSTACLE_TRACKING = False

UR20_CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "robot_configs", "ur20")
TOOL_FRAME_NAME = "tool0"
TOOL_FRAME_LIVE_PRIM_SUBPATH = "wrist_3_link/flange"
UR20_DEFAULT_JOINT_POSITIONS = [1.5708, -1.5708, -1.5708, -1.5708, 1.5708, 3.1415]
UR20_PRE_PLACE_JOINT_POSITIONS = [-1.5708, -1.5708, -1.5708, -1.5708, 1.5708, 3.1415]

# Same pose as UR20_PRE_PLACE_JOINT_POSITIONS (shoulder_pan +-360deg apart,
# within its +-2*pi range) but reached via +180deg instead of -90deg, so the
# STAGE_FOR_PICK<->STAGE_FOR_PLACE swing arcs the other way round - confirmed
# via FK probe that the direct joint-space path's midpoint swings toward
# local -X normally, +X with this. Use for a robot with a neighbor on its -X
# side (see MagicAttachPickPlace's pre_place_joint_positions param).
UR20_PRE_PLACE_JOINT_POSITIONS_AWAY = [
    UR20_DEFAULT_JOINT_POSITIONS[0] + math.pi
] + UR20_PRE_PLACE_JOINT_POSITIONS[1:]


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


def _rotation_matrix_to_quaternion_wxyz(m: np.ndarray) -> np.ndarray:
    """Standard 3x3 rotation matrix -> quaternion (w, x, y, z), for m acting as p' = m @ p."""
    tr = np.trace(m)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    return np.array([w, x, y, z])

ATTACH_MAX_DISTANCE = 0.05  # meters (5 cm)
PICK_SETTLE_LINEAR_SPEED = 0.02  # m/s
#Down is weird
DOWN_ORIENTATION = np.array([-0.5, 0.5, -0.5, 0.5])


def _quat_wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    return np.array([q[1], q[2], q[3], q[0]])


def _quat_xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    return np.array([q[3], q[0], q[1], q[2]])


# Place orientation: DOWN_ORIENTATION rotated 180 deg about the world Z axis
PLACE_ORIENTATION = _quat_xyzw_to_wxyz(
    (
        Rotation.from_euler("z", 180, degrees=True) * Rotation.from_quat(_quat_wxyz_to_xyzw(DOWN_ORIENTATION))
    ).as_quat()
)

# Per-phase tick budget 
PHASE_TICKS = {
    "STAGE_FOR_PICK": 600,
    "DESCEND_TO_PICK": 1200,
    "STAGE_FOR_PLACE": 600,
    "DESCEND_TO_PLACE": 1000,
}

MAX_JOINT_VELOCITIES = [2.0, 2.0, 2.0, 2.0, 2.0, 2.0]
MAX_JOINT_ACCELERATIONS = [10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
PLANNING_RETRIES = 3
IK_CSPACE_LIMIT_BIASING_WEIGHT = 1.0  # relative weight; see IkConfig docs


class Phase(IntEnum):
    WAITING = 0
    STAGE_FOR_PICK = 1
    DESCEND_TO_PICK = 2
    ATTACH = 3
    STAGE_FOR_PLACE = 4
    DESCEND_TO_PLACE = 5
    DETACH = 6


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
    usd_path = get_assets_root_path() + "/Isaac/Robots/UniversalRobots/ur20/ur20.usd"
    robot_prim = stage_utils.add_reference_to_stage(usd_path=usd_path, path=robot_path, variants=[])
    xformable = UsdGeom.Xformable(robot_prim)
    xformable.ClearXformOpOrder()
    xformable.AddTranslateOp().Set(Gf.Vec3d(px, py, pz + pedestal_height))

    robot = Articulation(robot_path)
    robot.set_default_state(dof_positions=UR20_DEFAULT_JOINT_POSITIONS)
    return robot


def _build_motion_planner(
    robot: Articulation, robot_path: str, exclude_obstacle_paths: list[str]
) -> tuple[GraphBasedMotionPlanner, mg.WorldBinding, CumotionRobot]:
    """Load the generated UR20 cuMotion config and wire up a collision-aware GraphBasedMotionPlanner.

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
    obstacle_strategy.set_default_configuration(Mesh, mg.ObstacleConfiguration("triangulated_mesh", 0.005))
    obstacle_strategy.set_default_configuration(Cone, mg.ObstacleConfiguration("obb", 0.0))
    obstacle_strategy.set_default_configuration(Cylinder, mg.ObstacleConfiguration("obb", 0.0))
    exclude_paths = list(exclude_obstacle_paths)
    max_attempts = 40

    # Isaac Sim bug workaround (filed upstream, not fixed as of this writing):
    # WorldBinding.initialize() wraps every tracked Mesh-type obstacle in an
    # isaacsim.core.experimental.objects.Mesh (world_binding.py's
    # _add_mesh_from_prim/_add_triangulated_mesh_from_prim), and that class
    # defaults reset_xform_op_properties=True. reset_xform_op_properties()
    # deletes xformOp:rotateXYZ
    rotation_guard_paths = (
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
    stage = stage_utils.get_current_stage()
    rotation_guard_snapshot = {}
    for guard_path in rotation_guard_paths:
        local_matrix = UsdGeom.Xformable(stage.GetPrimAtPath(guard_path)).GetLocalTransformation(Usd.TimeCode.Default())
        rotation_guard_snapshot[guard_path] = Gf.Transform(local_matrix).GetRotation().GetQuat()

    try:
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
                    f"[pick_and_place] WARNING: excluding from the planner's obstacle set "
                    f"(pre-existing conveyor_setup.usd quirk, see comment above): {offending_paths}",
                    flush=True,
                )
                exclude_paths = exclude_paths + offending_paths
    finally:
        for guard_path, original_local_quat in rotation_guard_snapshot.items():
            guard_prim = stage.GetPrimAtPath(guard_path)
            if guard_prim.IsValid() and "xformOp:orient" in guard_prim.GetPropertyNames():
                # Match whatever precision xformOp:orient is actually
                # authored at (Gf.Quatf vs Gf.Quatd) rather than assuming
                # Quatd unconditionally - Usd.Attribute.Set() raises a type-
                # mismatch Tf error otherwise on any guarded prim whose
                # orient happens to be single-precision (confirmed against
                # 5_conv_env.usd's SteelBoxTruck body mesh, authored as
                # `quatf`; racetrack.usd apparently never exercised this
                # restore path against a quatf-typed prim). GetRotation().GetQuat()
                # above always returns a Quatd regardless of the source
                # attribute's precision, so the mismatch is with THIS write,
                # not the snapshot.
                orient_attr = guard_prim.GetAttribute("xformOp:orient")
                quat_type = type(orient_attr.Get()) if orient_attr.Get() is not None else Gf.Quatd
                orient_attr.Set(quat_type(original_local_quat))

    world_binding.get_world_interface().update_world_to_robot_root_transforms(poses=(robot_pos, robot_ori))
    world_binding.synchronize_transforms()

    planner = GraphBasedMotionPlanner(
        cumotion_robot=cumotion_robot,
        cumotion_world_interface=world_binding.get_world_interface(),
        tool_frame=TOOL_FRAME_NAME,
    )
    return planner, world_binding, cumotion_robot


class MagicAttachPickPlace:
    """Phase state machine: pick a box off the pick-zone belt, place it on the place-zone belt.

    Call ``forward(pick_ready, box_path)`` once per physics step. The scene
    can have multiple physics-enabled boxes cycling the pick loop (see
    conveyor_indexer.py's KNOWN_BOX_PATHS), so the box to
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
        default_joint_positions: list[float] = UR20_DEFAULT_JOINT_POSITIONS,
        pre_place_joint_positions: list[float] = UR20_PRE_PLACE_JOINT_POSITIONS,
    ) -> None:
        self.robot = robot
        self.place_xy = place_xy
        self.place_belt_top_z = place_belt_top_z
        self.box_rigid_prims = box_rigid_prims
        self._physics_dt = physics_dt
        # See UR20_PRE_PLACE_JOINT_POSITIONS_AWAY for when to override these.
        self._default_joint_positions = default_joint_positions
        self._pre_place_joint_positions = pre_place_joint_positions

        self.planner, self.world_binding, self._cumotion_robot = _build_motion_planner(
            robot,
            robot_path,
            exclude_obstacle_paths=[robot_path, *box_rigid_prims.keys(), *extra_exclude_obstacle_paths],
        )
        wrist_link_name, flange_subprim_name = TOOL_FRAME_LIVE_PRIM_SUBPATH.split("/")
        link_names = list(robot.link_names)
        self._wrist_link_path = robot.link_paths[0][link_names.index(wrist_link_name)]
        self._tool_prim = GeomPrim(paths=f"{self._wrist_link_path}/{flange_subprim_name}")
        # Fixed joint path is fixed for the object's lifetime - only ever one
        # box attached at a time, created fresh at ATTACH and deleted at
        # DETACH (see _attach_box/_detach_box).
        self._attach_joint_path = f"{self._wrist_link_path}/BoxAttachJoint"

        # Depends on which box is being carried (set per-cycle below, since
        # box_rigid_prims isn't guaranteed to hold same-size boxes) - box
        # bottom flush on the belt means top-center (what the end
        # effector targets, same convention as _pick_point) sits at
        # belt_top_z + 2*box_half_height.
        self.place_position: np.ndarray | None = None

        self.box: RigidPrim | None = None
        self._box_half_height: float | None = None

        self._phase = Phase.WAITING
        self._step = 0
        self._t = 0.0
        self._trajectory = None
        self._pick_point: np.ndarray | None = None
        # Which of STAGE_FOR_PICK/STAGE_FOR_PLACE's two visits per cycle this
        # is: False on the way to a descent, True on the way back up after
        # ATTACH/DETACH (see forward()'s STAGE_FOR_PICK/STAGE_FOR_PLACE
        # branches and module docstring).
        self._holding_box = False

    def _box_top_center(self) -> np.ndarray:
        """Privileged query: the box's current world-space top-face center.

        `get_world_poses()` returns the box's `xformOp:translate` origin,
        which for these prims is its BOTTOM face, not its center - confirmed
        directly against 5_conv_env.usd's authored data: e.g.
        CubeBox_A04_26cm_PR_NVD_01's translate.z (1.7805...) exactly matches
        ConveyorTrack's belt-top Z, not the ~0.13 m higher value a
        center-origin box resting on that same belt would have. Adding only
        `_box_half_height` therefore reaches the box's MIDDLE, not its top -
        confirmed as the cause of a real bug (the tool descending into the
        box rather than stopping at its top surface) observed running the
        full scaffold. The full height (2x half_height) is what actually
        reaches the top - matching the convention `place_position` already
        uses for the equivalent "box's top surface height once its bottom
        rests on the belt" calculation.
        """
        position, _ = self.box.get_world_poses()
        bottom_center = position.numpy()[0]
        return bottom_center + np.array([0.0, 0.0, 2.0 * self._box_half_height])

    def _tool_world_position(self) -> np.ndarray:
        return self._tool_prim.get_world_poses()[0].numpy()[0]

    def _attach_box(self) -> None:
        """Rigidly attach self.box to wrist_3_link via a PhysX FixedJoint, so
        it moves as a real extension of the arm's kinematic chain (position
        AND orientation) rather than being teleported to a computed offset
        every tick.

        The box's rigid body was disabled since WAITING->MOVE_ABOVE_PICK (see
        module docstring) - re-enabled here, in the same tick the joint is
        created, so there's no gap where it could fall under gravity before
        the joint constrains it.
        """
        box_path = self.box.paths[0]
        # Box's current pose expressed in wrist_3_link's local frame - the
        # exact relative transform to preserve for the rest of the carry.
        relative_transform = xform_utils.get_relative_transform(box_path, self._wrist_link_path)
        local_pos0 = relative_transform[:3, 3]
        # CubeBox_* prims carry a non-unity xformOp:scale:unitsResolve
        # (0.01) in their own local transform chain, so this relative
        # transform's rotation block isn't a pure rotation - each column is
        # scaled, not unit-length. Left uncorrected, the box was observed
        # snapping to a wrong (effectively the tool's own) orientation the
        # instant the FixedJoint was created, instead of staying at
        # whatever orientation it actually had when grasped - confirmed via
        # running the full scaffold. Normalizing each column to unit length
        # first (the scale is uniform/diagonal here, no shear, so this
        # exactly removes it) recovers the true rotation before extracting
        # the quaternion. Position isn't affected by this - only the
        # rotation block carries the scale.
        rotation_block = relative_transform[:3, :3]
        rotation_block = rotation_block / np.linalg.norm(rotation_block, axis=0, keepdims=True)
        local_rot0 = _rotation_matrix_to_quaternion_wxyz(rotation_block)
        print(
            f"[pick_and_place] DEBUG attaching: box_path={box_path} "
            f"local_pos0(rel. wrist_3_link)={local_pos0} local_rot0(wxyz)={local_rot0}",
            flush=True,
        )

        self.box.set_enabled_rigid_bodies([True])

        stage = stage_utils.get_current_stage()
        joint = UsdPhysics.FixedJoint.Define(stage, self._attach_joint_path)
        joint.CreateBody0Rel().SetTargets([Sdf.Path(self._wrist_link_path)])
        joint.CreateBody1Rel().SetTargets([Sdf.Path(box_path)])
        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*local_pos0.tolist()))
        joint.CreateLocalRot0Attr().Set(Gf.Quatf(float(local_rot0[0]), Gf.Vec3f(*local_rot0[1:].tolist())))
        joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))

    def _detach_box(self) -> None:
        """Remove the FixedJoint created by _attach_box, releasing the box
        (already a live rigid body since _attach_box) to fall/settle
        naturally under gravity from wherever it currently is."""
        stage_utils.delete_prim(self._attach_joint_path)

    def _solve_ik_target(
        self, target_position: np.ndarray, orientation: np.ndarray, q_initial: np.ndarray
    ) -> np.ndarray:
        """Solve for a single joint configuration reaching (target_position, orientation),
        seeded at q_initial and biased toward the middle of each joint's range.

        Gives plan_to_cspace_target a concrete destination to plan to, instead
        of letting plan_to_pose_target's JtRRT settle on whatever
        pose-satisfying (but possibly contorted) configuration its random
        tree happens to reach first - see IK_CSPACE_LIMIT_BIASING_WEIGHT
        comment above.
        """
        position_world_to_base, quaternion_world_to_base = (
            self.world_binding.get_world_interface().get_world_to_robot_base_transform()
        )
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
            kinematics=self._cumotion_robot.kinematics,
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

    def _drive_to(
        self,
        target_position: np.ndarray | None,
        phase_name: str,
        orientation: np.ndarray = DOWN_ORIENTATION,
        use_ik_cspace_target: bool = False,
        cspace_target: np.ndarray | list[float] | None = None,
    ) -> bool:
        """Plan a fresh collision-free path to a target at phase entry, then play
        back the resulting trajectory open-loop; return True once this phase
        should end.

        Exactly one target flavor applies, in this order of precedence:
          - cspace_target: plan_to_cspace_target directly to a known joint
            configuration (e.g. UR20_DEFAULT_JOINT_POSITIONS /
            UR20_PRE_PLACE_JOINT_POSITIONS) - target_position/orientation are
            ignored.
          - use_ik_cspace_target: solve IK to one concrete joint target first
            (see _solve_ik_target), then plan_to_cspace_target to exactly
            that configuration, instead of plan_to_pose_target's task-space
            JtRRT.
          - otherwise: plan_to_pose_target(target_position, orientation)
            directly (task-space JtRRT).
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
                q_target = self._solve_ik_target(target_position, orientation, q_initial)
            else:
                q_target = None

            if q_target is not None:
                for attempt in range(PLANNING_RETRIES):
                    path = self.planner.plan_to_cspace_target(q_initial=q_initial, q_final=q_target)
                    if path is not None:
                        break
                    print(
                        f"[pick_and_place] WARNING: cspace planning attempt {attempt + 1}/{PLANNING_RETRIES} "
                        f"failed entering {phase_name} (q_target={q_target}), retrying",
                        flush=True,
                    )
            else:
                for attempt in range(PLANNING_RETRIES):
                    path = self.planner.plan_to_pose_target(
                        q_initial=q_initial, position=target_position, orientation=orientation
                    )
                    if path is not None:
                        break
                    print(
                        f"[pick_and_place] WARNING: planning attempt {attempt + 1}/{PLANNING_RETRIES} "
                        f"failed entering {phase_name} (target={target_position}), retrying",
                        flush=True,
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
            print(
                f"[pick_and_place] {phase_name} exceeded its {PHASE_TICKS[phase_name]}-tick budget "
                f"before the planned trajectory (duration={self._trajectory.duration:.2f}s) finished",
                flush=True,
            )
        if finished or timed_out:
            self._step = 0
            return True
        return False

    def forward(self, pick_ready: bool, box_path: str | None) -> None:
        # No per-tick box-following code needed here anymore - once
        # _attach_box() creates the FixedJoint, PhysX itself keeps the box
        # rigidly following wrist_3_link (position and orientation) for the
        # whole STAGE_FOR_PICK(carrying)/STAGE_FOR_PLACE/DESCEND_TO_PLACE carry.

        if self._phase == Phase.WAITING:
            if pick_ready and box_path is not None:
                # Track whichever box actually triggered pick_ready - not
                # necessarily the same one as last cycle (any box in
                # box_rigid_prims can occupy this zone). Re-checked every physics
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
                    f"[pick_and_place] DEBUG WAITING->STAGE_FOR_PICK: box_path={box_path} "
                    f"pick_point={self._pick_point} box_half_height={self._box_half_height} settled_speed={speed}",
                    flush=True,
                )
                # Same top-center convention as _pick_point, recomputed here
                # since this box's height may differ from the last one placed.
                self.place_position = np.array(
                    [self.place_xy[0], self.place_xy[1], self.place_belt_top_z + 2 * self._box_half_height]
                )
                self._phase = Phase.STAGE_FOR_PICK
                self._step = 0

        elif self._phase == Phase.STAGE_FOR_PICK:
            # Plain joint-space move to a known, mid-range posture (see
            # UR20_DEFAULT_JOINT_POSITIONS) - visited twice per cycle (see
            # _holding_box/module docstring): first as the staging point
            # before DESCEND_TO_PICK's reach, then again right after ATTACH
            # to lift the (now carried) box back up through the same pose,
            # before handing off to STAGE_FOR_PLACE.
            if self._drive_to(None, "STAGE_FOR_PICK", cspace_target=self._default_joint_positions):
                self._phase = Phase.STAGE_FOR_PLACE if self._holding_box else Phase.DESCEND_TO_PICK

        elif self._phase == Phase.DESCEND_TO_PICK:
            if self._drive_to(self._pick_point, "DESCEND_TO_PICK"):
                print(
                    f"[pick_and_place] DEBUG DESCEND_TO_PICK end: ee_pos={self._tool_world_position()} "
                    f"pick_point={self._pick_point} box={self.box.paths} "
                    f"tool_z_axis_world={_local_z_axis_in_world(self._tool_prim.get_world_poses()[1].numpy()[0])}",
                    flush=True,
                )
                self._phase = Phase.ATTACH

        elif self._phase == Phase.ATTACH:
            # Gated on real proximity, not a tick count: only create the
            # FixedJoint once the tool is actually at the package (see
            # ATTACH_MAX_DISTANCE) - DESCEND_TO_PICK's planned trajectory
            # finishing is a strong signal but not a guarantee (planning or
            # tracking error could leave it short).
            distance = float(np.linalg.norm(self._tool_world_position() - self._pick_point))
            if distance <= ATTACH_MAX_DISTANCE:
                self._attach_box()
                self._holding_box = True
                self._phase = Phase.STAGE_FOR_PICK

        elif self._phase == Phase.STAGE_FOR_PLACE:
            if self._drive_to(None, "STAGE_FOR_PLACE", cspace_target=self._pre_place_joint_positions):
                self._phase = Phase.DESCEND_TO_PLACE if self._holding_box else Phase.WAITING

        elif self._phase == Phase.DESCEND_TO_PLACE:
            if self._drive_to(
                self.place_position, "DESCEND_TO_PLACE", orientation=PLACE_ORIENTATION, use_ik_cspace_target=True
            ):
                self._phase = Phase.DETACH

        elif self._phase == Phase.DETACH:
            box_pos_before, _ = self.box.get_world_poses()
            print(
                f"[pick_and_place] DEBUG detaching: box_pos={box_pos_before.numpy()[0]} "
                f"ee_pos={self._tool_world_position()} place_target={self.place_position}",
                flush=True,
            )
            self._detach_box()
            self._holding_box = False
            self._phase = Phase.STAGE_FOR_PLACE

    @property
    def phase_name(self) -> str:
        return self._phase.name
