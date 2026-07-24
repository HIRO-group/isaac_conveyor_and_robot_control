"""Magic-attach pick-and-place (parent side): UR20 moves a box from a pick-zone
conveyor to a place-zone conveyor on another loop, using collision-aware
NVIDIA cuMotion motion planning.

The cuMotion solve itself runs OUT OF PROCESS (see planner_server.py /
planner_server_impl.py) because a single solve is a monolithic C++/CUDA call
that holds the Python GIL for its whole duration - running it in this process
froze the main sim loop until it returned. This module is the parent side: it
owns the pick/place phase state machine, ships each plan request to the planner
subprocess via PlannerClient, and plays back the sampled trajectory the
subprocess returns - without blocking, so the main loop keeps stepping every
robot and conveyor while a plan is being computed. It imports no cuMotion.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from enum import IntEnum

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics
from scipy.spatial.transform import Rotation

import isaacsim.core.experimental.utils.stage as stage_utils
import isaacsim.core.experimental.utils.xform as xform_utils
from isaacsim.core.experimental.objects import Cylinder
from isaacsim.core.experimental.prims import Articulation, GeomPrim, RigidPrim
from isaacsim.storage.native import get_assets_root_path

# Collision-aware motion planning (cuMotion) now runs OUT OF PROCESS in a
# dedicated headless planner subprocess (see planner_server.py /
# planner_server_impl.py) - a single cuMotion solve is a monolithic C++/CUDA
# call that holds the Python GIL for its whole duration, so running it in this
# process (even on a worker thread) froze the main sim loop until it returned.
# This module is now the PARENT side: it owns the pick/place state machine and
# plays back the trajectories the subprocess computes, and imports NO cuMotion
# at all. MagicAttachPickPlace ships each plan request to the subprocess via
# PlannerClient and polls for the result without blocking, so the main loop
# keeps stepping every robot and conveyor while a plan is being computed.

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

PLANNING_RETRIES = 3

# Ticks to pause before re-requesting a plan whose target isn't recoverable by
# nudging a conveyor (STAGE_FOR_PICK/STAGE_FOR_PLACE/DESCEND_TO_PLACE) - avoids
# hammering the planner subprocess every physics tick on a target that just
# keeps failing.
REPLAN_COOLDOWN_TICKS = 60


class PlanningFailedError(RuntimeError):
    """Raised by _drive_to once PLANNING_RETRIES plan requests all come back
    failed from the planner subprocess.

    Caught in forward() - a planning failure must never propagate out of
    forward() and crash the whole simulation app (see main()'s loop).
    """


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


class PlannerClient:
    """Parent-side handle to the out-of-process cuMotion planner.

    Wraps the full-duplex ``multiprocessing.connection`` to the planner
    subprocess (see planner_server.py). Shared by every MagicAttachPickPlace on
    this line - each tags its requests with a ``robot_id`` so results route
    back to the right one. The main loop calls ``pump()`` once per physics tick
    to drain whatever results have arrived; each robot then calls ``take()``.

    Tolerant of the subprocess dying: once the connection breaks, ``pump()``
    marks the client dead, ``submit()`` becomes a no-op, and ``take()`` returns
    None forever - so the arms simply hold their pose and the rest of the sim
    (conveyors, logging) keeps running, rather than crashing.
    """

    def __init__(self, conn) -> None:
        self._conn = conn
        self._next_id = 1
        self._results: dict[int, dict] = {}  # robot_id -> latest result message
        self._alive = True

    @property
    def alive(self) -> bool:
        return self._alive

    def submit(
        self,
        robot_id: int,
        mode: str,
        q_initial: np.ndarray,
        q_target: np.ndarray | None = None,
        position: np.ndarray | None = None,
        orientation: np.ndarray | None = None,
        phase_name: str = "",
    ) -> int | None:
        """Send a plan request; returns its request_id (None if the planner is dead)."""
        if not self._alive:
            return None
        request_id = self._next_id
        self._next_id += 1
        try:
            self._conn.send(
                {
                    "type": "plan",
                    "request_id": request_id,
                    "robot_id": robot_id,
                    "mode": mode,
                    "q_initial": np.asarray(q_initial, dtype=np.float64),
                    "q_target": None if q_target is None else np.asarray(q_target, dtype=np.float64),
                    "position": None if position is None else np.asarray(position, dtype=np.float64),
                    "orientation": None if orientation is None else np.asarray(orientation, dtype=np.float64),
                    "phase_name": phase_name,
                }
            )
        except (EOFError, OSError, BrokenPipeError) as exc:
            self._mark_dead(exc)
            return None
        return request_id

    def pump(self) -> None:
        """Drain all pending result messages from the subprocess (non-blocking)."""
        if not self._alive:
            return
        try:
            while self._conn.poll():
                msg = self._conn.recv()
                if msg.get("type") == "plan_result":
                    self._results[msg["robot_id"]] = msg
        except (EOFError, OSError) as exc:
            self._mark_dead(exc)

    def take(self, robot_id: int, request_id: int) -> dict | None:
        """Return the result for (robot_id, request_id) if it has arrived, else None."""
        msg = self._results.get(robot_id)
        if msg is not None and msg["request_id"] == request_id:
            del self._results[robot_id]
            return msg
        return None

    def _mark_dead(self, exc: Exception) -> None:
        if self._alive:
            print(
                f"[pick_and_place] WARNING: planner subprocess connection lost ({exc!r}); "
                "arms will hold, the rest of the sim keeps running",
                flush=True,
            )
        self._alive = False

    def close(self) -> None:
        """Tell the subprocess to stop and close the connection."""
        if self._alive:
            try:
                self._conn.send({"type": "stop"})
            except (EOFError, OSError):
                pass
        try:
            self._conn.close()
        except OSError:
            pass
        self._alive = False


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
        planner_client: PlannerClient,
        robot_id: int,
        default_joint_positions: list[float] = UR20_DEFAULT_JOINT_POSITIONS,
        pre_place_joint_positions: list[float] = UR20_PRE_PLACE_JOINT_POSITIONS,
        nudge_pick_zone_fn: Callable[[], None] | None = None,
    ) -> None:
        self.robot = robot
        self.place_xy = place_xy
        self.place_belt_top_z = place_belt_top_z
        self.box_rigid_prims = box_rigid_prims
        self._physics_dt = physics_dt
        # Shared handle to the planner subprocess + this robot's id within it
        # (the subprocess builds one planner per robot; see planner_server.py).
        self._planner = planner_client
        self._robot_id = robot_id
        # See UR20_PRE_PLACE_JOINT_POSITIONS_AWAY for when to override these.
        self._default_joint_positions = default_joint_positions
        self._pre_place_joint_positions = pre_place_joint_positions
        # Called when DESCEND_TO_PICK planning fails (see forward()) to pulse
        # the pick zone's belt a bit, so the box resettles at a slightly
        # different spot before the next attempt - owned by conveyor_indexer.py,
        # which has the actual ConveyorZone.
        self._nudge_pick_zone_fn = nudge_pick_zone_fn

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
        self._replan_cooldown = 0
        self._planning_attempt = 0
        # Plan request/playback state (see _drive_to). While a request is
        # outstanding the arm holds its pose; once the subprocess returns a
        # sampled trajectory it's played back one sample per physics tick.
        self._pending_request_id: int | None = None
        self._samples: np.ndarray | None = None  # (T, n_active_joints)
        self._sample_indices: np.ndarray | None = None  # dof indices for set_dof_position_targets
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

    def _drive_to(
        self,
        target_position: np.ndarray | None,
        phase_name: str,
        orientation: np.ndarray = DOWN_ORIENTATION,
        use_ik_cspace_target: bool = False,
        cspace_target: np.ndarray | list[float] | None = None,
    ) -> bool:
        """Request a collision-free plan from the planner subprocess at phase
        entry, then play back the sampled trajectory it returns open-loop;
        return True once this phase should end.

        Non-blocking: the plan request is shipped to the subprocess and this
        returns False (arm holds its pose) every tick until the result arrives,
        so the main loop keeps stepping the other robot and the conveyors while
        cuMotion computes. Raises PlanningFailedError after PLANNING_RETRIES
        failed results (caught in forward()).

        Exactly one target flavor applies, in this order of precedence:
          - cspace_target: plan straight to a known joint configuration (e.g.
            UR20_DEFAULT_JOINT_POSITIONS / UR20_PRE_PLACE_JOINT_POSITIONS) -
            target_position/orientation are ignored.
          - use_ik_cspace_target: the subprocess solves IK to one concrete
            joint target first, then plans to it (avoids a contorted
            task-space JtRRT configuration).
          - otherwise: plan straight to the task-space pose
            (target_position, orientation).
        """
        if self._samples is None:
            # Planning phase - no trajectory yet. Submit once, then poll.
            if self._pending_request_id is None:
                q_initial = self.robot.get_dof_positions().numpy()[0].astype(np.float64)
                if cspace_target is not None:
                    mode, q_target = "cspace", np.asarray(cspace_target, dtype=np.float64)
                    self._pending_request_id = self._planner.submit(
                        self._robot_id, mode, q_initial, q_target=q_target, phase_name=phase_name
                    )
                else:
                    mode = "ik_cspace" if use_ik_cspace_target else "pose"
                    self._pending_request_id = self._planner.submit(
                        self._robot_id, mode, q_initial,
                        position=target_position, orientation=orientation, phase_name=phase_name,
                    )
                if self._pending_request_id is None:
                    # Planner subprocess is gone (see PlannerClient) - hold
                    # pose indefinitely; the rest of the sim keeps running.
                    return False

            resp = self._planner.take(self._robot_id, self._pending_request_id)
            if resp is None:
                # Still computing - hold pose, let the rest of the sim advance.
                return False
            self._pending_request_id = None

            if not resp["ok"]:
                self._planning_attempt += 1
                if self._planning_attempt >= PLANNING_RETRIES:
                    self._planning_attempt = 0
                    raise PlanningFailedError(
                        f"planner subprocess found no collision-free path entering {phase_name} "
                        f"after {PLANNING_RETRIES} attempts (last error: {resp.get('error')})"
                    )
                print(
                    f"[pick_and_place] WARNING: plan request {self._planning_attempt}/{PLANNING_RETRIES} "
                    f"failed entering {phase_name} ({resp.get('error')}), re-requesting next tick",
                    flush=True,
                )
                # Not finished; re-submit next tick, letting the sim step between.
                return False

            self._planning_attempt = 0
            self._samples = resp["positions"]
            # The subprocess returns positions in `joint_names` order; map those
            # names to this articulation's dof indices once (cached), so the
            # ordering never has to match implicitly.
            if self._sample_indices is None:
                dof_names = list(self.robot.dof_names)
                self._sample_indices = np.array(
                    [dof_names.index(name) for name in resp["joint_names"]], dtype=np.int64
                )
            self._step = 0

        # Playback: one sampled joint target per physics tick.
        if self._step < len(self._samples):
            self.robot.set_dof_position_targets(
                positions=self._samples[self._step], dof_indices=self._sample_indices
            )

        self._step += 1
        finished = self._step >= len(self._samples)
        timed_out = self._step >= PHASE_TICKS[phase_name]
        if timed_out and not finished:
            print(
                f"[pick_and_place] {phase_name} exceeded its {PHASE_TICKS[phase_name]}-tick budget "
                f"before the sampled trajectory ({len(self._samples)} samples) finished",
                flush=True,
            )
        if finished or timed_out:
            self._step = 0
            self._samples = None  # _sample_indices is constant; keep it cached
            return True
        return False

    def forward(self, pick_ready: bool, box_path: str | None) -> None:
        # No per-tick box-following code needed here anymore - once
        # _attach_box() creates the FixedJoint, PhysX itself keeps the box
        # rigidly following wrist_3_link (position and orientation) for the
        # whole STAGE_FOR_PICK(carrying)/STAGE_FOR_PLACE/DESCEND_TO_PLACE carry.

        if self._replan_cooldown > 0:
            # Pausing between a plan_to_cspace_target/plan_to_pose_target
            # failure (see PlanningFailedError handlers below) and the next
            # attempt at the SAME target - otherwise a persistently
            # unreachable target would hammer the planner every physics tick.
            self._replan_cooldown -= 1
            return

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
            try:
                finished = self._drive_to(None, "STAGE_FOR_PICK", cspace_target=self._default_joint_positions)
            except PlanningFailedError as exc:
                print(f"[pick_and_place] WARNING: {exc}; pausing then retrying", flush=True)
                self._replan_cooldown = REPLAN_COOLDOWN_TICKS
                return
            if finished:
                self._phase = Phase.STAGE_FOR_PLACE if self._holding_box else Phase.DESCEND_TO_PICK

        elif self._phase == Phase.DESCEND_TO_PICK:
            try:
                finished = self._drive_to(self._pick_point, "DESCEND_TO_PICK")
            except PlanningFailedError as exc:
                # Recoverable, unlike the other phases' targets: the pick
                # point is just wherever the box happened to settle, so
                # nudging the pick-zone belt a bit and re-detecting the box
                # (reusing WAITING's own settle-check/pick-point logic below)
                # gives the next attempt a genuinely different, hopefully
                # reachable target - instead of retrying the same failing one.
                print(
                    f"[pick_and_place] WARNING: {exc}; nudging pick zone conveyor and retrying with box's new position",
                    flush=True,
                )
                self.box.set_enabled_rigid_bodies([True])
                if self._nudge_pick_zone_fn is not None:
                    self._nudge_pick_zone_fn()
                self._phase = Phase.WAITING
                self._step = 0
                return
            if finished:
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
            try:
                finished = self._drive_to(None, "STAGE_FOR_PLACE", cspace_target=self._pre_place_joint_positions)
            except PlanningFailedError as exc:
                print(f"[pick_and_place] WARNING: {exc}; pausing then retrying", flush=True)
                self._replan_cooldown = REPLAN_COOLDOWN_TICKS
                return
            if finished:
                self._phase = Phase.DESCEND_TO_PLACE if self._holding_box else Phase.WAITING

        elif self._phase == Phase.DESCEND_TO_PLACE:
            try:
                finished = self._drive_to(
                    self.place_position, "DESCEND_TO_PLACE", orientation=PLACE_ORIENTATION, use_ik_cspace_target=True
                )
            except PlanningFailedError as exc:
                print(f"[pick_and_place] WARNING: {exc}; pausing then retrying", flush=True)
                self._replan_cooldown = REPLAN_COOLDOWN_TICKS
                return
            if finished:
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
