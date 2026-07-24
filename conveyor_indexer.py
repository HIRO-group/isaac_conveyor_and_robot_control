"""Standalone Isaac Sim script: zone-accumulation indexing over the inbuilt
surface-velocity conveyors in ~/5_conv_env.usd - two short, OPEN (non-looping)
conveyor lines rather than racetrack.usd's closed ovals - with per-tick data
logging for later imitation/RL training, plus a UR20 pick-and-place (driven
by NVIDIA cuMotion RMPflow) that moves boxes from loop 1 to loop 2, which
then carries them off its far end into a waiting SteelBoxTruck.

Run with Isaac Sim's bundled python, e.g.:
    ./python.sh ~/conveyor_indexing/conveyor_indexer.py

See README.md in this directory for setup (protobuf codegen), and everything
this scaffold does NOT implement yet.
"""

from __future__ import annotations

import ctypes
import math
import os
import pathlib
import signal
import subprocess
import sys
import tempfile
from multiprocessing.connection import Listener

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import carb
import omni.kit.app
import omni.usd
from omni.physics.core import get_physics_scene_query_interface
from pxr import Gf, PhysicsSchemaTools, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics
from isaacsim.core.api import World
from isaacsim.core.experimental.prims import RigidPrim

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_DIR)
sys.path.insert(0, "/tmp/proto_gen")  # see README.md: gen_proto.sh output

import plc_connector_pb2 as plc
import sim_conveyor_action_pb2 as sim_action

try:
    from common import types_pb2 as common_types
except ModuleNotFoundError:
    import types_pb2 as common_types

from conveyor_state_machine import ConveyorZoneStateMachine
from conveyor_indexing_logger import ConveyorIndexingLogger
from pick_and_place import (
    create_pedestal_and_robot,
    MagicAttachPickPlace,
    PlannerClient,
    UR20_PRE_PLACE_JOINT_POSITIONS_AWAY,
)
from scene_setup import apply_truck_collision, deactivate_frame_meshes, localize_asset_references

STAGE_PATH = os.path.join(os.path.expanduser("~"), "5_conv_env.usd")
LOG_OUTPUT_DIR = os.path.join(REPO_DIR, "data")
# Matches the 120Hz physics rate - occupancy detection and hold-zone stop
# commands used to only run at 30Hz, letting a box drift up to 1/30s past
# where it should've been caught before the belt reacted.
CONTROL_HZ = 120.0
DEBUG_LOG_OCCUPANCY_HITS = False  # set True to print which prim triggers each occupancy hit
DEBUG_LOG_HOLD_ZONE_STATE = False  # set True to print every hold zone's state machine each control tick

# 5_conv_env.usd references 5 distinct assets (the conveyor belt, the truck,
# and 3 box variants - each referenced multiple times) from the public
# Omniverse S3 content bucket over HTTPS; opening the stage re-fetches them
# over the network every run unless localized. `download_assets.py` (see
# README) mirrors the full recursive dependency tree of everything under
# REMOTE_ASSET_ROOT into LOCAL_ASSET_ROOT, preserving the bucket's own
# relative directory structure - confirmed (by mirroring, then re-resolving
# every reference in the scene against the mirror) to be a clean string-
# prefix swap, no other path rewriting needed. See
# _localize_asset_references, which performs that swap at runtime.
REMOTE_ASSET_ROOT = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/"
LOCAL_ASSET_ROOT = os.path.join(os.path.expanduser("~"), "isaac_assets")

# ---------------------------------------------------------------------------
# Zone tables - two independent OPEN (non-looping) lines, confirmed via
# direct world-bbox inspection of 5_conv_env.usd: loop 1 (ConveyorTrack/_01/_02)
# runs along Y=0, loop 2 (ConveyorTrack_09/_10) runs along Y~2.186, both
# spanning the same X range and both straight-only (no curved zones, unlike
# racetrack.usd's ovals). Loop 2's far end (ConveyorTrack_10's -X edge) sits
# right at the near wall of `/World/SteelBoxTruck_A01_01`, whose bed sits
# ~0.83 m below belt-top height - boxes that reach the end of loop 2 are
# meant to run off the belt and drop into the truck bed, not hand off to
# another zone.
# ---------------------------------------------------------------------------
ZONE_NODE_PATHS_LOOP1 = [
    "/World/ConveyorTrack/ConveyorBeltGraph/ConveyorNode",
    "/World/ConveyorTrack_01/ConveyorBeltGraph/ConveyorNode",
    "/World/ConveyorTrack_02/ConveyorBeltGraph/ConveyorNode",
]
ZONE_NODE_PATHS_LOOP2 = [
    "/World/ConveyorTrack_09/ConveyorBeltGraph/ConveyorNode",
    "/World/ConveyorTrack_10/ConveyorBeltGraph/ConveyorNode",
]

# ConveyorTrack_01 (loop 1) and ConveyorTrack_09 (loop 2) directly face each
# other across the gap (both centered at local X=-3) - the natural spot for a
# fixed pick/place robot between the two loops. ConveyorTrack_02, the third
# loop-1 zone, sits at X=-5 - out of a UR20's reach from that same robot
# position, so it's left as an (unused) upstream buffer; every box starts the
# run already stacked on ConveyorTrack (zone 0), one line-length from the
# pick zone.
PICK_ZONE_INDEX = 1  # ConveyorTrack_01 within ZONE_NODE_PATHS_LOOP1
PLACE_ZONE_INDEX = 0  # ConveyorTrack_09 within ZONE_NODE_PATHS_LOOP2

# Second pick/place station, one zone further downstream on each loop - see
# ConveyorLineController.set_hold_zone_ready_check for overflow between them.
PICK_ZONE_INDEX_2 = 2  # ConveyorTrack_02 within ZONE_NODE_PATHS_LOOP1
PLACE_ZONE_INDEX_2 = 1  # ConveyorTrack_10 within ZONE_NODE_PATHS_LOOP2

ROBOT_PATH = "/World/PickPlaceRobot"
PEDESTAL_PATH = "/World/PickPlacePedestal"
ROBOT_PATH_2 = "/World/PickPlaceRobot_02"
PEDESTAL_PATH_2 = "/World/PickPlacePedestal_02"
TRUCK_PATH = "/World/SteelBoxTruck_A01_01"

# 5_conv_env.usd's `/World/GroundPlane/CollisionPlane` is a genuine
# UsdGeom.Plane-typed prim with PhysicsCollisionAPI (unlike racetrack.usd's
# ground, which apparently has no such prim - this scaffold never hit this
# path against that scene) - `_build_motion_planner`'s obstacle scan picks it
# up and classifies it as an `ObstacleRepresentation.PLANE` obstacle via
# `get_shape_type()`'s `Plane.are_of_type()` check, which routes into
# `WorldBinding._add_plane_from_prim` -> `CumotionWorldInterface.add_planes`.
# That code path is broken independent of anything in this scaffold -
# confirmed via direct traceback: `add_planes` hits pick_and_place.py's own
# `np.reshape` compat shim (see its module docstring - a real, separate
# NumPy 1.26/2.1 API mismatch), and that shim recurses into itself
# (`RecursionError: maximum recursion depth exceeded`) rather than calling
# through to the real `np.reshape` - a second, independent bug in the shim
# itself (not yet root-caused), not something introduced here. The ground
# plane doesn't need cuMotion obstacle avoidance anyway (the arm, mounted on
# a 1.6 m pedestal, never reaches down into it), so it's excluded from
# obstacle tracking via each robot's `extra_exclude` in the planner init
# config (see main()) rather than chasing the recursion bug itself.
GROUND_PLANE_COLLISION_PATH = "/World/GroundPlane/CollisionPlane"

# 5_conv_env.usd already ships its own pallet of ~18 CubeBox_* prims
# pre-authored directly on top of ConveyorTrack's belt (confirmed via
# world-bbox inspection: every one sits within ConveyorTrack's belt-top XY
# footprint, stacked in two Z layers) - unlike racetrack.usd, which ships
# with no boxes at all and needed them referenced in at runtime. So this
# scaffold discovers whatever CubeBox_* prims are actually present
# (_discover_box_prim_paths) rather than spawning a fixed count itself.
#
# Those prims are pure visual payloads with no physics schemas at all
# (confirmed: HasAPI(UsdPhysics.RigidBodyAPI) / CollisionAPI both False on
# every one) - unlike racetrack.usd's sm_box_multiDepth_brown_b08_01
# references, which carry physics baked into the referenced asset itself.
# _apply_box_physics() adds RigidBodyAPI + convex-hull CollisionAPI + a mass
# at runtime, the same "don't edit the source USD" convention this scaffold
# already uses for _deactivate_frame_meshes etc.
BOX_PRIM_NAME_PREFIX = "CubeBox_"

# Approximate cardboard-box-with-contents density (kg/m^3), used to derive
# each box's mass from its own bbox volume in _apply_box_physics. Real
# corrugated cardboard alone is far lighter than this - this is a plausible
# ballpark for a lightly-packed shipping box, not a measured value, since
# 5_conv_env.usd doesn't author a mass for these at all.
BOX_DENSITY_KG_PER_M3 = 150.0

# The pallet's two Z layers are spaced ~0.29 m apart (confirmed via bbox
# inspection), but the tallest box variant (CubeBox_A06, 42 cm) doesn't fit
# in that clearance - so at least some boxes start out physically
# interpenetrating their neighbors the instant RigidBodyAPI is applied.
# Without a cap, PhysX's depenetration resolves that overlap as a large
# one-tick separating impulse - confirmed empirically: every box ended up
# flung clear of the belt entirely (bizarre positions like world X > 0, well
# past the conveyor's own -6..0 footprint, motionless on the bare ground
# for the rest of the run) rather than just jostling apart. Capping each
# box's own `physxRigidBody:maxDepenetrationVelocity` bounds how fast PhysX
# is allowed to push overlapping bodies apart per step, turning that into a
# gradual, physically plausible separation over the first second or so
# instead of an explosion - standard PhysX practice for bodies that may
# start out overlapping, not a 5_conv_env.usd-specific hack.
BOX_MAX_DEPENETRATION_VELOCITY = 0.5  # m/s

# Value of each track's `graph:variable:Velocity` (see ConveyorZone.apply_command)
# when a zone is commanded to run at 100% speed. Matches racetrack.usd's
# value (confirmed empirically there: a box transitions into/through a
# curved track without flying off at this speed). Each loop here instead
# dials its own actual run speed down from this via LOOP1_RUN_SPEED_PCT /
# LOOP2_RUN_SPEED_PCT - see those constants.
ZONE_RUN_VELOCITY = 2.0

# Loop 1 (the pick side) at full speed conveys boxes into the pick zone
# faster than really needed for a comfortable watch/pick cadence - slowed
# down per explicit direction after watching the full scaffold run live.
# NOTE: an earlier attempt at a bigger global slowdown (effectively 1.0 m/s
# on both loops, via ZONE_RUN_VELOCITY itself) caused the arm's ATTACH phase
# (which has no timeout, by design - see pick_and_place.py) to get stuck
# indefinitely, never quite converging within ATTACH_MAX_DISTANCE of a
# settling box. Not fully root-caused; if this speed reproduces that, the
# real fix is likely a timeout/fallback for ATTACH itself rather than
# avoiding slower belt speeds.
LOOP1_RUN_SPEED_PCT = 55  # was 60; nudged down for more stopping margin at ConveyorTrack_02

# Loop 2's own goal differs from loop 1's: a box should run off
# ConveyorTrack_10's end and land INSIDE the waiting truck's bed, not
# overshoot it. Confirmed empirically at full speed (ZONE_RUN_VELOCITY,
# 2.0 m/s): boxes cleared the truck's far wall entirely and landed on the
# ground beyond it (a box leaving the belt at speed v, falling from belt-top
# to the bed floor, travels roughly v * fall_time horizontally - at 2.0 m/s
# that distance exceeded the truck's own ~0.77 m depth). This speed is
# confirmed working (landed cleanly in the truck) - left as-is per explicit
# direction ("the speed going into the truck is fine").
LOOP2_RUN_SPEED_PCT = 50

# ConveyorZone.nudge() parameters for a pick zone that failed to plan a
# descend (see MagicAttachPickPlace's nudge_pick_zone_fn) - short and slow
# enough to shift the box a few cm rather than send it overflowing off the
# zone entirely.
PICK_REPLAN_NUDGE_TICKS = 12
PICK_REPLAN_NUDGE_SPEED_PCT = 20

# Both loops in 5_conv_env.usd are already authored close enough for a UR20
# (1.75 m spec reach - see robot_configs/ur20/, generated via cuMotion) to
# bridge without any runtime repositioning: ConveyorTrack_01 (loop 1 pick
# zone) and ConveyorTrack_09 (loop 2 place zone) are both centered at local
# X=-3, with loop 1's belt-near-edge at Y=0.45 and loop 2's at Y=1.736 - a
# robot at the Y midpoint reaches each at ~1.09 m, well inside spec (this
# mirrors the reach-balancing racetrack.usd needed LOOP2_Y_SHIFT for with a
# UR10, but here the un-shifted geometry already fits a UR20 comfortably).
ROBOT_POSITION = (-3.0, 1.0928, 0.0)  # (x, y, z-of-ground-contact); Y = loop midpoint
PEDESTAL_HEIGHT = 1.6
PEDESTAL_RADIUS = 0.15  # matches create_pedestal_and_robot's default
PLACE_XY = (-3.0, 2.1857)  # ConveyorTrack_09's belt-top Y center

# Sent to the planner subprocess as this robot's cuMotion config (see
# planner_server_impl.py). Per-joint limits for time-parameterizing planned
# paths; the XRDF/URDF live in robot_configs/ur20 (loaded there, not here).
UR20_XRDF_PATH = os.path.join(REPO_DIR, "robot_configs", "ur20", "robot.xrdf")
UR20_URDF_PATH = os.path.join(REPO_DIR, "robot_configs", "ur20", "robot.urdf")
UR20_TOOL_FRAME = "tool0"
MAX_JOINT_VELOCITIES = [2.0, 2.0, 2.0, 2.0, 2.0, 2.0]
MAX_JOINT_ACCELERATIONS = [10.0, 10.0, 10.0, 10.0, 10.0, 10.0]

CONVEYOR_TRACK_ROOTS = ("/World/ConveyorTrack", "/World/ConveyorTrack_01", "/World/ConveyorTrack_02", "/World/ConveyorTrack_09", "/World/ConveyorTrack_10")

# Any occupancy hit whose prim path falls under one of these roots is belt/
# structure/robot/truck geometry, not a transported item, and is excluded
# from occupancy detection.
EXCLUDED_STRUCTURE_ROOTS = CONVEYOR_TRACK_ROOTS + (
    ROBOT_PATH,
    PEDESTAL_PATH,
    ROBOT_PATH_2,
    PEDESTAL_PATH_2,
    TRUCK_PATH,
)


class ConveyorZone:
    """Bridges one ConveyorZoneStateMachine to its USD ConveyorNode + belt bbox."""

    def __init__(self, index: int, node_path: str, stage: Usd.Stage, run_speed_pct: int = 100) -> None:
        self.index = index
        self.node_path = node_path
        self.node_prim = stage.GetPrimAtPath(node_path)
        if not self.node_prim.IsValid():
            raise RuntimeError(f"ConveyorNode prim not found at {node_path}")

        rel = self.node_prim.GetRelationship("inputs:conveyorPrim")
        targets = rel.GetTargets() if rel else []
        if not targets:
            raise RuntimeError(f"{node_path} has no inputs:conveyorPrim target")
        self.belt_prim = stage.GetPrimAtPath(targets[0])

        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
        world_bound = bbox_cache.ComputeWorldBound(self.belt_prim)
        aligned_range = world_bound.ComputeAlignedRange()
        size = aligned_range.GetSize()
        # `self.belt_prim` ("Belt") is just the thin moving-surface mesh, not
        # the full occupying box's volume - using its own (near-zero) Z
        # extent for the occupancy query missed boxes resting on top by even
        # a hair's-width contact gap. Query a generous column above the belt
        # top instead, tall enough for any box variant in this scene.
        belt_top_z = aligned_range.GetMax()[2]
        occupancy_query_half_height = 0.5  # meters; tallest known box is ~0.42m
        self.bbox_half_extent = [size[0] * 0.5, size[1] * 0.5, occupancy_query_half_height]
        self.bbox_center = [aligned_range.GetMidpoint()[0], aligned_range.GetMidpoint()[1], belt_top_z + occupancy_query_half_height]
        # Belts are treated as axis-aligned and static; identity rotation.
        self._quat = carb.Float4(0.0, 0.0, 0.0, 1.0)

        # `inputs:velocity` on every ConveyorNode is wired (via a ReadVariable
        # node) to this per-track OmniGraph variable rather than holding a
        # plain value - confirmed by inspecting the ConveyorBeltGraph's
        # `inputs:velocity` connections. It is unauthored on every track in
        # racetrack.usd (no WriteVariable node anywhere sets it either), so
        # OgnIsaacConveyor's `targetVelocity = direction * velocity` was
        # always (0,0,0) regardless of `enabled` - confirmed by enabling a
        # belt and reading back its actual PhysxSurfaceVelocityAPI value
        # after stepping physics. `apply_command` must author this directly.
        self.velocity_var_attr = self.node_prim.GetParent().GetAttribute("graph:variable:Velocity")
        if not self.velocity_var_attr:
            raise RuntimeError(f"{node_path}'s graph has no graph:variable:Velocity")

        # `inputs:direction` (unlike Velocity) IS authored per-track in
        # racetrack.usd. Straight zones store a flat (z=0) unit translation
        # vector, and direct stage inspection found ConveyorTrack/_01/_02 (one
        # straight leg of the oval) baked with the SAME (1,0,0) as
        # ConveyorTrack_04/_05/_06 (the return leg), despite ConveyorTrack/_01/_02's
        # bodies being rotated 180 deg about Z in world space relative to
        # _04/_05/_06's identity rotation. An earlier pass here assumed this meant
        # `inputs:direction` needed a compensating negation for the flipped row
        # (`physxSurfaceVelocity:surfaceVelocityLocalSpace` defaults True, so the
        # assumption was that PhysX consumes this vector in the belt's own body
        # frame) - visually confirmed WRONG by running the full scaffold: with
        # that negation applied, the flipped row ran clockwise (backwards) while
        # the unflipped row and both curves ran counterclockwise correctly. PhysX
        # does not need this vector pre-compensated for the belt's own world
        # rotation; the same raw (1,0,0) on both rows, authored as-is, is what's
        # actually correct here (see ConveyorLineController._fix_zone_directions).
        # Curved zones store a
        # nonzero-z angular-velocity axis instead - same magnitude (37) on all 4
        # curves in both loops, self-consistent but never actually validated
        # against this scaffold's own ZONE_RUN_VELOCITY scaling. Confirmed via a
        # standalone headless test (enable one curve, drop a box, read back its
        # world position every physics tick): 37 combined with apply_command's
        # velocity scaling authors a ~74 rad/s angular velocity - for this curve's
        # ~1.5 m turn radius that launches an occupying box clean off the belt on
        # first contact instead of conveying it (the box's world position jumped
        # >2 m within half a second). The correct magnitude for this radius at
        # ZONE_RUN_VELOCITY is closer to 1.3 rad/s.
        # See ConveyorLineController._fix_zone_directions, which recomputes
        # straight-zone direction from actual belt geometry once every zone's
        # bbox_center is known (confirmed to reconstruct these same baked values
        # given the rotations above) and, for curved zones, rederives both the
        # angular-velocity magnitude AND sign from the curve's actual geometry -
        # the sign can't just be copied from the baked value either, despite both
        # curves in a loop baking the identical -37: see that method's own
        # docstring for why (the two curves are mirror-image ends of the same
        # racetrack, so the same absolute spin sense is forward at one end and
        # backward at the other) - kept as the more robust geometry-derived
        # source rather than relying on the authored data staying
        # correct/well-scaled if the scene layout ever changes.
        self.direction_attr = self.node_prim.GetAttribute("inputs:direction")
        baked_direction = self.direction_attr.Get()
        self.is_straight = baked_direction is not None and baked_direction[2] == 0.0

        # Set by ConveyorLineController._fix_zone_directions for straight
        # zones: the WORLD-space unit direction of travel, same vector as
        # direction_attr for straight zones (see that method's docstring -
        # no body-frame compensation is applied). Kept as its own field
        # rather than having is_past_center read direction_attr directly
        # since is_past_center is only ever meaningful for straight zones,
        # while direction_attr is also set (to a different kind of vector -
        # an angular-velocity axis) for curved ones.
        self.world_travel_direction: Gf.Vec3f | None = None

        self.state_machine = ConveyorZoneStateMachine(name=node_path, run_speed_pct=run_speed_pct)
        self.state_machine.start()

        # Set by nudge(); consumed in ConveyorLineController.step(), which
        # overrides the state machine's own command (a held pick zone
        # otherwise commands run=False every tick) for this many remaining
        # ticks - see nudge()'s own docstring.
        self._nudge_ticks_remaining = 0
        self._nudge_speed_pct = 0

    def nudge(self, ticks: int, speed_pct: int) -> None:
        """Briefly force this zone's belt to run, overriding its own hold
        state machine, so an occupying part shifts a bit rather than staying
        pinned exactly where it first settled.

        Used when pick_and_place.py's motion planner fails to reach a box at
        its current settled position - see MagicAttachPickPlace's
        nudge_pick_zone_fn - since retrying the identical unreachable target
        would just fail again.
        """
        self._nudge_ticks_remaining = ticks
        self._nudge_speed_pct = speed_pct

    def get_occupying_prim_paths(self) -> list:
        """Return the paths of every non-excluded rigid body overlapping this zone.

        Used both for the boolean occupied check and, for the pick zone
        specifically, to identify WHICH box is actually present - there are
        ~18 physics-enabled boxes on the line (see _discover_box_prim_paths),
        and picking must track whichever one actually triggered `pick_ready`,
        not a hardcoded path (a fixed path silently tracks the wrong box, and
        the wrong box, whenever it happens to be, the moment it's "detached"
        ends up wherever the arm's current position, unrelated to the belt).
        """
        hits = []

        def report_hit(hit) -> bool:
            path = str(PhysicsSchemaTools.intToSdfPath(hit.rigid_body))
            if not path.startswith(EXCLUDED_STRUCTURE_ROOTS):
                hits.append(path)
                if DEBUG_LOG_OCCUPANCY_HITS:
                    print(f"[conveyor_indexer] DEBUG occupancy hit: zone={self.node_path} hit_path={path}", flush=True)
            return True

        get_physics_scene_query_interface().overlap_box(
            carb.Float3(*self.bbox_half_extent),
            carb.Float3(*self.bbox_center),
            self._quat,
            report_hit,
        )
        return hits

    def check_occupied(self) -> bool:
        return len(self.get_occupying_prim_paths()) > 0

    def is_past_center(self, world_position, stop_fraction: float = 0.5) -> bool:
        """True once world_position has reached/passed the stop point that's
        `stop_fraction` of the way through this straight zone along its
        direction of travel (0.5 = geometric midpoint).

        Used by the hold zone(s) to settle an occupying part at a fixed,
        robot-reachable point instead of wherever it first entered the
        zone's occupancy sensor. Only meaningful for straight zones - the
        one caller (ConveyorLineController.step, for zones in
        hold_zone_indices) never uses this on a curved zone.

        Uses world_travel_direction (set by
        ConveyorLineController._fix_zone_directions), NOT
        direction_attr - the latter is a LOCAL-frame vector once corrected
        for PhysX, and using its sign directly against world_position would
        be backwards on any zone whose body isn't identity-rotated in world
        space (see world_travel_direction's own comment in __init__).
        """
        travel = self.world_travel_direction
        assert travel is not None, f"{self.node_path}: is_past_center called before world_travel_direction was set"
        axis = 0 if abs(travel[0]) >= abs(travel[1]) else 1
        sign = 1.0 if travel[axis] > 0 else -1.0
        stop_point = self.bbox_center[axis] + sign * self.bbox_half_extent[axis] * (2 * stop_fraction - 1)
        return (world_position[axis] - stop_point) * sign >= 0.0

    def apply_command(self, run: bool, speed_pct: int) -> None:
        self.node_prim.GetAttribute("inputs:enabled").Set(run)
        if run:
            # direction is baked in per-track (unit vector for straight
            # belts, an angular-velocity axis scaled for the local curve
            # radius for curved ones) so the same Velocity value works for
            # both; only its magnitude is set here.
            self.velocity_var_attr.Set(ZONE_RUN_VELOCITY * speed_pct / 100.0)
        else:
            # `inputs:enabled=False` alone does NOT stop the belt: OgnIsaacConveyor's
            # compute() early-returns on disabled without ever touching the belt's
            # PhysxSurfaceVelocityAPI attributes, so the last nonzero surface
            # velocity stays authored and PhysX keeps driving the belt regardless of
            # `enabled` - confirmed by reading OgnIsaacConveyor.cpp. Zero it directly
            # here; re-enabling needs no corresponding restore, since the node's own
            # hasVelocityChanged check (currentVelocity=0 vs its target) will rewrite
            # the correct nonzero velocity on the next tick where enabled=True again.
            surface_velocity_api = PhysxSchema.PhysxSurfaceVelocityAPI.Apply(self.belt_prim)
            zero = Gf.Vec3f(0.0, 0.0, 0.0)
            surface_velocity_api.GetSurfaceVelocityAttr().Set(zero)
            surface_velocity_api.GetSurfaceAngularVelocityAttr().Set(zero)


class ConveyorLineController:
    """Owns one line's ordered zones, wires neighbor occupancy, applies commands.

    Supports both a closed loop (racetrack.usd's ovals, wrapping neighbor
    index i-1/i+1 around modulo the zone count) and an open line
    (5_conv_env.usd's two short straight runs, where the first zone has no
    real upstream and the last has no real downstream) via `closed_loop`.
    """

    def __init__(
        self,
        stage: Usd.Stage,
        node_paths: list,
        hold_zone_indices: frozenset = frozenset(),
        closed_loop: bool = False,
        run_speed_pct: int = 100,
    ) -> None:
        self.zones = [ConveyorZone(i, path, stage, run_speed_pct=run_speed_pct) for i, path in enumerate(node_paths)]
        self.hold_zone_indices = hold_zone_indices
        self.closed_loop = closed_loop
        self.occupied: list = [False] * len(self.zones)
        self.machine_states: list = [None] * len(self.zones)
        self._box_rigid_prims: dict | None = None
        # Per-hold-zone robot-readiness callback - see set_hold_zone_ready_check.
        # Absent means always-ready (unconditional hold, the original behavior).
        self._hold_zone_ready_checks: dict = {}
        self._fix_zone_directions()

    def set_hold_zone_ready_check(self, zone_index: int, ready_fn) -> None:
        """While `ready_fn()` (zero-arg, returns bool) is True, hold zone
        `zone_index` holds its occupant as before. While False (that robot
        is busy), it behaves like an ordinary pass-through zone instead, so
        a box overflows to the next pick station rather than backing up
        behind a busy robot - needed once two hold zones share one line.
        """
        self._hold_zone_ready_checks[zone_index] = ready_fn

    def set_box_rigid_prims(self, box_rigid_prims: dict) -> None:
        """Give this controller the live RigidPrim for every known box, so
        hold_zone_indices zones can check is_past_center() against the
        occupying box's actual position. Built in main() only after
        world.reset() (see main()'s ordering comments), so it's injected
        here rather than passed to __init__.
        """
        self._box_rigid_prims = box_rigid_prims

    @staticmethod
    def _is_body_flipped(belt_prim) -> bool:
        """True if belt_prim's world rotation is ~180deg about some axis.

        `physxSurfaceVelocity:surfaceVelocityLocalSpace` defaults True, so
        `inputs:direction` is consumed in the belt's own BODY frame, not world
        space - this tells callers whether that body frame is flipped relative
        to world space and needs compensating for.
        """
        world_rotation = UsdGeom.Xformable(belt_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractRotation()
        return abs(world_rotation.GetQuat().GetReal()) < 0.5

    def _fix_zone_directions(self) -> None:
        """Overwrite every zone's inputs:direction from actual belt geometry.

        Needed because racetrack.usd's authored directions aren't reliable (see
        ConveyorZone.__init__).

        Straight zones: derives the correct unit vector from this zone's
        bbox_center to the next zone's (i -> i+1, wrapping around the closed loop),
        snapped to whichever axis the belt is actually elongated along, since a
        straight belt only ever needs to move along its own long axis and blending
        in the other axis (e.g. from a curved neighbor's offset bbox_center) would
        put a wrong lateral component into an otherwise axis-aligned belt. This
        world-space vector is authored directly as `inputs:direction`, with NO
        per-body negation for flipped (180deg-about-Z) rows despite
        `physxSurfaceVelocity:surfaceVelocityLocalSpace` defaulting True - an
        earlier version of this method assumed that flag meant `inputs:direction`
        needed compensating for each belt's own world rotation (negating for
        ConveyorTrack/_01/_02, loop 1's flipped near row), which happened to
        reconstruct the same values already baked in racetrack.usd and so looked
        confirmed at the time. Visually running the full scaffold (pick-and-place
        enabled, 5 boxes) proved that assumption backwards: the flipped near row
        (both loops' ConveyorTrack/_01/_02 and _08/_09/_10) ran CLOCKWISE - the
        wrong way - while the unflipped far row and both (correctly-signed, see
        below) curves ran counterclockwise correctly. PhysX evidently does not
        need `inputs:direction` pre-compensated for the belt's own world rotation
        here; authoring the plain world-space vector on every straight zone,
        flipped row or not, is what actually matches the rest of the loop.

        Curved zones: rederives the angular-velocity magnitude from the curve's
        own radius - half the distance between its two straight neighbors' bbox
        centers - so the resulting tangential speed (omega * radius) matches
        ZONE_RUN_VELOCITY at 100%, same as the straight zones' linear speed. See
        ConveyorZone.direction_attr's own comment for why this matters: the baked
        magnitude (37, uncorrected) is roughly 55x too large for this scaffold's
        velocity scaling and launches an occupying box off the belt outright.

        The sign, by contrast, can NOT just be copied from the baked value even
        though both curves in a loop bake the identical (0, 0, -37): the two
        curves sit at mirror-image ends of the same racetrack (confirmed via
        bbox inspection - e.g. loop 1's _03 at x=-6.97 vs _07 at x=+0.97,
        straddling the two straight rows), so the SAME absolute spin sense is
        forward at one end and backward at the other, the same way a real
        racetrack's two turntable-style end caps would need to counter-rotate to
        send an item the same way around the whole loop. Confirmed via a full
        run of this scaffold (pick-and-place enabled, all 5 boxes, PICK_ZONE
        occupancy logged every tick): with the sign simply copied from the baked
        value, the box placed on loop 2 traveled correctly through zone9 into
        zone8, then stalled dead at the zone8/_15 boundary for 20+ seconds
        instead of continuing into the curve, while a loop-1 box run the same way
        traveled BACKWARDS through _07 (from zone0 toward zone6 - the wrong way
        for the i -> i+1 model every zone's occupancy/handoff logic assumes) even
        though _03 conveyed correctly with that same copied sign. The reliable
        per-curve discriminator, confirmed against both loops: whether the
        curve's OWN entry-side neighbor (i-1) is itself flipped - _03's entry
        neighbor (zone2) is flipped and _03's baked sign works as-is; _07's entry
        neighbor (zone6) is NOT flipped and _07's baked sign needs negating.
        Loop 2's _11/_15 mirror this exactly (entry neighbors zone10/zone14).
        """
        n = len(self.zones)
        for i, zone in enumerate(self.zones):
            if not zone.is_straight:
                baked = zone.direction_attr.Get()
                prev_zone = self.zones[(i - 1) % n]
                next_center = self.zones[(i + 1) % n].bbox_center
                radius = 0.5 * math.dist(prev_zone.bbox_center, next_center)
                baked_sign = -1.0 if baked[2] < 0.0 else 1.0
                sign = baked_sign if self._is_body_flipped(prev_zone.belt_prim) else -baked_sign
                corrected = Gf.Vec3f(0.0, 0.0, sign / radius)
                if Gf.Vec3f(baked) != corrected:
                    print(
                        f"[conveyor_indexer] correcting {zone.node_path} inputs:direction "
                        f"{tuple(baked)} -> {tuple(corrected)} (radius={radius:.3f}m)",
                        flush=True,
                    )
                    zone.direction_attr.Set(corrected)
                continue

            if i + 1 < n:
                next_center = self.zones[i + 1].bbox_center
            elif self.closed_loop:
                next_center = self.zones[0].bbox_center
            else:
                # Last zone of an open line: no downstream neighbor to derive
                # a direction from - extrapolate from the previous zone's
                # center through this one instead (every straight run in
                # this scaffold is colinear) rather than wrapping back to
                # zone 0, which would point the wrong way (backwards into
                # the line) on anything but a real closed loop.
                prev_center = self.zones[i - 1].bbox_center
                next_center = (
                    2 * zone.bbox_center[0] - prev_center[0],
                    2 * zone.bbox_center[1] - prev_center[1],
                )
            dx = next_center[0] - zone.bbox_center[0]
            dy = next_center[1] - zone.bbox_center[1]
            corrected = Gf.Vec3f(1.0, 0.0, 0.0) if abs(dx) >= abs(dy) else Gf.Vec3f(0.0, 1.0, 0.0)
            if (dx if abs(dx) >= abs(dy) else dy) < 0.0:
                corrected = -corrected
            # World-space travel direction - used by is_past_center, which
            # only cares about actual world-space geometry and must NOT be
            # negated below (unlike what gets authored as inputs:direction).
            zone.world_travel_direction = Gf.Vec3f(corrected)

            # What actually gets authored as inputs:direction, by contrast,
            # DOES need a per-body sign flip here - the opposite of
            # racetrack.usd's own finding for its flipped rows (see this
            # method's docstring above: there, the SAME raw vector, un-negated,
            # was empirically correct for both flipped and unflipped rows).
            # Confirmed empirically against 5_conv_env.usd via the DEBUG box
            # position log in main(): with the un-negated vector authored,
            # every box on ConveyorTrack (zone 0) drifted steadily in +X -
            # AWAY from the pick zone at more negative X, off the belt's near
            # (open) edge entirely - the opposite of the intended travel
            # direction. Every track in 5_conv_env.usd happens to be
            # uniformly ~180deg-rotated about Z (confirmed via
            # _is_body_flipped on each), unlike racetrack.usd's per-row
            # split (some flipped, some not) - negating for that uniform
            # flip is what actually matches observed motion here. Why this
            # differs from racetrack.usd's finding isn't fully root-caused;
            # revisit both together if racetrack.usd is run again.
            authored = -corrected if self._is_body_flipped(zone.belt_prim) else corrected
            baked = zone.direction_attr.Get()
            if Gf.Vec3f(baked) != authored:
                print(
                    f"[conveyor_indexer] correcting {zone.node_path} inputs:direction "
                    f"{tuple(baked)} -> {tuple(authored)}",
                    flush=True,
                )
                zone.direction_attr.Set(authored)

    def step(self, state_msg, commands_msg) -> None:
        """Advance every zone by one control tick, appending into shared log messages."""
        self.occupied = [zone.check_occupied() for zone in self.zones]

        n = len(self.zones)
        for i, zone in enumerate(self.zones):
            # Closed loop: neighbors wrap around rather than terminating at
            # open ends. Open line (5_conv_env.usd): zone 0 has no real
            # upstream zone, so it's treated as always having more infeed
            # available (per ConveyorZoneStateMachine.step's own
            # upstream_occupied docstring) - moot in practice here since
            # this scaffold's boxes start out pre-placed directly on zone 0
            # rather than trickling in one at a time, so `occupied` alone
            # already drives EMPTY->IDLE; this only matters once zone 0 has
            # genuinely run dry of boxes. Symmetrically, the last zone hands
            # off to an unmodeled outfeed (here, boxes simply run off the
            # belt's end - into the truck, for loop 2), so its downstream is
            # always reported clear. A held zone (e.g. the pick zone) never
            # reports downstream_clear regardless, so it holds an arriving
            # item indefinitely instead of auto-advancing it further -
            # "starved" again only once whatever emptied it (the robot) lets
            # it go.
            if i == 0 and not self.closed_loop:
                upstream_occupied = True
            else:
                upstream_occupied = self.occupied[(i - 1) % n]

            # See set_hold_zone_ready_check: a hold zone only overflows (falls
            # through to the ordinary-zone rule) while its robot is busy - and
            # never if it's the line's last zone, which has nowhere to
            # overflow TO (it would just run off the belt's open end).
            is_last = i == n - 1 and not self.closed_loop
            robot_ready = i in self.hold_zone_indices and self._hold_zone_ready_checks.get(i, lambda: True)()
            holding = i in self.hold_zone_indices and (robot_ready or is_last)
            if holding:
                downstream_clear = False
            elif is_last:
                downstream_clear = True
            else:
                downstream_clear = not self.occupied[(i + 1) % n]

            # Only a zone that's actually holding defines a stop position
            # (see ConveyorZone.is_past_center) - every other zone (including
            # a hold zone currently overflowing) defaults to True, reproducing
            # the old stop-as-soon-as-occupied behavior.
            at_stop_position = True
            if holding and self._box_rigid_prims is not None and self.occupied[i]:
                occupying_paths = zone.get_occupying_prim_paths()
                box_rigid_prim = self._box_rigid_prims.get(occupying_paths[0]) if occupying_paths else None
                if box_rigid_prim is not None:
                    position, _ = box_rigid_prim.get_world_poses()
                    # is_last has no downstream to overshoot into but open
                    # ground - stop much earlier, leaving most of the belt
                    # as deceleration runway instead of just the back half.
                    stop_fraction = 0.18 if is_last else 0.5
                    at_stop_position = zone.is_past_center(position.numpy()[0], stop_fraction=stop_fraction)

            observation, command = zone.state_machine.step(
                occupied=self.occupied[i],
                upstream_occupied=upstream_occupied,
                downstream_clear=downstream_clear,
                at_stop_position=at_stop_position,
            )
            if zone._nudge_ticks_remaining > 0:
                # Overrides the state machine's own command (e.g. a held pick
                # zone's run=False) for a few ticks - see ConveyorZone.nudge().
                command.run = True
                command.speed_pct = zone._nudge_speed_pct
                zone._nudge_ticks_remaining -= 1
            if DEBUG_LOG_HOLD_ZONE_STATE and i in self.hold_zone_indices:
                print(
                    f"[conveyor_indexer] DEBUG hold zone: {zone.node_path} occupied={self.occupied[i]} "
                    f"holding={holding} downstream_clear={downstream_clear} at_stop_position={at_stop_position} "
                    f"machine={observation.machine} run={command.run}",
                    flush=True,
                )
            zone.apply_command(command.run, command.speed_pct)
            self.machine_states[i] = observation.machine

            item = state_msg.Conveyors.add()
            item.Name = zone.node_path
            item.Type = plc.ConveyorTypeCode.CONVEYOR_TYPE_BUFFER  # TODO: set real per-zone type
            item.Fault = plc.ConveyorFaultCode.CONVEYOR_FAULT_UNSPECIFIED  # not modeled yet, see README
            item.PackML = self._machine_to_packml(observation.machine)
            item.Speed = command.speed_pct
            item.Direction = command.direction
            item.Machine = observation.machine

            cmd = commands_msg.commands.add()
            cmd.zone_index = i
            cmd.conveyor_node_path = zone.node_path
            cmd.run = command.run
            cmd.speed = command.speed_pct
            cmd.direction = command.direction

    @staticmethod
    def _machine_to_packml(machine) -> int:
        """Coarse, documented-as-approximate PackML projection - see README.

        Real PackML reporting on the physical line almost certainly carries
        more nuance (Starting/Stopping/Held/Aborted transitions) than this
        two-bucket mapping. Revisit once real behavior is available, same
        caveat as the Machine state machine itself.
        """
        Machine = plc.ConveyorStateMachineCode
        running_states = {
            Machine.CONVEYOR_STATE_MACHINE_INDUCTING,
            Machine.CONVEYOR_STATE_MACHINE_ADVANCE_ITEM,
            Machine.CONVEYOR_STATE_MACHINE_PASSTHROUGH,
        }
        return common_types.PACKML_EXECUTE if machine in running_states else common_types.PACKML_IDLE


def _discover_box_prim_paths(stage: Usd.Stage) -> list:
    """Find every pre-authored `CubeBox_*` prim - 5_conv_env.usd's pallet of
    boxes stacked directly on ConveyorTrack's belt (confirmed via world-bbox
    inspection: every one sits within ConveyorTrack's belt-top XY footprint,
    in two stacked Z layers), unlike racetrack.usd, which ships with no
    boxes and needed them referenced in at runtime (see git history).

    Matches top-level box instances only (direct children of either the
    stage root or `/World` - 5_conv_env.usd has both: some boxes parented
    under `/World`, most left at the stage root, apparently from how the
    scene was assembled in Create; harmless either way, this scaffold only
    needs each box's own prim path), not their child meshes (also named with
    this prefix's asset naming, e.g. `SM_CubeBox_A04_Body_01`, which doesn't
    itself start with `BOX_PRIM_NAME_PREFIX`).
    """
    paths = []
    for prim in stage.Traverse():
        if not prim.GetName().startswith(BOX_PRIM_NAME_PREFIX):
            continue
        parent = prim.GetParent()
        if parent.IsPseudoRoot() or parent.GetPath() == Sdf.Path("/World"):
            paths.append(str(prim.GetPath()))
    return sorted(paths)


# Iteration cap for _resolve_box_overlaps's separation passes, and the extra
# gap (beyond just-touching) left between two boxes once separated - small
# enough to be visually unnoticeable, large enough that the two don't start
# back in (near-)contact and immediately re-trigger PhysX's own depenetration
# push once physics starts.
BOX_OVERLAP_RESOLVE_MAX_PASSES = 8
BOX_OVERLAP_CLEARANCE = 0.002  # meters


def _resolve_box_overlaps(stage: Usd.Stage, box_paths: list) -> None:
    """Nudge apart any box prims whose world AABBs actually overlap, before
    physics ever runs.

    5_conv_env.usd's pre-authored pallet places each box by hand; direct
    inspection of the authored (pre-physics) positions found every pair's
    clearance uncomfortably tight - some margins as small as ~2 cm - and
    running the sim confirmed the consequence: boxes were observed drifting
    tens of centimeters within the very first tick and continuing to spread
    over the following seconds, well beyond gentle settling, evidently from
    real (if small) initial interpenetration somewhere in the stack that
    BOX_MAX_DEPENETRATION_VELOCITY's cap alone wasn't enough to keep
    contained to the belt. Rather than guess at which pair(s) and hand-author
    a fix, this resolves it generally: repeatedly scans every pair for AABB
    overlap and, for any that overlap, pushes the second of the pair away
    from the first along whichever axis has the SMALLEST overlap depth (a
    standard minimum-translation-vector separation) - preserving the
    designer's intended layout as closely as possible rather than
    rearranging boxes wholesale. Runs to a fixed pass cap rather than
    looping until clean, since resolving one pair can (rarely) reintroduce a
    small overlap with a third box; a handful of passes is enough for ~19
    boxes with only marginal initial overlaps.
    """

    def _translate_op(prim: Usd.Prim) -> UsdGeom.XformOp:
        for op in UsdGeom.Xformable(prim).GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                return op
        raise RuntimeError(f"{prim.GetPath()} has no xformOp:translate")

    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    prims = [stage.GetPrimAtPath(path) for path in box_paths]
    translate_ops = [_translate_op(prim) for prim in prims]

    total_nudges = 0
    for _ in range(BOX_OVERLAP_RESOLVE_MAX_PASSES):
        bounds = [bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange() for prim in prims]
        any_overlap = False
        for i in range(len(prims)):
            for j in range(i + 1, len(prims)):
                min_i, max_i = bounds[i].GetMin(), bounds[i].GetMax()
                min_j, max_j = bounds[j].GetMin(), bounds[j].GetMax()
                overlap = [min(max_i[axis], max_j[axis]) - max(min_i[axis], min_j[axis]) for axis in range(3)]
                if not all(depth > 0.0 for depth in overlap):
                    continue
                any_overlap = True
                axis = min(range(3), key=lambda a: overlap[a])
                push = overlap[axis] / 2.0 + BOX_OVERLAP_CLEARANCE
                sign = 1.0 if (min_j[axis] + max_j[axis]) >= (min_i[axis] + max_i[axis]) else -1.0
                delta = [0.0, 0.0, 0.0]
                delta[axis] = sign * push
                current = translate_ops[j].Get()
                translate_ops[j].Set(type(current)(current[0] + delta[0], current[1] + delta[1], current[2] + delta[2]))
                bounds[j] = bbox_cache.ComputeWorldBound(prims[j]).ComputeAlignedRange()
                total_nudges += 1
        if not any_overlap:
            break
    print(f"[conveyor_indexer] resolved box overlaps with {total_nudges} nudge(s)", flush=True)


def _apply_box_physics(stage: Usd.Stage, box_paths: list) -> None:
    """Add RigidBodyAPI + convex-hull CollisionAPI + a mass to every box prim.

    5_conv_env.usd's CubeBox_* prims are pure visual payloads with no physics
    schemas at all (confirmed via direct inspection) - unlike racetrack.usd's
    referenced sm_box_multiDepth_brown_b08_01 boxes, which carry physics
    baked into the referenced asset itself. Applied to each box's own root
    Xform, the same pattern already used for every ConveyorTrack's `Belt`
    (RigidBodyAPI + CollisionAPI on the Xform rather than a specific child
    Mesh) - PhysX resolves the actual collision geometry from the Mesh prims
    nested underneath either way. Convex-hull (not the default exact
    triangle mesh) since these are dynamic bodies that will contact each
    other and the moving belt - the same approximation
    `_build_motion_planner` already prefers for cuMotion's obstacle Mesh
    prims, for the same dynamic-contact-stability reason. Also caps each
    box's `maxDepenetrationVelocity` - see BOX_MAX_DEPENETRATION_VELOCITY.
    """
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    for path in box_paths:
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            raise RuntimeError(f"Expected box prim not found at {path}")
        UsdPhysics.RigidBodyAPI.Apply(prim)
        UsdPhysics.CollisionAPI.Apply(prim)
        UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr().Set("convexHull")
        PhysxSchema.PhysxRigidBodyAPI.Apply(prim).CreateMaxDepenetrationVelocityAttr().Set(
            BOX_MAX_DEPENETRATION_VELOCITY
        )

        size = bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange().GetSize()
        mass_kg = size[0] * size[1] * size[2] * BOX_DENSITY_KG_PER_M3
        UsdPhysics.MassAPI.Apply(prim).CreateMassAttr().Set(mass_kg)
    print(f"[conveyor_indexer] applied rigid-body physics to {len(box_paths)} boxes", flush=True)


# Each ConveyorTrack authors TWO separate collision meshes, confirmed by
# traversing the stage: `SM_ConveyorBelt_A06_02` (the frame/upright
# posts/bolts, an ordinary static Mesh) and `Belt` (just the moving surface,
# under an Xform with PhysxSurfaceVelocityAPI). Only the latter fails
# cuMotion's RMPflow obstacle world (`WorldBinding.initialize()`, called from
# pick_and_place.py's `_build_rmpflow_controller`) - confirmed two different
# ways (non-unity-scaled ancestors on some tracks; the belt mesh isn't one of
# the supported obstacle shape types at all on others), both pre-existing
# authoring quirks of the underlying ConveyorBelt_A06 asset, not something
# this scaffold edits into the source scene.
#
# An early attempt here blanket-excluded ALL conveyor structure (both
# meshes) to route around this - which meant the frame/upright posts were no
# longer tracked either, and the arm was observed physically colliding with
# them. `_build_rmpflow_controller`'s own retry loop already discovers and
# excludes exactly the prims that fail (belt surfaces, occasional
# non-unity-scaled ancestors) one at a time, so no exclusion list needs to be
# passed from here at all - the frame/posts mesh, which never fails those
# checks, stays tracked and avoided.
#
# A separate, even earlier attempt created synthetic capsule obstacles sized
# from each zone's bbox as a substitute for the untrackable belt surface.
# That was wrong in a more basic way than sizing, though: applying
# UsdPhysics.CollisionAPI makes a prim a REAL PhysX collider, not just a
# planning-time hint - the capsules physically collided with and shoved the
# actual boxes on the belt (confirmed visually - grossly oversized red
# capsules overlapping the real boxes/structure). There's no "obstacle hint,
# not a real object" middle ground via this API.


def _discover_cumotion_ext_paths() -> list[str]:
    """Return the on-disk dirs for the `warp` and `cumotion` packages.

    These ship as Isaac Sim extensions whose sys.path entries are added by
    Kit's extension manager; the planner subprocess (which never boots Kit)
    can't find them on its own, so this process - which HAS Kit - resolves
    them from the loaded modules and passes them down. Nothing is hardcoded.
    """
    import warp

    paths = [os.path.dirname(os.path.dirname(os.path.abspath(warp.__file__)))]
    try:
        import cumotion
    except ImportError:
        import isaacsim.robot_motion.cumotion  # noqa: F401  (loads the ext, adds cumotion to sys.path)
        import cumotion
    paths.append(os.path.dirname(os.path.dirname(os.path.abspath(cumotion.__file__))))
    return paths


def _build_obstacle_specs(zones: list, robot_positions: list) -> list[dict]:
    """World-frame cuboid obstacle specs for the planner subprocess.

    The old design let cuMotion auto-scan the USD stage; the direct-API
    subprocess can't, so we hand it exactly the obstacles that matter here:
    each robot's pedestal, and each conveyor as a solid block from the floor up
    to its belt-top surface (belt XY footprint). The truck is out of reach (per
    explicit confirmation) and omitted; boxes are the manipulated targets and
    never obstacles; the arms are separated so they don't model each other.
    Each spec is a cuboid {side_lengths, position (world), orientation wxyz};
    the child re-expresses them in each robot's base frame.
    """
    specs = []
    for px, py, pz in robot_positions:
        specs.append(
            {
                "side_lengths": [2 * PEDESTAL_RADIUS, 2 * PEDESTAL_RADIUS, PEDESTAL_HEIGHT],
                "position": [px, py, pz + PEDESTAL_HEIGHT / 2.0],
                "orientation": [1.0, 0.0, 0.0, 0.0],
            }
        )
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    for zone in zones:
        aligned = bbox_cache.ComputeWorldBound(zone.belt_prim).ComputeAlignedRange()
        mn, mx = aligned.GetMin(), aligned.GetMax()
        belt_top = mx[2]
        specs.append(
            {
                "side_lengths": [float(mx[0] - mn[0]), float(mx[1] - mn[1]), float(belt_top)],
                "position": [float(0.5 * (mn[0] + mx[0])), float(0.5 * (mn[1] + mx[1])), float(belt_top / 2.0)],
                "orientation": [1.0, 0.0, 0.0, 0.0],
            }
        )
    return specs


def _spawn_planner_subprocess(init_config: dict, ext_paths: list[str]):
    """Launch the out-of-process cuMotion planner and hand it its init config.

    cuMotion planning holds the GIL for the whole solve, so it runs in this
    separate process (its own GIL + CUDA context) rather than blocking the main
    sim loop - see planner_server.py / planner_server_impl.py. The subprocess
    uses cuMotion's low-level API directly (no SimulationApp), so it starts in
    ~1-2s. Launched via Popen (NOT multiprocessing spawn, which would re-import
    this module's top-level SimulationApp and start a second visible sim). IPC
    is a full-duplex multiprocessing.connection over an AF_UNIX socket.

    Blocks at accept() until the child has built every planner and reports
    ready. Returns (listener, child, conn) for the caller to wrap in a
    PlannerClient and to tear down at the end.
    """
    address = os.path.join(tempfile.gettempdir(), f"conveyor_planner_{os.getpid()}.sock")
    if os.path.exists(address):
        os.unlink(address)
    authkey = os.urandom(16)
    listener = Listener(address, authkey=authkey)
    server_script = os.path.join(REPO_DIR, "planner_server.py")
    argv = [sys.executable, server_script, "--addr", address, "--authkey", authkey.hex()]
    for path in ext_paths:
        argv += ["--ext-path", path]

    def _set_pdeathsig():
        # Linux: ask the kernel to SIGKILL this child the instant the parent
        # dies, for ANY reason (including a hard `kill -9` that bypasses our
        # finally). Without this, a hard-killed parent orphans the planner,
        # which keeps holding a CUDA context and makes the next run's GPU init
        # stall. prctl(PR_SET_PDEATHSIG=1, SIGKILL).
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGKILL)

    child = subprocess.Popen(argv, cwd=REPO_DIR, preexec_fn=_set_pdeathsig)
    print(
        f"[conveyor_indexer] launched planner subprocess pid={child.pid}; "
        "waiting for it to build its planners...",
        flush=True,
    )
    conn = listener.accept()  # blocks until the child connects (after its startup)
    conn.send({"type": "init", **init_config})
    ready = conn.recv()
    if ready.get("type") != "ready":
        raise RuntimeError(f"planner subprocess sent {ready!r} instead of a ready message")
    print("[conveyor_indexer] planner subprocess ready", flush=True)
    return listener, child, conn


def main() -> None:
    # isaacsim.asset.gen.conveyor (provides the ConveyorNode OmniGraph node
    # type every ConveyorBeltGraph uses) is not guaranteed to already be
    # enabled by the app config SimulationApp resolves - confirmed directly:
    # a run against this exact app config logged 16 (one per ConveyorNode)
    # "Could not find node type interface for
    # 'isaacsim.asset.gen.conveyor.IsaacConveyor'" warnings and no box ever
    # reached the pick zone in 15+ seconds of sim time, i.e. no belt actually
    # moved anything despite `inputs:enabled`/Velocity being authored
    # correctly. Enabling it explicitly, before the stage (and its
    # ConveyorBeltGraphs) is even opened, makes this independent of whatever
    # extensions the resolved app happens to auto-load.
    omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate(
        "isaacsim.asset.gen.conveyor", True
    )

    # Open the target stage BEFORE constructing World: World() attaches to
    # whatever stage is open at construction time, and open_stage() after the
    # fact replaces the stage out from under it (World._scene ends up
    # referencing a physics_sim_view tied to the old, now-gone stage -
    # observed as "AttributeError: 'World' object has no attribute '_scene'"
    # inside world.reset() when this ordering was wrong).
    localize_asset_references(STAGE_PATH, REMOTE_ASSET_ROOT, LOCAL_ASSET_ROOT)
    print(f"[conveyor_indexer] opening stage {STAGE_PATH}", flush=True)
    ctx = omni.usd.get_context()
    ctx.open_stage(STAGE_PATH)
    stage = ctx.get_stage()
    deactivate_frame_meshes(stage, CONVEYOR_TRACK_ROOTS)
    apply_truck_collision(stage, TRUCK_PATH)
    box_paths = _discover_box_prim_paths(stage)
    if not box_paths:
        raise RuntimeError(
            f"No prims named '{BOX_PRIM_NAME_PREFIX}*' found in {STAGE_PATH} - "
            "expected the pre-authored box pallet on ConveyorTrack"
        )
    _resolve_box_overlaps(stage, box_paths)
    _apply_box_physics(stage, box_paths)
    print("[conveyor_indexer] stage open, constructing World", flush=True)

    world = World(physics_dt=1.0 / 120.0, rendering_dt=1.0 / 60.0, stage_units_in_meters=1.0)
    print("[conveyor_indexer] World constructed, building logger + zones", flush=True)

    # Pre-create the log directory rather than relying on
    # ConveyorIndexingLogger's exists-at-call-time heuristic (mirrors
    # data_collection_vol2.py's _resolve_parquet_path) to decide whether
    # LOG_OUTPUT_DIR is a directory or a file stem - on a clean checkout
    # (directory doesn't exist yet) that heuristic instead treats it as a
    # file stem, e.g. writing conveyor_indexing/data.parquet directly rather
    # than a timestamped file under conveyor_indexing/data/.
    pathlib.Path(LOG_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    logger = ConveyorIndexingLogger(LOG_OUTPUT_DIR)

    loop1 = ConveyorLineController(
        stage,
        ZONE_NODE_PATHS_LOOP1,
        hold_zone_indices=frozenset({PICK_ZONE_INDEX, PICK_ZONE_INDEX_2}),
        closed_loop=False,
        run_speed_pct=LOOP1_RUN_SPEED_PCT,
    )
    loop2 = ConveyorLineController(
        stage, ZONE_NODE_PATHS_LOOP2, closed_loop=False, run_speed_pct=LOOP2_RUN_SPEED_PCT
    )
    print(f"[conveyor_indexer] zones built, {len(box_paths)} boxes discovered, creating pedestal + robot", flush=True)

    robot = create_pedestal_and_robot(
        stage,
        robot_path=ROBOT_PATH,
        pedestal_path=PEDESTAL_PATH,
        position=ROBOT_POSITION,
        pedestal_height=PEDESTAL_HEIGHT,
    )

    # Reach-balanced position for the second robot, derived from actual zone
    # bbox geometry rather than hardcoded - loop 1 has 3 zones and loop 2 only
    # 2, so ConveyorTrack_02/_10 aren't guaranteed to line up in X the way
    # ConveyorTrack_01/_09 happen to (see ROBOT_POSITION's own comment).
    pick_zone_2 = loop1.zones[PICK_ZONE_INDEX_2]
    place_zone_2 = loop2.zones[PLACE_ZONE_INDEX_2]
    robot_2_x = (pick_zone_2.bbox_center[0] + place_zone_2.bbox_center[0]) / 2.0
    robot_2_y = (
        pick_zone_2.bbox_center[1] + pick_zone_2.bbox_half_extent[1]
        + place_zone_2.bbox_center[1] - place_zone_2.bbox_half_extent[1]
    ) / 2.0
    ROBOT_POSITION_2 = (robot_2_x, robot_2_y, 0.0)
    PLACE_XY_2 = (place_zone_2.bbox_center[0], place_zone_2.bbox_center[1])
    print(
        f"[conveyor_indexer] DEBUG robot 2 geometry: ROBOT_POSITION_2={ROBOT_POSITION_2} "
        f"PLACE_XY_2={PLACE_XY_2} reach_to_pick={math.dist((robot_2_x, robot_2_y), pick_zone_2.bbox_center[:2]):.3f}m "
        f"reach_to_place={math.dist((robot_2_x, robot_2_y), place_zone_2.bbox_center[:2]):.3f}m (UR20 spec ~1.75m)",
        flush=True,
    )

    robot2 = create_pedestal_and_robot(
        stage,
        robot_path=ROBOT_PATH_2,
        pedestal_path=PEDESTAL_PATH_2,
        position=ROBOT_POSITION_2,
        pedestal_height=PEDESTAL_HEIGHT,
    )

    # Place target Z depends on box height, computed per-cycle inside
    # MagicAttachPickPlace since any of the discovered boxes (see
    # _discover_box_prim_paths) could be the one being carried; only the
    # belt-top Z is fixed here.
    place_belt_prim = loop2.zones[PLACE_ZONE_INDEX].belt_prim
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    place_belt_top_z = bbox_cache.ComputeWorldBound(place_belt_prim).ComputeAlignedRange().GetMax()[2]
    place_belt_top_z_2 = bbox_cache.ComputeWorldBound(place_zone_2.belt_prim).ComputeAlignedRange().GetMax()[2]

    box_rigid_prims = {path: RigidPrim(path) for path in box_paths}
    # Only loop1 has hold zones (the pick zones) that need is_past_center()
    # checks against a real box position - see ConveyorLineController.step().
    loop1.set_box_rigid_prims(box_rigid_prims)
    print("[conveyor_indexer] robots ready, calling world.reset()", flush=True)

    world.reset()
    # World.reset() (isaacsim.core.api, the classic API) has no knowledge of
    # `isaacsim.core.experimental` prims like our plain `Articulation` robot -
    # confirmed by reading world.py, it never calls reset_to_default_state()
    # on them. So the dof_positions configured via set_default_state() in
    # create_pedestal_and_robot() was silently never applied - the arm always
    # started from its raw near-zero USD-authored pose regardless of what
    # was configured, which was the real reason changing
    # UR20_DEFAULT_JOINT_POSITIONS appeared to have no effect. Must be
    # triggered explicitly.
    robot.reset_to_default_state()
    robot2.reset_to_default_state()
    check_pos, _ = robot.get_world_poses()
    print(f"[conveyor_indexer] DEBUG robot base pose AFTER world.reset(): {check_pos.numpy()}", flush=True)
    print(f"[conveyor_indexer] DEBUG robot dof_positions AFTER reset_to_default_state(): {robot.get_dof_positions().numpy()}", flush=True)
    print("[conveyor_indexer] world.reset() done", flush=True)

    # Motion planning runs in a dedicated subprocess (see
    # _spawn_planner_subprocess and planner_server_impl.py) so a cuMotion solve
    # doesn't block the main loop. It uses cuMotion's low-level API directly (no
    # SimulationApp): we discover the warp/cumotion extension dirs here (this
    # process has Kit) and pass them down, and hand it the robots' base poses
    # plus explicit obstacle cuboids (this process has the geometry; the child
    # has no zone tables / USD stage). Robot base = pedestal top, identity
    # orientation (create_pedestal_and_robot only translates).
    ext_paths = _discover_cumotion_ext_paths()
    obstacle_specs = _build_obstacle_specs(
        list(loop1.zones) + list(loop2.zones),
        [ROBOT_POSITION, ROBOT_POSITION_2],
    )
    base_orientation = [1.0, 0.0, 0.0, 0.0]
    robot_base_0 = [float(ROBOT_POSITION[0]), float(ROBOT_POSITION[1]), float(ROBOT_POSITION[2]) + PEDESTAL_HEIGHT]
    robot_base_1 = [float(ROBOT_POSITION_2[0]), float(ROBOT_POSITION_2[1]), float(ROBOT_POSITION_2[2]) + PEDESTAL_HEIGHT]
    planner_init = {
        "physics_dt": world.get_physics_dt(),
        "robots": [
            {
                "robot_id": 0,
                "xrdf_path": UR20_XRDF_PATH,
                "urdf_path": UR20_URDF_PATH,
                "base_position": robot_base_0,
                "base_orientation": base_orientation,
                "tool_frame": UR20_TOOL_FRAME,
                "max_velocities": MAX_JOINT_VELOCITIES,
                "max_accelerations": MAX_JOINT_ACCELERATIONS,
                "obstacles": obstacle_specs,
            },
            {
                "robot_id": 1,
                "xrdf_path": UR20_XRDF_PATH,
                "urdf_path": UR20_URDF_PATH,
                "base_position": robot_base_1,
                "base_orientation": base_orientation,
                "tool_frame": UR20_TOOL_FRAME,
                "max_velocities": MAX_JOINT_VELOCITIES,
                "max_accelerations": MAX_JOINT_ACCELERATIONS,
                "obstacles": obstacle_specs,
            },
        ],
    }
    planner_listener, planner_child, planner_conn = _spawn_planner_subprocess(planner_init, ext_paths)
    planner_client = PlannerClient(planner_conn)

    # MagicAttachPickPlace is now the parent-side pick/place state machine +
    # trajectory playback (no cuMotion). It ships plan requests to the
    # subprocess via planner_client/robot_id and plays back the sampled
    # trajectories it returns.
    # pre_place_joint_positions override: robot 2 sits on this robot's -X
    # side, so its STAGE_FOR_PICK<->STAGE_FOR_PLACE swing must arc away from
    # it instead - see UR20_PRE_PLACE_JOINT_POSITIONS_AWAY.
    pick_place = MagicAttachPickPlace(
        robot=robot,
        robot_path=ROBOT_PATH,
        place_xy=PLACE_XY,
        place_belt_top_z=place_belt_top_z,
        box_rigid_prims=box_rigid_prims,
        physics_dt=world.get_physics_dt(),
        planner_client=planner_client,
        robot_id=0,
        pre_place_joint_positions=UR20_PRE_PLACE_JOINT_POSITIONS_AWAY,
        nudge_pick_zone_fn=lambda: loop1.zones[PICK_ZONE_INDEX].nudge(
            PICK_REPLAN_NUDGE_TICKS, PICK_REPLAN_NUDGE_SPEED_PCT
        ),
    )
    pick_place_2 = MagicAttachPickPlace(
        robot=robot2,
        robot_path=ROBOT_PATH_2,
        place_xy=PLACE_XY_2,
        place_belt_top_z=place_belt_top_z_2,
        box_rigid_prims=box_rigid_prims,
        physics_dt=world.get_physics_dt(),
        planner_client=planner_client,
        robot_id=1,
        nudge_pick_zone_fn=lambda: loop1.zones[PICK_ZONE_INDEX_2].nudge(
            PICK_REPLAN_NUDGE_TICKS, PICK_REPLAN_NUDGE_SPEED_PCT
        ),
    )
    loop1.set_hold_zone_ready_check(PICK_ZONE_INDEX, lambda: pick_place.phase_name == "WAITING")
    loop1.set_hold_zone_ready_check(PICK_ZONE_INDEX_2, lambda: pick_place_2.phase_name == "WAITING")
    print("[conveyor_indexer] pick/place controllers ready, entering main loop", flush=True)

    control_period_s = 1.0 / CONTROL_HZ
    last_control_time = 0.0
    sim_time = 0.0
    render_count = 0
    tick = 0
    pick_ready = False
    pick_box_path = None
    pick_ready_2 = False
    pick_box_path_2 = None

    # SIGTERM (container/systemd shutdown, `kill` without -INT) and SIGINT
    # (Ctrl+C) otherwise terminate the process abruptly - bypassing the
    # `finally` below, so the parquet writer never gets a clean close() (the
    # file is left truncated/unreadable) AND the planner subprocess + its
    # socket are left to be reaped by PR_SET_PDEATHSIG instead of shut down
    # cleanly. Route both through the normal loop-exit -> finally path.
    # (Note: this only helps once the main loop is running; a signal during
    # Kit's ~30s SimulationApp startup is queued but not serviced until Kit's
    # C++ init returns - unavoidable from here.)
    shutdown_requested = False

    def _handle_shutdown_signal(signum, frame):
        nonlocal shutdown_requested
        shutdown_requested = True

    signal.signal(signal.SIGTERM, _handle_shutdown_signal)
    signal.signal(signal.SIGINT, _handle_shutdown_signal)

    Machine = plc.ConveyorStateMachineCode

    try:
        while simulation_app.is_running() and not shutdown_requested:
            if world.is_playing():
                world.step(render=True); sim_time += world.get_physics_dt()

                # Drain any plan results the planner subprocess has returned
                # since last tick (non-blocking) before the robots poll for
                # them below - see PlannerClient.
                planner_client.pump()

                # Pick-and-place motion runs every physics step; conveyor
                # indexing runs at the coarser control rate. `pick_ready`/
                # `pick_box_path` only refresh at the control rate but are read
                # every physics step - fine, since they only matter at the
                # (infrequent) moment WAITING checks them. A robot whose plan
                # is still being computed just holds its pose (see _drive_to),
                # so this loop keeps stepping the other robot and the belts.
                pick_place.forward(pick_ready, pick_box_path)
                pick_place_2.forward(pick_ready_2, pick_box_path_2)

                if sim_time - last_control_time >= control_period_s:
                    state_msg = plc.StateConveyors()
                    commands_msg = sim_action.SimConveyorCommands()
                    loop1.step(state_msg, commands_msg)
                    loop2.step(state_msg, commands_msg)
                    logger.log_tick(
                        tick=tick,
                        sim_time_s=sim_time,
                        plc_state_conveyors=state_msg.SerializeToString(),
                        conveyor_commands=commands_msg.SerializeToString(),
                    )
                    # Only "ready" once the pick zone's own state machine has
                    # settled into holding (IDLE + occupied) - not mid-induction.
                    # Identify WHICH box is actually there (any of the
                    # discovered boxes can end up in this zone) rather than
                    # assuming a fixed one.
                    pick_zone_hits = loop1.zones[PICK_ZONE_INDEX].get_occupying_prim_paths()
                    pick_ready = (
                        bool(pick_zone_hits)
                        and loop1.machine_states[PICK_ZONE_INDEX] == Machine.CONVEYOR_STATE_MACHINE_IDLE
                    )
                    pick_box_path = pick_zone_hits[0] if pick_zone_hits else None
                    pick_zone_2_hits = loop1.zones[PICK_ZONE_INDEX_2].get_occupying_prim_paths()
                    pick_ready_2 = (
                        bool(pick_zone_2_hits)
                        and loop1.machine_states[PICK_ZONE_INDEX_2] == Machine.CONVEYOR_STATE_MACHINE_IDLE
                    )
                    pick_box_path_2 = pick_zone_2_hits[0] if pick_zone_2_hits else None
                    last_control_time = sim_time
                    tick += 1
                    if tick % 3 == 0 and pick_box_path is not None:
                        pick_zone = loop1.zones[PICK_ZONE_INDEX]
                        box_pos, _ = box_rigid_prims[pick_box_path].get_world_poses()
                        box_x = box_pos.numpy()[0][0]
                        print(
                            f"[conveyor_indexer] DEBUG pick zone centering: box={pick_box_path} "
                            f"box_x={box_x:.3f} target_x={pick_zone.bbox_center[0]:.3f} "
                            f"machine={loop1.machine_states[PICK_ZONE_INDEX]}",
                            flush=True,
                        )
                    if tick % 3 == 0:
                        print(
                            f"[conveyor_indexer] tick={tick} sim_time={sim_time:.2f} "
                            f"pick_phase={pick_place.phase_name} pick_ready={pick_ready} "
                            f"pick_phase_2={pick_place_2.phase_name} pick_ready_2={pick_ready_2}",
                            flush=True,
                        )
                        # TEMPORARY debug tracking - remove once direction issue is resolved.
                        zone0 = loop1.zones[0]
                        z0_dir = zone0.direction_attr.Get()
                        z0_enabled = zone0.node_prim.GetAttribute("inputs:enabled").Get()
                        z0_vel = zone0.velocity_var_attr.Get()
                        print(
                            f"[DEBUG] ConveyorTrack (zone0) direction={tuple(z0_dir)} "
                            f"enabled={z0_enabled} velvar={z0_vel} machine={loop1.machine_states[0]}",
                            flush=True,
                        )
                        for box_path, rigid_prim in box_rigid_prims.items():
                            pos, _ = rigid_prim.get_world_poses()
                            p = pos.numpy().tolist()[0]
                            print(f"[DEBUG] {box_path} pos=({p[0]:.3f},{p[1]:.3f},{p[2]:.3f})", flush=True)
            else:
                render_count += 1
                if render_count % 60 == 1:
                    print(f"[conveyor_indexer] world not playing (render_count={render_count})", flush=True)
                world.render()
    finally:
        # Tell the planner subprocess to stop, then reap it so it doesn't
        # linger; a short wait, escalating to kill if it won't exit.
        planner_client.close()
        try:
            planner_child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            print("[conveyor_indexer] planner subprocess did not exit; killing it", flush=True)
            planner_child.kill()
        planner_listener.close()
        # Listener.close() doesn't always remove the AF_UNIX socket file; do it
        # explicitly so a killed run doesn't leave /tmp litter.
        try:
            os.unlink(planner_listener.address)
        except OSError:
            pass
        logger.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
