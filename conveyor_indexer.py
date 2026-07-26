"""Zone-accumulation indexing over 5_conv_env.usd's two open conveyor lines, with
per-tick logging and a UR20 pick-and-place (cuMotion RMPflow) moving boxes from
loop 1 to loop 2 and into a waiting SteelBoxTruck.

Run with Isaac Sim's bundled python: ./python.sh ~/conveyor_indexing/conveyor_indexer.py
See README.md for setup and known gaps.
"""

from __future__ import annotations

import math
import os
import pathlib
import signal
import sys

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
from pick_and_place import create_pedestal_and_robot, MagicAttachPickPlace, UR20_PRE_PLACE_JOINT_POSITIONS_AWAY

STAGE_PATH = os.path.join(os.path.expanduser("~"), "5_conv_env.usd")
LOG_OUTPUT_DIR = os.path.join(REPO_DIR, "data")
CONTROL_HZ = 120.0  # matches physics rate; 30Hz let boxes drift past hold points before the belt reacted
DEBUG_LOG_OCCUPANCY_HITS = False  # set True to print which prim triggers each occupancy hit
DEBUG_LOG_HOLD_ZONE_STATE = False  # set True to print every hold zone's state machine each control tick

# 5_conv_env.usd fetches its assets from this public S3 bucket over HTTPS every run unless
# localized; download_assets.py (see README) mirrors them locally. See _localize_asset_references.
REMOTE_ASSET_ROOT = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/"
LOCAL_ASSET_ROOT = os.path.join(os.path.expanduser("~"), "isaac_assets")

# Two independent open (non-looping) lines: loop 1 runs along Y=0, loop 2 along Y~2.186.
# Loop 2's far end sits at the near wall of SteelBoxTruck_A01_01; boxes run off the belt
# there and drop into the truck bed rather than handing off to another zone.
ZONE_NODE_PATHS_LOOP1 = [
    "/World/ConveyorTrack/ConveyorBeltGraph/ConveyorNode",
    "/World/ConveyorTrack_01/ConveyorBeltGraph/ConveyorNode",
    "/World/ConveyorTrack_02/ConveyorBeltGraph/ConveyorNode",
]
ZONE_NODE_PATHS_LOOP2 = [
    "/World/ConveyorTrack_09/ConveyorBeltGraph/ConveyorNode",
    "/World/ConveyorTrack_10/ConveyorBeltGraph/ConveyorNode",
]

# ConveyorTrack_01/_09 face each other at local X=-3 - the natural spot for a fixed
# pick/place robot. ConveyorTrack_02 (X=-5) is out of a UR20's reach from there, so
# it's left as an unused upstream buffer.
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

# The ground plane's collider breaks cuMotion's obstacle scan (recursion bug in
# pick_and_place.py's np.reshape shim); excluded via extra_exclude_obstacle_paths instead.
GROUND_PLANE_COLLISION_PATH = "/World/GroundPlane/CollisionPlane"

# 5_conv_env.usd ships ~18 pre-placed CubeBox_* prims with no physics schemas of their
# own; discovered at runtime (_discover_box_prim_paths) and given physics by _apply_box_physics.
BOX_PRIM_NAME_PREFIX = "CubeBox_"

BOX_DENSITY_KG_PER_M3 = 150.0  # plausible ballpark for a packed shipping box; not measured

# Caps how fast PhysX may push overlapping bodies apart per step. Some boxes start out
# interpenetrating (pallet Z-layer spacing is tighter than the tallest box variant);
# without this cap PhysX's depenetration impulse flung boxes clean off the belt.
BOX_MAX_DEPENETRATION_VELOCITY = 0.5  # m/s

# Zone velocity at 100% speed; each loop scales its actual run speed down from this via
# LOOP1_RUN_SPEED_PCT / LOOP2_RUN_SPEED_PCT.
ZONE_RUN_VELOCITY = 2.0

# Slowed for a comfortable pick cadence. A bigger global slowdown (~1.0 m/s) previously
# stalled the arm's no-timeout ATTACH phase indefinitely - not fully root-caused.
LOOP1_RUN_SPEED_PCT = 55

# Tuned so boxes land inside the truck bed instead of overshooting it (at full speed
# they cleared the truck's far wall and landed on the ground beyond it).
LOOP2_RUN_SPEED_PCT = 50

# Both loops already sit close enough for the UR20 (1.75m reach) at the Y midpoint
# without any runtime repositioning.
ROBOT_POSITION = (-3.0, 1.0928, 0.0)  # (x, y, z-of-ground-contact); Y = loop midpoint
PEDESTAL_HEIGHT = 1.6
PLACE_XY = (-3.0, 2.1857)  # ConveyorTrack_09's belt-top Y center

PICK_MAX_REACH_M = 1.75  # UR20 spec reach; pick targets farther than this are dropped

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
        # Belt mesh has near-zero Z extent; query a column above it instead so
        # boxes resting on top are actually caught by the occupancy check.
        belt_top_z = aligned_range.GetMax()[2]
        occupancy_query_half_height = 0.5  # meters; tallest known box is ~0.42m
        self.bbox_half_extent = [size[0] * 0.5, size[1] * 0.5, occupancy_query_half_height]
        self.bbox_center = [aligned_range.GetMidpoint()[0], aligned_range.GetMidpoint()[1], belt_top_z + occupancy_query_half_height]
        # Belts are treated as axis-aligned and static; identity rotation.
        self._quat = carb.Float4(0.0, 0.0, 0.0, 1.0)

        # inputs:velocity is wired via a ReadVariable node to this per-track OmniGraph
        # variable rather than holding a plain value; apply_command authors it directly.
        self.velocity_var_attr = self.node_prim.GetParent().GetAttribute("graph:variable:Velocity")
        if not self.velocity_var_attr:
            raise RuntimeError(f"{node_path}'s graph has no graph:variable:Velocity")

        # inputs:direction is a unit vector for straight zones, an angular-velocity axis for
        # curved ones. The baked values aren't reliable (wrong sign/magnitude in places); see
        # ConveyorLineController._fix_zone_directions, which rederives both from geometry.
        self.direction_attr = self.node_prim.GetAttribute("inputs:direction")
        baked_direction = self.direction_attr.Get()
        self.is_straight = baked_direction is not None and baked_direction[2] == 0.0

        # World-space travel direction for straight zones; set by _fix_zone_directions,
        # only meaningful there (used by is_past_center).
        self.world_travel_direction: Gf.Vec3f | None = None

        self.state_machine = ConveyorZoneStateMachine(name=node_path, run_speed_pct=run_speed_pct)
        self.state_machine.start()

    def get_occupying_prim_paths(self) -> list:
        """Return paths of every non-excluded rigid body overlapping this zone.

        Used for the boolean occupied check and, for the pick zone, to identify
        WHICH box actually triggered pick_ready rather than a hardcoded path.
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
        """True once world_position has passed stop_fraction of the way through this
        straight zone along world_travel_direction (0.5 = midpoint). Used by hold
        zones to settle an occupying part at a fixed, robot-reachable point.
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
            # Direction is baked in per-track; only magnitude is set here.
            self.velocity_var_attr.Set(ZONE_RUN_VELOCITY * speed_pct / 100.0)
        else:
            # enabled=False alone doesn't stop the belt (OgnIsaacConveyor leaves the
            # last nonzero surface velocity authored) - zero it directly instead.
            surface_velocity_api = PhysxSchema.PhysxSurfaceVelocityAPI.Apply(self.belt_prim)
            zero = Gf.Vec3f(0.0, 0.0, 0.0)
            surface_velocity_api.GetSurfaceVelocityAttr().Set(zero)
            surface_velocity_api.GetSurfaceAngularVelocityAttr().Set(zero)


class ConveyorLineController:
    """Owns one line's ordered zones, wires neighbor occupancy, applies commands.

    Supports both a closed loop (neighbor indices wrap) and an open line (first
    zone has no upstream, last has no downstream) via `closed_loop`.
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
        self._hold_zone_ready_checks: dict = {}  # zone_index -> ready_fn; absent = always-ready
        self._fix_zone_directions()

    def set_hold_zone_ready_check(self, zone_index: int, ready_fn) -> None:
        """While ready_fn() is False (that zone's robot is busy), hold zone zone_index
        behaves like an ordinary pass-through zone, so a box overflows to the next
        pick station instead of backing up behind a busy robot.
        """
        self._hold_zone_ready_checks[zone_index] = ready_fn

    def set_box_rigid_prims(self, box_rigid_prims: dict) -> None:
        """Inject the live RigidPrim for every known box (built in main() only after
        world.reset()) so hold zones can check is_past_center() against real position.
        """
        self._box_rigid_prims = box_rigid_prims

    @staticmethod
    def _is_body_flipped(belt_prim) -> bool:
        """True if belt_prim's world rotation is ~180deg about some axis (i.e. its
        body frame is flipped relative to world space).
        """
        world_rotation = UsdGeom.Xformable(belt_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractRotation()
        return abs(world_rotation.GetQuat().GetReal()) < 0.5

    def _fix_zone_directions(self) -> None:
        """Overwrite every zone's inputs:direction from actual belt geometry - the
        baked values aren't reliable (see ConveyorZone.__init__).

        Straight zones: unit vector from this zone's bbox_center toward the next
        zone's, snapped to the belt's long axis, authored as-is in world space (no
        per-body flip negation - confirmed empirically the wrong way round). Curved
        zones: angular-velocity magnitude rederived from the curve's own radius (the
        baked value is far too large for this scaffold's velocity scaling); sign
        rederived from whether the curve's entry-side neighbor is itself flipped
        (copying the baked sign directly is wrong at one end of each loop, since the
        two curves are mirror-image ends of the same track).
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
                # Last zone of an open line: extrapolate from the previous zone's
                # center through this one, rather than wrapping back to zone 0.
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

            # The authored inputs:direction (unlike world_travel_direction above) DOES need
            # a per-body sign flip here: every track is uniformly ~180deg-rotated about Z,
            # and the un-negated vector drove boxes the wrong way off the belt's open edge.
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
            # Open line: zone 0 has no real upstream, so it's always treated as having
            # more infeed available; the last zone's downstream (open end/truck) is
            # always clear, unless it's a held zone, which never reports clear.
            if i == 0 and not self.closed_loop:
                upstream_occupied = True
            else:
                upstream_occupied = self.occupied[(i - 1) % n]

            # See set_hold_zone_ready_check: a hold zone overflows only while its robot
            # is busy, and never if it's the line's last zone (nowhere to overflow to).
            is_last = i == n - 1 and not self.closed_loop
            robot_ready = i in self.hold_zone_indices and self._hold_zone_ready_checks.get(i, lambda: True)()
            holding = i in self.hold_zone_indices and (robot_ready or is_last)
            if holding:
                downstream_clear = False
            elif is_last:
                downstream_clear = True
            else:
                downstream_clear = not self.occupied[(i + 1) % n]

            # Any hold zone defines a stop position, whether or not it's currently
            # holding for its own robot - it should still index a box up to that point
            # (maximizing its own occupancy) even while overflowing because its robot is
            # busy or its downstream neighbor is full; only non-hold zones default to
            # True (stop-as-soon-as-occupied).
            at_stop_position = True
            if i in self.hold_zone_indices and self._box_rigid_prims is not None and self.occupied[i]:
                occupying_paths = zone.get_occupying_prim_paths()
                leading_path = _leading_occupant_path(zone, occupying_paths, self._box_rigid_prims)
                box_rigid_prim = self._box_rigid_prims.get(leading_path) if leading_path is not None else None
                if box_rigid_prim is not None:
                    position, _ = box_rigid_prim.get_world_poses()
                    # is_last has open ground past it, not another zone - tuned separately.
                    stop_fraction = 0.8 if is_last else 0.8
                    at_stop_position = zone.is_past_center(position.numpy()[0], stop_fraction=stop_fraction)

            observation, command = zone.state_machine.step(
                occupied=self.occupied[i],
                upstream_occupied=upstream_occupied,
                downstream_clear=downstream_clear,
                at_stop_position=at_stop_position,
            )
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
        """Coarse, approximate PackML projection - see README for real PackML nuance
        (Starting/Stopping/Held/Aborted) not modeled by this two-bucket mapping.
        """
        Machine = plc.ConveyorStateMachineCode
        running_states = {
            Machine.CONVEYOR_STATE_MACHINE_INDUCTING,
            Machine.CONVEYOR_STATE_MACHINE_ADVANCE_ITEM,
            Machine.CONVEYOR_STATE_MACHINE_PASSTHROUGH,
        }
        return common_types.PACKML_EXECUTE if machine in running_states else common_types.PACKML_IDLE


def _deactivate_frame_meshes(stage: Usd.Stage, track_roots: tuple) -> None:
    """Deactivate every track's frame/upright-posts mesh at runtime (not edited into
    the USD) - removes it from both rendering and cuMotion's obstacle tracking, so
    the arm no longer avoids the frame/posts, only the belt-top zone bboxes.
    """
    deactivated = []
    for root in track_roots:
        root_prim = stage.GetPrimAtPath(root)
        if not root_prim.IsValid():
            raise RuntimeError(f"Expected conveyor track prim not found at {root}")
        matched = False
        for prim in Usd.PrimRange(root_prim):
            if prim.GetName() == "SM_ConveyorBelt_A06_02":
                prim.SetActive(False)
                deactivated.append(str(prim.GetPath()))
                matched = True
        if not matched:
            raise RuntimeError(f"No SM_ConveyorBelt_A06_02 mesh found under: {root}")
    print(f"[conveyor_indexer] deactivated {len(deactivated)} SM_ConveyorBelt_A06_02 frame meshes", flush=True)


def _discover_box_prim_paths(stage: Usd.Stage) -> list:
    """Find every pre-authored CubeBox_* top-level prim (direct child of the stage
    root or /World) - excludes child meshes that share the naming prefix.
    """
    paths = []
    for prim in stage.Traverse():
        if not prim.GetName().startswith(BOX_PRIM_NAME_PREFIX):
            continue
        parent = prim.GetParent()
        if parent.IsPseudoRoot() or parent.GetPath() == Sdf.Path("/World"):
            paths.append(str(prim.GetPath()))
    return sorted(paths)


def _rank_pick_zone_hit_paths(
    hit_paths: list[str],
    box_rigid_prims: dict,
    robot_xy: tuple[float, float],
    max_reach_m: float = PICK_MAX_REACH_M,
) -> str | None:
    """Pick the closest reachable box to the robot (height as tie-breaker); drops
    anything farther than max_reach_m rather than ranking it.
    """
    def _distance(path: str) -> float:
        box_pos, _ = box_rigid_prims[path].get_world_poses()
        return float(math.dist(box_pos.numpy()[0][:2], robot_xy))

    reachable = [path for path in hit_paths if _distance(path) <= max_reach_m]
    if not reachable:
        return None

    def _score(path: str) -> tuple[float, float, str]:
        box_pos, _ = box_rigid_prims[path].get_world_poses()
        pos = box_pos.numpy()[0]
        return (_distance(path), -float(pos[2]), path)

    return min(reachable, key=_score)


def _leading_occupant_path(zone: "ConveyorZone", hit_paths: list[str], box_rigid_prims: dict) -> str | None:
    """Pick whichever occupying box is furthest downstream - hit_paths isn't in
    spatial order, so hit_paths[0] could be a trailing box instead.
    """
    if not hit_paths:
        return None

    travel = zone.world_travel_direction
    if travel is None:
        raise RuntimeError(f"{zone.node_path} has no world_travel_direction set")

    def _downstream(path: str) -> float:
        box_pos, _ = box_rigid_prims[path].get_world_poses()
        pos = box_pos.numpy()[0]
        return float(pos[0] * travel[0] + pos[1] * travel[1] + pos[2] * travel[2])

    return max(hit_paths, key=_downstream)


# Iteration cap for _resolve_box_overlaps's separation passes, and the extra
# gap (beyond just-touching) left between two boxes once separated - small
# enough to be visually unnoticeable, large enough that the two don't start
# back in (near-)contact and immediately re-trigger PhysX's own depenetration
# push once physics starts.
BOX_OVERLAP_RESOLVE_MAX_PASSES = 8
BOX_OVERLAP_CLEARANCE = 0.002  # meters


def _resolve_box_overlaps(stage: Usd.Stage, box_paths: list) -> None:
    """Nudge apart any box prims whose world AABBs actually overlap, before physics
    ever runs - the pre-authored pallet has some pairs uncomfortably tight (~2cm),
    which otherwise makes PhysX's depenetration fling boxes off the belt on tick one.
    Pushes along the smallest-overlap axis (a standard MTV separation), to a fixed pass cap.
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
    """Add RigidBodyAPI + convex-hull CollisionAPI + mass to every box prim (ships
    with no physics schemas at all). Convex-hull since these are dynamic bodies in
    contact with each other and the moving belt.
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


def _apply_truck_collision(stage: Usd.Stage, truck_path: str) -> None:
    """Add a static collider to the truck body mesh (a pure visual payload with no
    physics) so falling boxes land in the bed instead of clipping through.
    """
    body_prim = stage.GetPrimAtPath(f"{truck_path}/sm_steelboxtruck_a01_body_01")
    if not body_prim.IsValid():
        raise RuntimeError(f"Expected truck body mesh not found under {truck_path}")
    UsdPhysics.CollisionAPI.Apply(body_prim)
    print(f"[conveyor_indexer] added static collision to {body_prim.GetPath()}", flush=True)


def _truck_body_world_bounds(stage: Usd.Stage, truck_path: str) -> tuple[Gf.Vec3d, Gf.Vec3d]:
    """World-space AABB of the truck body mesh, computed once and reused every tick
    to test whether a box has landed inside it (see _despawn_boxes_in_truck).
    """
    body_prim = stage.GetPrimAtPath(f"{truck_path}/sm_steelboxtruck_a01_body_01")
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    aligned_range = bbox_cache.ComputeWorldBound(body_prim).ComputeAlignedRange()
    return aligned_range.GetMin(), aligned_range.GetMax()


# Parking spot for despawned boxes, far from any geometry so its AABB never
# matches truck_bed_min/max again.
DESPAWNED_BOX_PARK_POSITION = (100.0, 100.0, -100.0)


def _despawn_boxes_in_truck(
    box_rigid_prims: dict, truck_bed_min: Gf.Vec3d, truck_bed_max: Gf.Vec3d
) -> None:
    """Disable, hide, and park any box that's fallen inside the truck bed AABB.

    Deleting the prim outright crashes the app: RigidPrim's shared PhysX tensor
    view gets invalidated for every other tracked box too.
    """
    landed_paths = []
    for box_path, rigid_prim in box_rigid_prims.items():
        x, y, z = rigid_prim.get_world_poses()[0].numpy()[0]
        if truck_bed_min[0] <= x <= truck_bed_max[0] and truck_bed_min[1] <= y <= truck_bed_max[1] and z <= truck_bed_max[2]:
            landed_paths.append(box_path)
    for box_path in landed_paths:
        rigid_prim = box_rigid_prims[box_path]
        rigid_prim.set_enabled_rigid_bodies([False])
        rigid_prim.set_visibilities([False])
        rigid_prim.set_world_poses(positions=[DESPAWNED_BOX_PARK_POSITION])
        print(f"[conveyor_indexer] despawned {box_path} - landed in {TRUCK_PATH}", flush=True)


def _localize_asset_references(stage_path: str, remote_root: str, local_root: str) -> None:
    """Rewrite reference/payload asset paths from remote_root to local_root in
    stage_path's Sdf.Layer, before it's ever opened as a Usd.Stage (opening triggers
    composition, which is when Kit would fetch the un-rewritten paths over the
    network). Not saved to disk. No-op if local_root doesn't exist yet.
    """
    if not os.path.isdir(local_root):
        print(
            f"[conveyor_indexer] {local_root} not found - fetching assets from {remote_root} instead "
            "(see README's download_assets.py note to cache them locally)",
            flush=True,
        )
        return

    layer = Sdf.Layer.FindOrOpen(stage_path)
    if layer is None:
        raise RuntimeError(f"Could not open {stage_path} as an Sdf.Layer")

    def _iter_prim_specs(root_specs):
        stack = list(root_specs)
        while stack:
            spec = stack.pop()
            yield spec
            stack.extend(spec.nameChildren.values())

    def _localized(asset_path: str) -> str:
        return os.path.join(local_root, asset_path[len(remote_root):])

    rewritten = 0
    for spec in _iter_prim_specs(layer.rootPrims.values()):
        refs = spec.referenceList
        if refs.prependedItems:
            new_items = []
            for item in refs.prependedItems:
                if item.assetPath.startswith(remote_root):
                    item = Sdf.Reference(_localized(item.assetPath), item.primPath, item.layerOffset, item.customData)
                    rewritten += 1
                new_items.append(item)
            refs.prependedItems = new_items

        payloads = spec.payloadList
        if payloads.prependedItems:
            new_items = []
            for item in payloads.prependedItems:
                if item.assetPath.startswith(remote_root):
                    item = Sdf.Payload(_localized(item.assetPath), item.primPath, item.layerOffset)
                    rewritten += 1
                new_items.append(item)
            payloads.prependedItems = new_items

    print(f"[conveyor_indexer] localized {rewritten} asset reference(s)/payload(s) to {local_root}", flush=True)


# Only the Belt mesh (not the frame/posts mesh) fails cuMotion's obstacle scan;
# _build_rmpflow_controller's own retry loop already excludes exactly the failing
# prims one at a time, so no exclusion list is passed from here. Synthetic capsule
# obstacles were tried and rejected: CollisionAPI makes a real PhysX collider, not
# just a planning hint, so they physically shoved the real boxes on the belt.


def main() -> None:
    # isaacsim.asset.gen.conveyor isn't guaranteed enabled by the resolved app config;
    # enable it explicitly before the stage (and its ConveyorBeltGraphs) is opened.
    omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate(
        "isaacsim.asset.gen.conveyor", True
    )

    # Open the stage before constructing World: World() attaches at construction time,
    # and opening after leaves it referencing a stale, now-gone stage.
    _localize_asset_references(STAGE_PATH, REMOTE_ASSET_ROOT, LOCAL_ASSET_ROOT)
    print(f"[conveyor_indexer] opening stage {STAGE_PATH}", flush=True)
    ctx = omni.usd.get_context()
    ctx.open_stage(STAGE_PATH)
    stage = ctx.get_stage()
    _deactivate_frame_meshes(stage, CONVEYOR_TRACK_ROOTS)
    _apply_truck_collision(stage, TRUCK_PATH)
    truck_bed_min, truck_bed_max = _truck_body_world_bounds(stage, TRUCK_PATH)
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

    # Pre-create the log directory - on a clean checkout, ConveyorIndexingLogger's
    # exists-at-call-time heuristic otherwise mistakes it for a file stem.
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

    # Reach-balanced position for the second robot, derived from actual zone geometry
    # rather than hardcoded - ConveyorTrack_02/_10 aren't guaranteed to line up in X.
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

    # Place target Z depends on box height, computed per-cycle inside MagicAttachPickPlace;
    # only the belt-top Z is fixed here.
    place_belt_prim = loop2.zones[PLACE_ZONE_INDEX].belt_prim
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    place_belt_top_z = bbox_cache.ComputeWorldBound(place_belt_prim).ComputeAlignedRange().GetMax()[2]
    place_belt_top_z_2 = bbox_cache.ComputeWorldBound(place_zone_2.belt_prim).ComputeAlignedRange().GetMax()[2]

    box_rigid_prims = {path: RigidPrim(path) for path in box_paths}
    # Only loop1 has hold zones needing is_past_center() checks against a real box position.
    loop1.set_box_rigid_prims(box_rigid_prims)
    print("[conveyor_indexer] robots ready, calling world.reset()", flush=True)

    world.reset()
    # World.reset() has no knowledge of isaacsim.core.experimental prims like this
    # robot and never calls reset_to_default_state() on them - must be triggered explicitly.
    robot.reset_to_default_state()
    robot2.reset_to_default_state()
    check_pos, _ = robot.get_world_poses()
    print(f"[conveyor_indexer] DEBUG robot base pose AFTER world.reset(): {check_pos.numpy()}", flush=True)
    print(f"[conveyor_indexer] DEBUG robot dof_positions AFTER reset_to_default_state(): {robot.get_dof_positions().numpy()}", flush=True)
    print("[conveyor_indexer] world.reset() done", flush=True)

    # MagicAttachPickPlace builds the cuMotion RmpFlowController, which needs a valid
    # PhysX tensor entity - must happen after world.reset().
    # pre_place_joint_positions override: robot 2 sits on this robot's -X side, so its
    # pick<->place swing must arc away from it - see UR20_PRE_PLACE_JOINT_POSITIONS_AWAY.
    pick_place = MagicAttachPickPlace(
        robot=robot,
        robot_path=ROBOT_PATH,
        place_xy=PLACE_XY,
        place_belt_top_z=place_belt_top_z,
        box_rigid_prims=box_rigid_prims,
        physics_dt=world.get_physics_dt(),
        get_pick_zone_occupant_paths=loop1.zones[PICK_ZONE_INDEX].get_occupying_prim_paths,
        extra_exclude_obstacle_paths=[GROUND_PLANE_COLLISION_PATH],
        pre_place_joint_positions=UR20_PRE_PLACE_JOINT_POSITIONS_AWAY,
    )
    pick_place_2 = MagicAttachPickPlace(
        robot=robot2,
        robot_path=ROBOT_PATH_2,
        place_xy=PLACE_XY_2,
        place_belt_top_z=place_belt_top_z_2,
        box_rigid_prims=box_rigid_prims,
        physics_dt=world.get_physics_dt(),
        get_pick_zone_occupant_paths=loop1.zones[PICK_ZONE_INDEX_2].get_occupying_prim_paths,
        extra_exclude_obstacle_paths=[GROUND_PLANE_COLLISION_PATH],
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

    # SIGTERM otherwise kills the process immediately, skipping the finally block
    # below and leaving the parquet writer's file truncated - route it through the
    # normal loop-exit path instead, same as SIGINT.
    shutdown_requested = False

    def _handle_sigterm(signum, frame):
        nonlocal shutdown_requested
        shutdown_requested = True

    signal.signal(signal.SIGTERM, _handle_sigterm)

    Machine = plc.ConveyorStateMachineCode

    try:
        while simulation_app.is_running() and not shutdown_requested:
            if world.is_playing():
                world.step(render=True); sim_time += world.get_physics_dt()

                # Pick-and-place runs every physics step for smooth convergence; conveyor
                # indexing runs at the coarser control rate below.
                pick_place.forward(pick_ready, pick_box_path)
                pick_place_2.forward(pick_ready_2, pick_box_path_2)

                if sim_time - last_control_time >= control_period_s:
                    state_msg = plc.StateConveyors()
                    commands_msg = sim_action.SimConveyorCommands()
                    loop1.step(state_msg, commands_msg)
                    loop2.step(state_msg, commands_msg)
                    _despawn_boxes_in_truck(box_rigid_prims, truck_bed_min, truck_bed_max)
                    logger.log_tick(
                        tick=tick,
                        sim_time_s=sim_time,
                        plc_state_conveyors=state_msg.SerializeToString(),
                        conveyor_commands=commands_msg.SerializeToString(),
                    )
                    # Only "ready" once the pick zone has settled into holding (IDLE +
                    # occupied); identify which box is actually there rather than assuming a fixed one.
                    pick_zone_hits = loop1.zones[PICK_ZONE_INDEX].get_occupying_prim_paths()
                    pick_ready = (
                        bool(pick_zone_hits)
                        and loop1.machine_states[PICK_ZONE_INDEX] == Machine.CONVEYOR_STATE_MACHINE_IDLE
                    )
                    pick_box_path = _rank_pick_zone_hit_paths(
                        pick_zone_hits, box_rigid_prims, ROBOT_POSITION[:2]
                    )
                    pick_zone_2_hits = loop1.zones[PICK_ZONE_INDEX_2].get_occupying_prim_paths()
                    pick_ready_2 = (
                        bool(pick_zone_2_hits)
                        and loop1.machine_states[PICK_ZONE_INDEX_2] == Machine.CONVEYOR_STATE_MACHINE_IDLE
                    )
                    pick_box_path_2 = _rank_pick_zone_hit_paths(
                        pick_zone_2_hits, box_rigid_prims, ROBOT_POSITION_2[:2]
                    )
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
        logger.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
