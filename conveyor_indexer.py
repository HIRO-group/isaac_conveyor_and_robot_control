"""Standalone Isaac Sim script: zone-accumulation indexing over the inbuilt
surface-velocity conveyors in ~/conveyor_setup.usd, with per-tick data
logging for later imitation/RL training, plus a UR10 pick-and-place between
the two conveyor loops.

Run with Isaac Sim's bundled python, e.g.:
    ./python.sh /home/ubuntu/conveyor_indexing/conveyor_indexer.py

See README.md in this directory for setup (protobuf codegen), and everything
this scaffold does NOT implement yet.
"""

from __future__ import annotations

import pathlib
import signal
import sys

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import omni.kit.app

# isaacsim.robot.experimental.manipulators.examples (UR10/Franka + friends) is
# not enabled by default - must be turned on before importing from it (APIs
# come from the extension/runtime plugin system, loaded lazily).
omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate(
    "isaacsim.robot.experimental.manipulators.examples", True
)

import carb
import omni.usd
from omni.physics.core import get_physics_scene_query_interface
from pxr import Gf, PhysicsSchemaTools, PhysxSchema, Usd, UsdGeom, UsdPhysics
from isaacsim.core.api import World
from isaacsim.core.experimental.prims import RigidPrim

sys.path.insert(0, "/home/ubuntu/conveyor_indexing")
sys.path.insert(0, "/tmp/proto_gen")  # see README.md: gen_proto.sh output

import plc_connector_pb2 as plc
import sim_conveyor_action_pb2 as sim_action

try:
    from common import types_pb2 as common_types
except ModuleNotFoundError:
    import types_pb2 as common_types

from conveyor_state_machine import ConveyorZoneStateMachine
from conveyor_indexing_logger import ConveyorIndexingLogger
from pick_and_place import create_pedestal_and_robot, MagicAttachPickPlace

STAGE_PATH = "/home/ubuntu/conveyor_setup.usd"
LOG_OUTPUT_DIR = "/home/ubuntu/conveyor_indexing/data"
CONTROL_HZ = 30.0
DEBUG_LOG_OCCUPANCY_HITS = False  # set True to print which prim triggers each occupancy hit

# ---------------------------------------------------------------------------
# Zone tables - two independent closed loops (see README: confirmed via
# world-space translate/anchor chaining, same as loop 1). Loop 2
# (ConveyorTrack_08.._15) was added after loop 1 and sits offset +Y.
#
# Also excluded from both: a stray `/World/ConveyorBeltGraph/ConveyorNode` at
# the stage root, whose `inputs:conveyorPrim` targets `/World/DistantLight`
# instead of a belt - looks like a leftover/misconfigured graph in
# conveyor_setup.usd, not a real zone.
# ---------------------------------------------------------------------------
ZONE_NODE_PATHS_LOOP1 = [
    "/World/ConveyorTrack/ConveyorBeltGraph/ConveyorNode",
    "/World/ConveyorTrack_01/ConveyorBeltGraph/ConveyorNode",
    "/World/ConveyorTrack_02/ConveyorBeltGraph/ConveyorNode",
    "/World/ConveyorTrack_03/ConveyorBeltGraph/ConveyorNode",
    "/World/ConveyorTrack_04/ConveyorBeltGraph/ConveyorNode",
    "/World/ConveyorTrack_05/ConveyorBeltGraph/ConveyorNode",
    "/World/ConveyorTrack_06/ConveyorBeltGraph/ConveyorNode",
    "/World/ConveyorTrack_07/ConveyorBeltGraph/ConveyorNode",
]
ZONE_NODE_PATHS_LOOP2 = [
    "/World/ConveyorTrack_08/ConveyorBeltGraph/ConveyorNode",
    "/World/ConveyorTrack_09/ConveyorBeltGraph/ConveyorNode",
    "/World/ConveyorTrack_10/ConveyorBeltGraph/ConveyorNode",
    "/World/ConveyorTrack_11/ConveyorBeltGraph/ConveyorNode",
    "/World/ConveyorTrack_12/ConveyorBeltGraph/ConveyorNode",
    "/World/ConveyorTrack_13/ConveyorBeltGraph/ConveyorNode",
    "/World/ConveyorTrack_14/ConveyorBeltGraph/ConveyorNode",
    "/World/ConveyorTrack_15/ConveyorBeltGraph/ConveyorNode",
]

# ConveyorTrack_01 (loop 1) and ConveyorTrack_09 (loop 2) directly face each
# other across the gap (both span local X -2..-4) - the natural spot for a
# fixed pick/place robot between the two loops.
PICK_ZONE_INDEX = 1  # ConveyorTrack_01 within ZONE_NODE_PATHS_LOOP1
PLACE_ZONE_INDEX = 1  # ConveyorTrack_09 within ZONE_NODE_PATHS_LOOP2

ROBOT_PATH = "/World/PickPlaceRobot"
PEDESTAL_PATH = "/World/PickPlacePedestal"

# Every physics-enabled box that could appear in the pick zone (see README -
# discovered via occupancy-hit debug logging, not obvious from a stage
# traversal since the cardboard box's RigidBodyAPI is on a descendant, not
# its own top-level prim). RigidPrim wrappers for these are pre-built once
# at startup and reused (see MagicAttachPickPlace's box_rigid_prims) rather
# than constructed fresh mid-simulation.
KNOWN_BOX_PATHS = [
    "/World/sm_box_multiDepth_brown_b08_01",
    "/World/sm_box_cardboard_a02_01/Geometry/sm_box_a02_obj_00",
]

# Loop 2 as originally authored sits with its near edge 1.75 m from loop 1's
# (2.6 m centerline-to-centerline) - too far for a UR10 (~1.3 m reach) to
# reliably bridge: a first pass at this (ROBOT_POSITION Y=1.22, PLACE_XY
# Y=2.4, ~90% of spec reach each side) picked reliably but consistently
# failed to reach the place point, dropping the box back onto loop 1 instead
# (confirmed via 55s of logged data: the place zone never once registered
# occupied while loop 1 cycled the same box perpetually). Loop 2 is
# repositioned closer at runtime (LOOP2_Y_SHIFT, applied in main() before
# World construction) rather than edited into conveyor_setup.usd, so the
# authored scene file stays untouched.
LOOP2_Y_SHIFT = -0.81  # applied to every ConveyorTrack_08.._15 prim's Y translate

# With the shift applied, loop 2's centerline moves from Y=2.645 to ~1.835 and
# its near edge from Y=2.195 to ~1.385. Pick point is ~Y=0.04 (where boxes
# naturally rest on loop 1); place point below is safely inboard of the new
# near edge. Reach needed each side from the balanced midpoint is now ~0.8 m
# - ~62% of the UR10's 1.3 m spec, comfortable margin below the ~90% that
# failed above.
ROBOT_POSITION = (-3.0, 0.84, 0.0)  # (x, y, z-of-ground-contact)
PEDESTAL_HEIGHT = 1.6
PLACE_XY = (-3.0, 1.64)  # on ConveyorTrack_09's belt (post-shift), safely inboard of its near edge

# Any occupancy hit whose prim path falls under one of these roots is belt/
# structure/robot geometry, not a transported item, and is excluded from
# occupancy detection.
EXCLUDED_STRUCTURE_ROOTS = tuple(
    f"/World/ConveyorTrack{suffix}"
    for suffix in ["", "_01", "_02", "_03", "_04", "_05", "_06", "_07", "_08", "_09", "_10", "_11", "_12", "_13", "_14", "_15"]
) + (ROBOT_PATH, PEDESTAL_PATH)


class ConveyorZone:
    """Bridges one ConveyorZoneStateMachine to its USD ConveyorNode + belt bbox."""

    def __init__(self, index: int, node_path: str, stage: Usd.Stage) -> None:
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
        self._half_extent = [d * 0.5 for d in (aligned_range.GetSize())]
        self._center = list(aligned_range.GetMidpoint())
        # Belts are treated as axis-aligned and static; identity rotation.
        self._quat = carb.Float4(0.0, 0.0, 0.0, 1.0)

        self.state_machine = ConveyorZoneStateMachine(name=node_path)
        self.state_machine.start()

    def get_occupying_prim_paths(self) -> list:
        """Return the paths of every non-excluded rigid body overlapping this zone.

        Used both for the boolean occupied check and, for the pick zone
        specifically, to identify WHICH box is actually present - there are
        two real physics-enabled boxes in the scene (see README), and picking
        must track whichever one actually triggered `pick_ready`, not a
        hardcoded path (a fixed path silently tracks the wrong box, and the
        wrong box, whenever it happens to be, the moment it's "detached" ends
        up wherever the arm's current position, unrelated to the belt).
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
            carb.Float3(*self._half_extent),
            carb.Float3(*self._center),
            self._quat,
            report_hit,
        )
        return hits

    def check_occupied(self) -> bool:
        return len(self.get_occupying_prim_paths()) > 0

    def apply_command(self, run: bool) -> None:
        self.node_prim.GetAttribute("inputs:enabled").Set(run)
        if not run:
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
    """Owns one closed loop's ordered zones, wires neighbor occupancy, applies commands."""

    def __init__(
        self,
        stage: Usd.Stage,
        node_paths: list,
        hold_zone_indices: frozenset = frozenset(),
    ) -> None:
        self.zones = [ConveyorZone(i, path, stage) for i, path in enumerate(node_paths)]
        self.hold_zone_indices = hold_zone_indices
        self.occupied: list = [False] * len(self.zones)
        self.machine_states: list = [None] * len(self.zones)

    def step(self, state_msg, commands_msg) -> None:
        """Advance every zone by one control tick, appending into shared log messages."""
        self.occupied = [zone.check_occupied() for zone in self.zones]

        n = len(self.zones)
        for i, zone in enumerate(self.zones):
            # Closed loop: neighbors wrap around rather than terminating at
            # open ends (see README). A held zone (e.g. the pick zone) never
            # reports downstream_clear, so it holds an arriving item
            # indefinitely instead of auto-advancing it further - "starved"
            # again only once whatever emptied it (the robot) lets it go.
            upstream_occupied = self.occupied[(i - 1) % n]
            if i in self.hold_zone_indices:
                downstream_clear = False
            else:
                downstream_clear = not self.occupied[(i + 1) % n]

            observation, command = zone.state_machine.step(
                occupied=self.occupied[i],
                upstream_occupied=upstream_occupied,
                downstream_clear=downstream_clear,
            )
            zone.apply_command(command.run)
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


def _neutralize_stray_distant_light_rigid_body(stage: Usd.Stage) -> None:
    """Undo conveyor_setup.usd's stray ConveyorNode side effect on the DistantLight.

    The top-level `/World/ConveyorBeltGraph/ConveyorNode`'s `inputs:conveyorPrim`
    targets `/World/DistantLight` instead of a real belt - looks like a
    leftover/misconfigured graph from authoring. `create_conveyor_belt()`
    walks up looking for a RigidBodyAPI ancestor and, finding none, applies
    RigidBodyAPI + CollisionAPI + PhysxSurfaceVelocityAPI directly to
    whatever prim it's given - here, that's the light. The result is an
    uncontrolled dynamic rigid body with a degenerate "small sphere
    approximated" inertia tensor (see the runtime warning), which was
    observed tumbling through the scene and sporadically clipping zone
    bounding boxes - causing spurious, rapidly-flickering occupancy readings
    on unrelated zones. Disable its rigid body simulation so it stops
    interfering; leave the schema itself in place (less invasive than
    removing it).
    """
    light_prim = stage.GetPrimAtPath("/World/DistantLight")
    if light_prim.IsValid() and light_prim.HasAPI(UsdPhysics.RigidBodyAPI):
        UsdPhysics.RigidBodyAPI(light_prim).GetRigidBodyEnabledAttr().Set(False)
        print("[conveyor_indexer] disabled stray RigidBodyAPI on /World/DistantLight", flush=True)

    stray_node = stage.GetPrimAtPath("/World/ConveyorBeltGraph/ConveyorNode")
    if stray_node.IsValid():
        stray_node.GetAttribute("inputs:enabled").Set(False)


def _reposition_loop2(stage: Usd.Stage, delta_y: float) -> None:
    """Shift every loop-2 track prim by delta_y (see LOOP2_Y_SHIFT).

    Applied at runtime rather than edited into conveyor_setup.usd, so the
    authored scene file is left untouched; re-derived from each track's
    current translate rather than hardcoding absolute positions, so this
    stays correct regardless of how loop 2 happens to be authored.
    """
    loop2_track_paths = [f"/World/ConveyorTrack_{i:02d}" for i in range(8, 16)]
    for path in loop2_track_paths:
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            raise RuntimeError(f"Expected loop-2 track prim not found at {path}")
        xformable = UsdGeom.Xformable(prim)
        translate_op = None
        for op in xformable.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                translate_op = op
                break
        if translate_op is None:
            raise RuntimeError(f"{path} has no xformOp:translate to shift")
        current = translate_op.Get()
        translate_op.Set(type(current)(current[0], current[1] + delta_y, current[2]))
    print(f"[conveyor_indexer] shifted loop 2 by dY={delta_y} (runtime only, not saved to USD)", flush=True)


def main() -> None:
    # Open the target stage BEFORE constructing World: World() attaches to
    # whatever stage is open at construction time, and open_stage() after the
    # fact replaces the stage out from under it (World._scene ends up
    # referencing a physics_sim_view tied to the old, now-gone stage -
    # observed as "AttributeError: 'World' object has no attribute '_scene'"
    # inside world.reset() when this ordering was wrong).
    print(f"[conveyor_indexer] opening stage {STAGE_PATH}", flush=True)
    ctx = omni.usd.get_context()
    ctx.open_stage(STAGE_PATH)
    stage = ctx.get_stage()
    _neutralize_stray_distant_light_rigid_body(stage)
    _reposition_loop2(stage, LOOP2_Y_SHIFT)
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

    loop1 = ConveyorLineController(stage, ZONE_NODE_PATHS_LOOP1, hold_zone_indices=frozenset({PICK_ZONE_INDEX}))
    loop2 = ConveyorLineController(stage, ZONE_NODE_PATHS_LOOP2)
    print("[conveyor_indexer] zones built, creating pedestal + robot", flush=True)

    robot = create_pedestal_and_robot(
        stage,
        robot_path=ROBOT_PATH,
        pedestal_path=PEDESTAL_PATH,
        position=ROBOT_POSITION,
        pedestal_height=PEDESTAL_HEIGHT,
    )

    # Place target Z depends on box height, computed per-cycle inside
    # MagicAttachPickPlace since either of the two real boxes (see README)
    # could be the one being carried; only the belt-top Z is fixed here.
    place_belt_prim = stage.GetPrimAtPath("/World/ConveyorTrack_09/Belt")
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    place_belt_top_z = bbox_cache.ComputeWorldBound(place_belt_prim).ComputeAlignedRange().GetMax()[2]

    box_rigid_prims = {path: RigidPrim(path) for path in KNOWN_BOX_PATHS}
    pick_place = MagicAttachPickPlace(
        robot=robot, place_xy=PLACE_XY, place_belt_top_z=place_belt_top_z, box_rigid_prims=box_rigid_prims
    )
    print("[conveyor_indexer] robot + pick/place controller ready, calling world.reset()", flush=True)

    world.reset()
    check_pos, _ = robot.get_world_poses()
    check_ee_pos, _ = robot.end_effector_link.get_world_poses()
    print(
        f"[conveyor_indexer] DEBUG robot base pose AFTER world.reset(): {check_pos.numpy()} "
        f"ee_link pose AFTER world.reset(): {check_ee_pos.numpy()}",
        flush=True,
    )
    print("[conveyor_indexer] world.reset() done, entering main loop", flush=True)

    control_period_s = 1.0 / CONTROL_HZ
    last_control_time = 0.0
    sim_time = 0.0
    render_count = 0
    tick = 0
    pick_ready = False
    pick_box_path = None

    # SIGTERM (container/systemd shutdown, `kill` without -INT) otherwise
    # terminates the process immediately, bypassing the `finally` block below
    # and the parquet writer never gets a clean close() - the in-progress
    # file is left truncated/unreadable (missing footer). Route it through
    # the normal loop-exit -> finally path instead, same as SIGINT/window
    # close already do.
    shutdown_requested = False

    def _handle_sigterm(signum, frame):
        nonlocal shutdown_requested
        shutdown_requested = True

    signal.signal(signal.SIGTERM, _handle_sigterm)

    Machine = plc.ConveyorStateMachineCode

    try:
        while simulation_app.is_running() and not shutdown_requested:
            if world.is_playing():
                world.step(render=True)
                sim_time += world.get_physics_dt()

                # Pick-and-place motion runs every physics step for smooth IK
                # convergence; conveyor indexing runs at the coarser control
                # rate. `pick_ready`/`pick_box_path` only refresh at the
                # control rate but are read every physics step - fine, since
                # they only matter at the (infrequent) moment WAITING checks
                # them.
                pick_place.forward(pick_ready, pick_box_path)

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
                    # Identify WHICH box is actually there (two real boxes can
                    # end up in this zone - see README) rather than assuming a
                    # fixed one.
                    pick_zone_hits = loop1.zones[PICK_ZONE_INDEX].get_occupying_prim_paths()
                    pick_ready = (
                        bool(pick_zone_hits)
                        and loop1.machine_states[PICK_ZONE_INDEX] == Machine.CONVEYOR_STATE_MACHINE_IDLE
                    )
                    pick_box_path = pick_zone_hits[0] if pick_zone_hits else None
                    last_control_time = sim_time
                    tick += 1
                    if tick % 30 == 0:
                        print(
                            f"[conveyor_indexer] tick={tick} sim_time={sim_time:.2f} "
                            f"pick_phase={pick_place.phase_name} pick_ready={pick_ready}",
                            flush=True,
                        )
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
