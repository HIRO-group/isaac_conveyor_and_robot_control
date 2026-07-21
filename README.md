# Conveyor indexing (sim)

Zone-accumulation indexing controller for the inbuilt surface-velocity
conveyors in `~/conveyor_setup.usd` (two closed loops), with per-tick data
logging designed for later imitation learning and reinforcement learning on
the indexing policy, plus a UR10 that picks a box off loop 1 and places it on
loop 2 using magic attach (no real grasp physics).

## Design

- **Two independent closed loops**: `ConveyorTrack`.._07` (loop 1) and
  `ConveyorTrack_08`.._15` (loop 2, added later, offset +Y). Each is driven by
  its own `ConveyorLineController` with its own modulo-wrapped neighbor
  wiring - they don't interact except via the robot.
- **Actuation**: each generated belt segment has its own tiny ActionGraph
  (`OnTick -> ConveyorNode`, see `create_conveyor_belt()` in
  `isaacsim.asset.gen.conveyor`). `ConveyorNode.inputs:enabled` is a plain,
  unconnected bool, so this scaffold controls each zone purely by toggling it
  (Run / stop at whatever speed was authored). It does **not** currently
  change the belt's Velocity graph variable at runtime - see "Known gaps"
  below.
- **Occupancy sensing**: a PhysX box-overlap query (`overlap_box`) against
  each zone's belt bounding box, computed once at startup (belts are static).
  Hits whose rigid body path falls under a known `/World/ConveyorTrack*`,
  robot, or pedestal root are structure, not items, and are excluded.
- **Indexing logic**: `conveyor_state_machine.py` implements a best-effort
  "happy path" subset of the real `ConveyorStateMachineCode` enum from
  `~/theia/proto/plc-connector/plc-connector.proto` - see that module's
  docstring for the exact state cycle and what's deliberately not
  implemented.
- **Pick-and-place**: `pick_and_place.py`'s `MagicAttachPickPlace` runs a UR10
  (on a static pedestal at the reach-balanced midpoint between the two loops)
  through a fixed-duration phase state machine: approach, descend, "attach"
  (disable the box's rigid body and rigidly offset-follow the end effector -
  privileged pose queries only, no perception), lift, traverse, descend,
  detach, retract. `ConveyorTrack_01` (loop 1) is configured as the **hold
  zone**: `ConveyorLineController` forces its `downstream_clear` to always
  False, so an arriving box is held stopped in front of the robot instead of
  auto-advancing, and only "starves" (lets the next box advance in) once the
  robot's magic-attach has moved the box away and occupancy clears. See
  `PICK_ZONE_INDEX`/`hold_zone_indices` in `conveyor_indexer.py`.
- **Data schema**: chosen to match theia's real production schema so sim data
  is directly comparable/mergeable with real collected data, without
  depending on a live Zenoh session (see "Integration depth" below).
  - State: `plc_connector_pb2.StateConveyors` (one
    `StateConveyors_ConveyorsItem` per zone) - the exact message theia's real
    PLC connector publishes on `theia/plc/v1/state/Conveyors` and that
    `data_collection_vol2.py` already logs as its `plc_state_conveyors`
    column.
  - Action: `sim_conveyor_action_pb2.SimConveyorCommands` (see
    `proto/sim_conveyor_action.proto`) - a new, sim-only message mirroring
    the shape of theia's real per-conveyor `CmdConveyorsNRun/Speed/Direction`
    commands, collapsed into one repeated field for convenience. This does
    NOT modify theia's production proto files.
  - Both are written as binary columns per tick via
    `conveyor_indexing_logger.py`, following the same background-batched
    `ParquetWriter` pattern as `data_collection_vol2.py`.

## Setup

1. Generate the protobuf Python bindings (`protoc` and the `protobuf` Python
   package must be available - neither was present on this machine as of
   this writing; install `protoc` first, e.g. via your package manager):
   ```bash
   bash /home/ubuntu/conveyor_indexing/gen_proto.sh
   ```
2. Run the indexer with Isaac Sim's bundled python, with the generated
   bindings on `PYTHONPATH`:
   ```bash
   PYTHONPATH=/tmp/proto_gen /home/ubuntu/IsaacSim/python.sh /home/ubuntu/conveyor_indexing/conveyor_indexer.py
   ```
   (`conveyor_indexer.py` also inserts `/tmp/proto_gen` onto `sys.path`
   itself, but setting `PYTHONPATH` is more robust if you move things
   around.)

Note: launching a second Isaac Sim instance while another one is already
running interactively (e.g. the `cb_app.py` conveyor-belt Warp sample) will
contend heavily for this machine's single GPU and can make startup very
slow. Close other Isaac Sim instances first if this one seems to hang during
extension loading.

## Known gaps / TODOs

- ~~Zone order unconfirmed~~ **Resolved**: confirmed via world-space
  translate/anchor positions that `ZONE_NODE_PATHS`'s stage-traversal order
  (`ConveyorTrack`, `ConveyorTrack_01`, ... `_07`) matches physical adjacency,
  AND that it's a closed loop (`_07`'s anchor lands back on `ConveyorTrack`'s
  translate). `ConveyorLineController.step()` wraps neighbor indices with
  modulo accordingly.
- A stray `/World/ConveyorBeltGraph/ConveyorNode` exists at the stage root in
  `conveyor_setup.usd` whose `inputs:conveyorPrim` targets `/World/DistantLight`
  instead of a belt. Looks like a leftover/misconfigured graph - it's
  excluded from `ZONE_NODE_PATHS`, but worth cleaning up in the source scene.
- **Reject/fault states are not implemented.** `~/theia/docs/PLC/UDT.md`
  doesn't document the `Machine` field at all, and marks `Conveyor_Fault` as
  "pending definition" even though the `.proto` lists 9 fault codes. The
  real transition logic for `WAITING_TO_REJECT` / `REJECT_SINGLE` /
  `REJECT_STUCK` / `REJECT_FULL` / `REJECT_SPUR` / `SHIFTED_ITEM` /
  `PLACE_UNEXPECTED_ITEM` / `PURGE` lives in the physical PLC's ladder logic,
  not in this repo. `ConveyorZoneStateMachine._handle_exception_states()` is
  an explicit, never-called stub for this - wire it up once the real logic
  is available, rather than guessing.
- **Speed/direction are not actuated at runtime.** The logged `speed`/
  `direction` fields describe the nominal commanded behavior (state-machine
  intent). Changing speed at runtime would mean writing to each belt's own
  OmniGraph "Velocity" variable via `omni.graph.core` rather than the node's
  `inputs:velocity` attribute directly (that attribute is fed by a
  `read_speed` node and would just get overwritten next tick).
  ~~Only `enabled` (run/stop) is actually written back to the sim~~
  **Resolved**: toggling `inputs:enabled` alone does NOT stop a belt.
  `OgnIsaacConveyor.cpp`'s `compute()` early-returns unconditionally when
  `enabled=false` without ever touching the belt's `PhysxSurfaceVelocityAPI`
  attributes, so the last nonzero surface velocity stays authored and PhysX
  keeps driving the belt regardless of `enabled` - this was why held items
  kept coasting all the way around the loop instead of stopping.
  `ConveyorZone.apply_command()` now also zeros
  `physxSurfaceVelocity:surfaceVelocity`/`surfaceAngularVelocity` directly
  whenever commanding a stop; re-enabling needs no corresponding restore,
  since the node's own `hasVelocityChanged` check (authored velocity 0 vs.
  its `inputs:velocity`-derived target) rewrites the correct nonzero
  velocity on the next tick where `enabled=true` again.
- **`Fault` is always `CONVEYOR_FAULT_UNSPECIFIED`.** No jam/stall detection
  is implemented (e.g. a zone stuck `INDUCTING` for too long without becoming
  occupied). Possible future extension, not built here since it wasn't part
  of the current scope.
- **`Conveyor_Type` is hardcoded to `BUFFER`** for every zone in
  `conveyor_indexer.py`. Set real per-zone types once you know which belts
  are functioning as pick/place/buffer.
- **PackML mapping is a coarse two-bucket approximation**
  (`_machine_to_packml` in `conveyor_indexer.py`): EXECUTE while
  running, IDLE otherwise. Real PackML reporting almost certainly has more
  nuance (Starting/Stopping/Held/Aborted transitions).
- ~~Items may not yet be populated on the belts~~ **Resolved**: there are two
  real physics-enabled boxes. `sm_box_multiDepth_brown_b08_01` has
  `RigidBodyAPI` on its own top-level prim. `sm_box_cardboard_a02_01` does
  **not** have `RigidBodyAPI` on its top-level prim, but its descendant
  `sm_box_cardboard_a02_01/Geometry/sm_box_a02_obj_00` does - easy to miss by
  checking only the top prim (which is what an earlier pass here did). Both
  drive real `Machine` state transitions in the logged data, and having two
  boxes is what actually exercises the hold-zone/starvation behavior across
  more than one cycle in a single run.
- ~~Stray `/World/ConveyorBeltGraph/ConveyorNode` targets `/World/DistantLight`~~
  **Worked around at runtime, not fixed in the scene file**:
  `create_conveyor_belt()` walks up looking for a `RigidBodyAPI` ancestor and,
  finding none for this stray graph, applies `RigidBodyAPI` + `CollisionAPI` +
  `PhysxSurfaceVelocityAPI` directly to `/World/DistantLight` - turning the
  light into an uncontrolled dynamic rigid body (a "small sphere approximated"
  inertia tensor warning at startup is the tell). `_neutralize_stray_distant_light_rigid_body()`
  in `conveyor_indexer.py` disables its `RigidBodyAPI` each run. Worth
  actually deleting the stray graph from `conveyor_setup.usd` directly at some
  point instead of working around it in every script that opens the stage.
- **Pick-and-place phase timing is fixed-duration, not tolerance-based** (see
  `PHASE_TICKS` in `pick_and_place.py`, same pattern as this codebase's own
  `FrankaPickPlace`). If the IK hasn't converged by the time a phase's tick
  budget elapses, the next phase starts anyway from wherever the arm actually
  is. Worked fine in testing at the current pick/place distances but has no
  margin built in for e.g. a heavier box changing settle time.
- **Reach margin is real but not generous.** The UR10's ~1.3 m spec reach vs.
  ~1.2 m needed to each side (see conversation notes / `ROBOT_POSITION`,
  `PLACE_XY`) leaves less headroom than ideal, particularly near full
  extension where IK conditioning gets worse. Worked in testing; if you change
  `PLACE_XY` to sit closer to the belt centerline (more "natural" placement)
  or move the pedestal, re-check the reach math first.
- **No obstacle avoidance.** The differential-IK approach (`set_end_effector_pose`)
  has no notion of the conveyor structure, guard rails, or the pedestal itself
  as obstacles - waypoints were chosen to clear them by construction (approach
  height, etc.), not verified by collision checking.
- **Single-box capacity per zone/queue depth untested beyond two boxes** -
  behavior with a longer queue of boxes on loop 1, or a busy loop 2, hasn't
  been exercised.

## Files

| File | Purpose |
|---|---|
| `conveyor_indexer.py` | Standalone Isaac Sim entry point - occupancy sensing, zone wiring (both loops), pick/place wiring, per-tick logging. |
| `conveyor_state_machine.py` | `ConveyorZoneStateMachine` - the happy-path indexing logic. |
| `conveyor_indexing_logger.py` | Background-batched parquet writer, schema-compatible with theia's real data collection. |
| `pick_and_place.py` | `MagicAttachPickPlace` + UR10/pedestal setup - the pick-and-place phase state machine. |
| `proto/sim_conveyor_action.proto` | Sim-only action schema (see above). |
| `gen_proto.sh` | Generates Python bindings for both theia's real state schema and the sim action schema. |
