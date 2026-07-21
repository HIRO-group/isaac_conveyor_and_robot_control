# Conveyor indexing (sim)

Zone-accumulation indexing controller for the inbuilt surface-velocity
conveyors in `~/conveyor_setup.usd` (two closed loops), with per-tick data
logging designed for later imitation learning and reinforcement learning on
the indexing policy, plus a UR20 (driven by NVIDIA cuMotion RMPflow) that
picks a box off loop 1 and places it on loop 2 using magic attach (no real
grasp physics).

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
- **Pick-and-place**: `pick_and_place.py`'s `MagicAttachPickPlace` runs a UR20
  (on a static pedestal at the reach-balanced midpoint between the two loops)
  through a phase state machine: approach, descend, "attach" (disable the
  box's rigid body and rigidly offset-follow the end effector - privileged
  pose queries only, no perception), lift, traverse, descend, detach,
  retract. Motion is driven by NVIDIA cuMotion's GPU RMPflow planner
  (`isaacsim.robot_motion.cumotion.RmpFlowController`) rather than per-tick
  differential IK - smooth, collision-aware trajectories against the
  conveyor structure/pedestal as real obstacles, instead of a stateless
  one-shot IK solve every physics tick. Phase transitions are success-gated
  (converged tool-frame pose within `EE_POSITION_THRESHOLD`, with a minimum
  tick floor and a fixed-duration timeout as a fallback safety net) rather
  than purely fixed-duration. See "Known gaps" for the UR10->UR20 migration
  history and the config-generation pipeline under `robot_configs/`.
  `ConveyorTrack_01` (loop 1) is configured as the **hold zone**:
  `ConveyorLineController` forces its `downstream_clear` to always False, so
  an arriving box is held stopped in front of the robot instead of
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

Note: this machine has a real X server at `:0` (bridged via `x11vnc` +
noVNC - the same one the interactive GUI screenshots in this project's
history come from), but a plain non-interactive shell (e.g. an automated
script/agent session, not a logged-in desktop session) doesn't have
`DISPLAY` set. Running `conveyor_indexer.py` (which uses
`SimulationApp({"headless": False})`) without `DISPLAY` set in such a shell
makes the app exit cleanly after a few seconds with no error and no window
- pass `DISPLAY=:0` explicitly. When running as a long-lived background
process from a shell whose session may itself get torn down between
commands, use `setsid nohup ... & disown` (plain `command &` alone was
observed getting reaped between separate tool invocations in an automated
session; `setsid`/`nohup`/`disown` together fully detach it).

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
- ~~Pick-and-place phase timing is fixed-duration, not tolerance-based~~
  **Resolved**: phase advancement is now success-gated on the tool frame's
  world position converging within `EE_POSITION_THRESHOLD` of the phase
  target (with a `MIN_STEPS_PER_PHASE` floor to avoid a one-tick fluke). The
  old fixed-duration values (`PHASE_TICKS`) are kept as a logged timeout
  safety net, not the primary signal - see `_drive_to()` in
  `pick_and_place.py`.
- ~~Reach margin is real but not generous~~ **Resolved by the UR10->UR20
  swap** (see below) - not yet re-validated empirically in the running sim
  at time of writing. The UR20 has a 1.75 m spec reach vs. the ~1.18 m needed
  each side from the balanced midpoint (`ROBOT_POSITION`/`PLACE_XY`), about
  67% of spec reach vs. the UR10's failed ~90% attempt - comfortable margin
  on paper, but confirm in the running sim before relying on it, same as the
  UR10 numbers before it were confirmed via 55s of logged data.
- ~~No obstacle avoidance~~ **Resolved by the cuMotion migration** (see
  below): `MagicAttachPickPlace` now registers the conveyor structure and
  pedestal as real collision obstacles via
  `isaacsim.robot_motion.experimental.motion_generation.WorldBinding` +
  `SceneQuery`, so RMPflow plans around them instead of relying on
  waypoints chosen to clear them by construction. Both known box paths are
  explicitly excluded from the tracked obstacle set (a carried/targeted box
  must never register as something to dodge).
- **UR10 -> UR20 + raw-IK -> cuMotion RMPflow migration.** The previous
  version drove the UR10 with a stateless per-tick differential-IK solve
  (`set_end_effector_pose`) - visibly jerky motion, no obstacle awareness,
  and (worse) let the approaching arm physically collide with and displace
  the box before "attaching" it (~0.34 m offset observed between the box and
  end effector at `ATTACH` in one logged run). Migrated to:
  - **UR20** (1.75 m spec reach) instead of the UR10 (~1.3 m), to stop the
    reach margin being a nail-biter.
  - **cuMotion RMPflow** (`isaacsim.robot_motion.cumotion.RmpFlowController`)
    instead of raw differential IK, for smooth, collision-aware motion.
  - **Disabling the box's rigid body the moment it's selected as the pick
    target** (`WAITING` -> `MOVE_ABOVE_PICK`), not at `ATTACH` as before -
    this is what actually fixes the shove-before-grasp bug (a disabled
    rigid body can't be physically displaced by anything, including the
    approaching arm), not the smoother motion by itself.

  cuMotion ships a ready-made config only for Franka and UR10 (confirmed:
  `isaacsim.robot_motion.cumotion`'s `robot_configurations/` directory has
  exactly those two - nothing for UR20 or any other UR variant). A UR20
  config was generated from scratch under `robot_configs/ur20/` via three
  scripts in `robot_configs/` (kept for reproducibility, not one-off
  interactive work):
  1. `generate_ur20_urdf.py` - exports `robot.urdf` directly from the
     bundled `ur20.usd` asset via `isaacsim.asset.exporter.urdf.UsdToUrdfConverter`
     (rather than sourcing/processing Universal Robots' external ROS xacro
     files), so the URDF's kinematics/limits are guaranteed to match what's
     actually in the scene. Also appends a `tool0` fixed frame (absent from
     the bundled asset, which only has a `flange` Xform under
     `wrist_3_link`) sourced from that prim's actual authored transform, not
     guessed.
  2. `generate_ur20_spheres_morphit.py` - generates collision spheres per
     link using [MorphIt](https://github.com/HIRO-group/MorphIt-1) (a
     gradient-based sphere-packing optimizer, run in an isolated venv - see
     script docstring) instead of cuMotion's own simpler built-in generator,
     per explicit request. Two of UR20's seven collision meshes
     (`upper_arm_link`, `forearm_link`) are non-watertight, which makes
     MorphIt's final escaped-sphere safety prune considerably more
     aggressive (e.g. 12 requested spheres down to 5 survived on the first
     pass); requesting roughly double the desired count and letting the
     prune do its job recovered most of the intended density (12->10,
     15->14) - documented in the script rather than silently accepted.
  3. `generate_ur20_xrdf.py` - builds `robot.xrdf` from the MorphIt sphere
     data using `isaacsim.robot_setup.xrdf_editor`'s headless core API
     (`EditorState`/`CollisionSphereEditor.add_sphere()`, bypassing Lula's
     own mesh-based generator entirely) rather than the interactive "Lula
     Robot Description Editor" UI - confirmed scriptable (no GUI needed) by
     reading its source. Mirrors the bundled UR10 config's own split:
     `upper_arm_link`/`forearm_link`/`wrist_1_link`/`wrist_2_link` get
     spheres for general obstacle avoidance, `shoulder_link` gets a
     self-collision-only set, and `base_link`/`wrist_3_link` are instead
     covered by `rmp_flow.yaml`'s `body_capsules`/`body_collision_controllers`
     (same split, not an arbitrary omission).

  `rmp_flow.yaml` itself is NOT auto-generated by either tool - same as
  UR10's own bundled copy, it's a small hand-authored tuning file. UR20's
  version was adapted from UR10's: the RMPflow gain/weight constants
  (position/damping gains, metric scalars) are solver tuning, not physical
  facts about the robot, so they're reused as-is; only the physically-real
  quantities (`joint_velocity_cap_rmp.max_velocity`, `body_capsules`/
  `body_collision_controllers` radii) were changed, each sourced from UR20's
  actual exported URDF limits or measured mesh bounds - see the comments at
  the top of `robot_configs/ur20/rmp_flow.yaml` for exact values and
  sourcing. Both `isaacsim.asset.exporter.urdf` and
  `isaacsim.robot_setup.xrdf_editor` (plus the `lula` Python bindings)
  needed for this generation pipeline are missing from the runtime Isaac Sim
  distribution at `/home/ubuntu/IsaacSim` - only present in the built
  extensions under `/home/ubuntu/IsaacSim-source`'s `_build` output (see
  each script's docstring for the exact `PYTHONPATH`/enable-extension
  invocation needed to re-run them).

  Also hit and worked around: `isaacsim.robot_motion.cumotion`'s own shipped
  code (`transforms.py`, `cumotion_world_interface.py`) calls
  `np.reshape(arr, shape=[...])` - the `shape=` keyword only exists from
  NumPy 2.1 onward, but this Isaac Sim install's bundled NumPy is 1.26.4, so
  every such call raises `TypeError`. This affects any robot driven through
  `RmpFlowController` on this machine, not just UR20 - even the
  officially-supported bundled UR10 example would hit it. Patched with a
  small, reversible `np.reshape` compatibility shim local to
  `pick_and_place.py`'s own process (never touches the shared Isaac Sim
  installation) rather than editing the vendored file.

- **RMPflow convergence in the real scene is NOT yet fully working - the
  arm reaches the right X/Y position above the box but stalls part-way
  through descending, and orientation doesn't settle to straight-down.**
  This is the main open item from this session. Confirmed facts, in the
  order discovered (re-run `PYTHONPATH=/tmp/proto_gen
  /home/ubuntu/IsaacSim/python.sh /home/ubuntu/conveyor_indexing/conveyor_indexer.py`
  with `DISPLAY=:0` to keep debugging - see "Setup" for why `DISPLAY` needs
  setting explicitly in a non-interactive shell on this machine):
  - Fixed real bugs along the way, all still valid: `create_pedestal_and_robot()`'s
    `robot.set_default_state(...)` was never actually taking effect -
    `World.reset()` (`isaacsim.core.api`, the classic API) has no knowledge
    of `isaacsim.core.experimental` prims and never called
    `reset_to_default_state()` on our plain `Articulation` robot, confirmed
    by reading `world.py`. The arm was silently starting from its raw
    near-zero USD-authored pose every run regardless of
    `UR20_DEFAULT_JOINT_POSITIONS`. Fixed by calling
    `robot.reset_to_default_state()` explicitly right after `world.reset()`
    in `conveyor_indexer.py`.
  - `UR20_DEFAULT_JOINT_POSITIONS` is a real, verified-reachable "ready"
    pose (tool0 pointing down, 0.5 m below the robot's own base) - derived
    by literally running `RmpFlowController` to convergence against that
    target in an empty, obstacle-free scene and reading back the converged
    joint angles (see the constant's own comment in `pick_and_place.py`),
    not hand-picked. `robot_configs/ur20/robot.xrdf`'s
    `default_joint_positions` was updated to match (both must stay in
    sync - see `generate_ur20_xrdf.py`'s `DEFAULT_JOINT_POSITIONS_RAD`).
  - A **control test with RMPflow's obstacle tracking fully disabled**
    (`_DEBUG_DISABLE_OBSTACLE_TRACKING`, left `False` in committed code)
    converged to a WORSE final position (0.95 m from target) than with
    obstacle tracking on (0.54 m) - this conclusively rules out
    collision-avoidance repulsion as the cause of the stall, despite it
    being the most obvious suspect (a first attempt at a synthetic capsule
    obstacle proxy had earlier caused real problems - see below - which is
    why it was suspected first).
  - The actual fix that mattered: `rmp_flow.yaml`'s `target_rmp` (position)
    vs `axis_target_rmp` (orientation) `accel_p_gain` were still at UR10's
    original values (80 / 200 respectively - orientation weighted 2.5x
    position). Rebalanced to 300 / 80 (position now dominant). This
    produced a real, measurable improvement: X/Y position converged to
    within ~3 cm of the pick target (previously off by 0.5-1 m).
  - Z (descent) and full orientation convergence remain unresolved:
    doubling the phase tick budgets (`PHASE_TICKS`) only bought ~6 cm of
    further Z progress, and the tool's actual world Z-axis direction
    (see the `_local_z_axis_in_world()` diagnostic helper and the
    `tool_z_axis_world=` value logged in `DESCEND_TO_PICK`'s debug print)
    swung noticeably between separate runs rather than settling toward the
    intended `(0, 0, -1)` - more consistent with a genuine stall (a joint
    limit or near-singularity, or a still-unresolved gain issue between the
    Z-descent and orientation terms specifically) than with "just needs
    more time." Not yet root-caused. Next steps worth trying: log per-joint
    positions against `robot_configs/ur20/robot.urdf`'s `<limit>` values
    during the stall to check whether a specific joint is pinned; check
    whether the exact combination of the pick point's position and
    `DOWN_ORIENTATION` is kinematically reachable at all for this
    pedestal/`ROBOT_POSITION` (a manipulability/near-singularity check, not
    just a reach-distance one); consider whether `target_rmp`/
    `axis_target_rmp`'s `accel_d_gain` (damping, unchanged from UR10) also
    needs rebalancing alongside the p_gain change already made.
  - A **first attempt at obstacle-proxy geometry was wrong and reverted**:
    synthetic capsules sized from each zone's bbox were created as
    real PhysX colliders (`UsdPhysics.CollisionAPI` applied) to work around
    the real belt geometry being untrackable by `WorldBinding` (see
    `conveyor_indexer.py` git history) - this made them physically collide
    with and shove the real boxes, and they were also grossly oversized
    relative to the actual belt bboxes (root cause not diagnosed - possibly
    a units/frame mismatch in the capsule `axes`/`radii`/`heights`
    parameterization, worth checking before trying this approach again).
    The current, working approach instead just excludes each track's
    untrackable `.../Belt` sub-prim specifically (not the whole track),
    leaving the real support-post structure mesh (a different, ordinary,
    trackable Mesh - `SM_ConveyorBelt_A06_02`) correctly tracked and
    avoided with no synthetic geometry at all.
- **Single-box capacity per zone/queue depth untested beyond two boxes** -
  behavior with a longer queue of boxes on loop 1, or a busy loop 2, hasn't
  been exercised.

## Files

| File | Purpose |
|---|---|
| `conveyor_indexer.py` | Standalone Isaac Sim entry point - occupancy sensing, zone wiring (both loops), pick/place wiring, per-tick logging. |
| `conveyor_state_machine.py` | `ConveyorZoneStateMachine` - the happy-path indexing logic. |
| `conveyor_indexing_logger.py` | Background-batched parquet writer, schema-compatible with theia's real data collection. |
| `pick_and_place.py` | `MagicAttachPickPlace` + UR20/pedestal setup - the cuMotion-RMPflow-driven pick-and-place phase state machine. |
| `robot_configs/generate_ur20_urdf.py` | Exports `robot_configs/ur20/robot.urdf` from the bundled `ur20.usd` asset (+ appends a `tool0` frame). |
| `robot_configs/generate_ur20_spheres_morphit.py` | Generates per-link collision spheres via MorphIt, writes `robot_configs/ur20/morphit_spheres/*.json`. |
| `robot_configs/generate_ur20_xrdf.py` | Builds `robot_configs/ur20/robot.xrdf` (+ Lula `robot_description.yaml`) from the MorphIt spheres. |
| `robot_configs/smoke_test_ur20_rmpflow.py` | Standalone RmpFlowController convergence test against the generated UR20 config, in isolation from the full scaffold. |
| `robot_configs/ur20/` | Generated UR20 cuMotion config: `robot.urdf`, `robot.xrdf`, hand-authored `rmp_flow.yaml`, `meshes/`, `morphit_spheres/`. |
| `proto/sim_conveyor_action.proto` | Sim-only action schema (see above). |
| `gen_proto.sh` | Generates Python bindings for both theia's real state schema and the sim action schema. |
