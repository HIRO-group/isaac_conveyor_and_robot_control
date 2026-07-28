# Conveyor indexing (sim)

Zone-accumulation indexing controller for the inbuilt surface-velocity
conveyors in `~/5_conv_env.usd` (two short, OPEN/non-looping lines), with
per-tick data logging designed for later imitation learning and
reinforcement learning on the indexing policy, plus a UR20 (driven by
NVIDIA cuMotion) that picks a box off loop 1 and places it on loop 2 using
magic attach (no real grasp physics), where the belt then carries it off the
far end into a waiting truck.

## Environment

`sim_cell.layout`'s `STAGE_PATH` currently points at
`/home/ubuntu/5_conv_env.usd`, a smaller, purpose-built scene distinct from
the two earlier ones this repo was originally developed against
(`conveyor_setup.usd`, then `racetrack.usd` - see "Known gaps" below for
that history, most of which describes those older scenes, not this one):

- **Two open (non-looping) lines**, not closed ovals: loop 1
  (`ConveyorTrack`/`_01`/`_02`, along Y=0) and loop 2 (`ConveyorTrack_09`/
  `_10`, along Y~2.186), both straight-only (no curved zones) and spanning
  the same X range. `ConveyorLineController` supports both a closed loop
  (modulo-wrapped neighbor indices, for the older scenes) and an open line
  (`closed_loop=False`, the default and what's used here) - the first zone's
  upstream and the last zone's downstream are treated as always
  available/clear rather than wrapping around.
- **Boxes ship pre-authored in the scene**: ~18-19 `CubeBox_*` prims already
  sit stacked (two layers) directly on `ConveyorTrack`'s belt - unlike
  `racetrack.usd`, which shipped with no boxes and needed them referenced in
  at runtime. They're pure visual payloads with no physics schemas at all;
  `sim_cell.stage_setup.boxes.discover_box_prim_paths` discovers them and adds
  `RigidBodyAPI` + convex-hull `CollisionAPI` + an estimated mass at runtime
  (`apply_box_physics`), the same "don't edit the source USD" convention
  already used for `sim_cell.stage_setup.tracks.deactivate_frame_meshes`.
- **A `SteelBoxTruck_A01_01` sits at loop 2's far end**: its bed is ~0.83 m
  below belt-top height, right past `ConveyorTrack_10`'s end, so a box that
  rides loop 2 to completion runs off the belt and drops into the truck bed.
  Like the boxes, the truck ships as a pure visual payload -
  `sim_cell.stage_setup.truck.apply_truck_collision` adds a static collider to
  its body mesh so boxes actually land in it instead of clipping through.
- **Pick/place geometry fits a UR20 without any runtime repositioning**:
  `ConveyorTrack_01` (loop 1 pick zone) and `ConveyorTrack_09` (loop 2 place
  zone) are both centered at local X=-3; a robot at the Y midpoint between
  the two loops' near edges reaches each at ~1.09 m, comfortably inside the
  UR20's 1.75 m spec reach.
- **Referenced assets are fetched from the public Omniverse S3 content
  bucket over HTTPS** (the conveyor belt asset, the truck, and the 3 box
  variants) - every stage open re-downloads them unless localized. See
  "Setup" for caching them locally.

## Design

- **Two independent lines**: see "Environment" above for 5_conv_env.usd's
  specific zone layout; `ConveyorLineController` (with its own
  neighbor-occupancy wiring) is shared code for both open lines here and the
  closed loops in the older `conveyor_setup.usd`/`racetrack.usd` scenes.
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
  robot, pedestal, or truck root are structure, not items, and are excluded.
- **Indexing logic**: `conveyor_indexing.state_machine` implements a
  best-effort "happy path" subset of the real `ConveyorStateMachineCode` enum
  from `~/theia/proto/plc-connector/plc-connector.proto` - see that module's
  docstring for the exact state cycle and what's deliberately not
  implemented.
- **Pick-and-place**: `pick_and_place.controller.MagicAttachPickPlace` runs a UR20
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
  `sim_cell.layout.PICK_ZONE_INDEX` / the `hold_zone_indices` argument to
  `conveyor_indexing.line_controller.ConveyorLineController`. The hold
  zone also keeps running past first-occupied until the box reaches the
  zone's geometric center (`conveyor_indexing.zone.ConveyorZone.is_past_center`,
  gated via `ConveyorZoneStateMachine.step`'s `at_stop_position` argument) rather than
  stopping wherever it first entered the occupancy sensor - the robot needs
  a fixed, reachable pick point every cycle, not one that drifts with
  however far the box carried into the zone before the belt cut out.
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
    `conveyor_indexing.parquet_logger.ConveyorIndexingLogger`, following the
    same background-batched `ParquetWriter` pattern as `data_collection_vol2.py`.
- **Cameras**: 6 runtime-created cameras - `pick_cam`/`place_cam`/`hand_cam`
  for each of the two robot stations, matching theia's production camera
  seed (`~/theia/infra/etcd/bootstrap/seed/defaults.json`'s `camera.*`, two
  robot cells x three roles). Color only for now (RGB8, 640x480@30 by
  default - see `sim_cell.settings.CAMERA_*`); depth fields exist in the
  wire contract but stay zeroed. Overhead cams (`pick_cam`/`place_cam`) are
  positioned above each zone's belt from real belt-bbox geometry (same
  `conveyor_indexing.belt_geometry.compute_belt_bounds` helper
  `robot_placement.py` already uses); hand cams are parented under each
  UR20's wrist flange (`sim_cell.layout.HAND_CAM_PARENT`) so they ride the
  arm's kinematics. See `src/cameras/` for the package and "Camera wire
  contract" / "Camera tuning" below for the publishing contract and the
  visualize-adjust-save workflow for getting each camera's pose right.
  - **GPU-first capture**: this machine is CPU-throttled, so the whole
    per-frame path after rendering - reading the Replicator "rgb" annotator,
    converting RGBA to RGB - runs on the GPU (`cameras.rig.CameraRig`, via a
    zero-copy Warp array -> `warp.torch.to_torch` -> GPU-side slice), with
    exactly one host copy per published frame (the final RGB buffer, not the
    RGBA source). Falls back to CPU capture (logged once) if CUDA/Torch
    interop isn't available in a given Isaac Sim install.

### Camera wire contract

Cameras publish over Zenoh on the same key namespace and message shapes
theia's real camera service and data collection use
(`~/theia/proto/camera/camera.proto`, `~/theia/data_collection/src/data_collection_vol2.py`)
so recorded sim data and theia's collectors are directly compatible - but
**conveyor_indexing does not depend on theia** (this repo is public research
code; theia is private). `proto/sim_camera.proto` is a local mirror,
wire-compatible by field number/type with theia's schema (protobuf's wire
format only cares about those, not package names) - the same pattern
`proto/sim_conveyor_action.proto` already uses for the action schema. See
that file's header comment for exactly what's mirrored and what's
deliberately omitted (notify/health, future work).

- `theia/camera/list`: latched publisher + queryable serving a `CameraList`
  protobuf (one `CameraInfo` per camera) - `cameras.zenoh_publisher.CameraZenohPublisher`
  does both a `put()` and a live `declare_queryable()` on this key, since
  consumers query it differently (a one-shot `session.get()` vs. a
  timeout-bounded query).
- `theia/camera/{serial}/color`: raw RGB8 bytes (no proto wrapper,
  `len == height*width*3`), one per camera, with a `FrameMetadata` protobuf
  (timestamps + monotonic frame number + a UUIDv7) as the Zenoh
  *attachment*, not the payload. `depth` topics are advertised in
  `CameraInfo` but nothing publishes to them yet.
- Session setup (`cameras.zenoh_publisher._open_session`) mirrors theia's own
  collector: `ZENOH_ROUTER` env var set -> connect to that endpoint; unset ->
  open in peer-to-peer mode, so the sim runs standalone with no router
  needed (see "Setup" below for verifying this without theia at all).

## Setup

1. Run `scripts/setup.sh`, which does two things:
   - Generates the protobuf Python bindings (`protoc` and the `protobuf`
     Python package must be available - neither was present on this
     machine as of this writing; install `protoc` first, e.g. via your
     package manager) via `gen_proto.sh`.
   - Installs `eclipse-zenoh==1.7.1` into **Isaac Sim's bundled python**
     (`/home/ubuntu/IsaacSim/python.sh`, not the system python) - required
     for camera publishing (see "Design" above and `src/cameras/zenoh_publisher.py`);
     the sim exits with an actionable error if it's missing. `scripts/run.sh`
     also pre-flight checks for it before paying a full Isaac Sim startup.
   ```bash
   bash /home/ubuntu/conveyor_indexing/scripts/setup.sh
   ```
2. Run the indexer via `scripts/run.sh`, which sets `PYTHONPATH` to both
   `src/` (this repo's `conveyor_indexing`/`pick_and_place`/`sim_cell`/`cameras`
   packages) and `/tmp/proto_gen` (the bindings from step 1), then execs
   Isaac Sim's bundled python on `scripts/run_conveyor_indexing.py`:
   ```bash
   DISPLAY=:0 bash /home/ubuntu/conveyor_indexing/scripts/run.sh
   ```
   (see "Repo layout" below for what lives in `src/` vs `scripts/`.)

   By default this opens a Zenoh session in **peer-to-peer mode** (no router
   needed) - set `ZENOH_ROUTER` (e.g. `tcp/127.0.0.1:7447`, theia's own
   default) to instead connect to a running Zenoh router, such as one from a
   real theia deployment:
   ```bash
   ZENOH_ROUTER=tcp/127.0.0.1:7447 DISPLAY=:0 bash /home/ubuntu/conveyor_indexing/scripts/run.sh
   ```

   For data-collection runs where wall-clock throughput matters (faster than
   realtime), skip the GUI window entirely with `CONVEYOR_INDEXING_HEADLESS=1`
   (camera publishing and logging are unaffected - only the viewport and its
   `DISPLAY` requirement go away):
   ```bash
   CONVEYOR_INDEXING_HEADLESS=1 bash /home/ubuntu/conveyor_indexing/scripts/run.sh
   ```
3. Verify the camera contract without theia at all: with the sim running,
   in another shell:
   ```bash
   PYTHONPATH=/tmp/proto_gen python3 /home/ubuntu/conveyor_indexing/scripts/camera_probe.py
   ```
   This fetches `theia/camera/list`, subscribes to one camera's color topic,
   validates the frame size and `FrameMetadata` attachment, and writes the
   frame out as a `.ppm` image. See `scripts/camera_probe.py`'s docstring for
   options (`--serial`, `--out`).

### Camera tuning

Overhead camera positions are derived from belt geometry and hand cams from
a fixed flange offset (see "Design" above) - close, but likely needing a
manual nudge to actually frame each zone/tool correctly. To tune and save
camera transforms:

```bash
CONVEYOR_INDEXING_CAMERA_TUNING=1 DISPLAY=:0 bash /home/ubuntu/conveyor_indexing/scripts/run.sh
```

This opens one small viewport per camera, each locked to that camera's live
feed (`sim_cell.camera_tuning`). Drag the camera prims in the main
viewport/stage tree with the ordinary USD gizmo while watching the locked
viewports update, then press **F10** to save every camera's current
transform to `environments/camera_poses.json` (local/flange-relative for
hand cams, so the saved offset stays correct as the arm moves; world-frame
for overhead cams). Commit that file - subsequent normal runs (env var
unset) load it automatically and take it over the derived defaults (see
`sim_cell.camera_layout.build_camera_specs`). Saving is an explicit,
one-shot action; nothing is auto-persisted, so accidental nudges are safe to
ignore by just not pressing F10.

Note: launching a second Isaac Sim instance while another one is already
running interactively (e.g. the `cb_app.py` conveyor-belt Warp sample) will
contend heavily for this machine's single GPU and can make startup very
slow. Close other Isaac Sim instances first if this one seems to hang during
extension loading.

Note: this machine has a real X server at `:0` (bridged via `x11vnc` +
noVNC - the same one the interactive GUI screenshots in this project's
history come from), but a plain non-interactive shell (e.g. an automated
script/agent session, not a logged-in desktop session) doesn't have
`DISPLAY` set. Running `scripts/run_conveyor_indexing.py` (which uses
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
  `conveyor_indexing.telemetry.append_conveyor_state`. Set real per-zone
  types once you know which belts are functioning as pick/place/buffer.
- **PackML mapping is a coarse two-bucket approximation**
  (`conveyor_indexing.telemetry.machine_to_packml`): EXECUTE while
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
  more than one cycle in a single run. (Describes `conveyor_setup.usd`; see
  "Environment" above for `5_conv_env.usd`'s own pre-authored box pallet.)
- **Box mass in `5_conv_env.usd` is an estimate, not a measured value.**
  `_apply_box_physics`'s `BOX_DENSITY_KG_PER_M3` (150 kg/m^3, applied to each
  box's own bbox volume) is a plausible ballpark for a lightly-packed
  shipping box, chosen because the scene doesn't author a mass for these at
  all. Revisit if carried-box dynamics (settling time, how the belt handles
  it, cuMotion's grasp) look off.
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
  safety net, not the primary signal - see
  `pick_and_place.trajectory.TrajectoryDriver.drive_to`.
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
  small, reversible `np.reshape` compatibility shim (`pick_and_place.compat`,
  imported first thing in `pick_and_place/__init__.py`) local to this
  process (never touches the shared Isaac Sim installation) rather than
  editing the vendored file.

- **RMPflow convergence in the real scene is NOT yet fully working - the
  arm reaches the right X/Y position above the box but stalls part-way
  through descending, and orientation doesn't settle to straight-down.**
  This is the main open item from this session. Confirmed facts, in the
  order discovered (re-run `DISPLAY=:0 bash
  /home/ubuntu/conveyor_indexing/scripts/run.sh` to keep debugging - see
  "Setup" for why `DISPLAY` needs setting explicitly in a non-interactive
  shell on this machine):
  - Fixed real bugs along the way, all still valid: `create_pedestal_and_robot()`'s
    `robot.set_default_state(...)` was never actually taking effect -
    `World.reset()` (`isaacsim.core.api`, the classic API) has no knowledge
    of `isaacsim.core.experimental` prims and never called
    `reset_to_default_state()` on our plain `Articulation` robot, confirmed
    by reading `world.py`. The arm was silently starting from its raw
    near-zero USD-authored pose every run regardless of
    `UR20_DEFAULT_JOINT_POSITIONS`. Fixed by calling
    `robot.reset_to_default_state()` explicitly right after `world.reset()`
    in `sim_cell.cell.build_cell`.
  - `UR20_DEFAULT_JOINT_POSITIONS` is a real, verified-reachable "ready"
    pose (tool0 pointing down, 0.5 m below the robot's own base) - derived
    by literally running `RmpFlowController` to convergence against that
    target in an empty, obstacle-free scene and reading back the converged
    joint angles (see the constant's own comment in `pick_and_place.ur20`),
    not hand-picked. `robot_configs/ur20/robot.xrdf`'s
    `default_joint_positions` was updated to match (both must stay in
    sync - see `generate_ur20_xrdf.py`'s `DEFAULT_JOINT_POSITIONS_RAD`).
  - A **control test with RMPflow's obstacle tracking fully disabled**
    (`sim_cell.settings.DISABLE_OBSTACLE_TRACKING`, left `False` by default)
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
    (see the `pick_and_place.transforms.local_z_axis_in_world` diagnostic
    helper and the
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
- **Cameras have not yet been run against a live Isaac Sim instance** - the
  package was written and read-verified against the exact APIs present in
  this machine's Isaac Sim install (`omni.replicator.core-1.13.27`,
  `omni.warp.core`'s `warp.torch.to_torch`), but not yet exercised end-to-end
  in the running sim. Things to confirm on first real run (see "Camera
  tuning" above and `scripts/camera_probe.py`):
  - Hand-cam orientation (`sim_cell.camera_layout._hand_cam_spec`'s 180 deg
    X rotation) - verify the tuning viewport shows the tool, not sky, and
    adjust/save if not.
  - Overhead cam framing (`settings.CAMERA_HEIGHT_ABOVE_BELT_M`/focal
    length) actually covers each belt zone with reasonable margin.
  - Actual render-time cost of 6 extra 640x480 render products on this
    single-GPU machine; `settings.CAMERA_FPS` then resolution are the knobs
    if it's too heavy.
  - That the GPU capture path (`cameras.rig.CameraRig._attach_annotator`)
    actually takes the `cuda` branch rather than silently falling back to
    CPU - check the "camera rig ready" log line's `gpu_capture=` value.

## Repo layout

A `src/` layout with four packages, plus thin `scripts/` entry points -
each package is independent, single-purpose, and split into small,
atomic modules:

- **`conveyor_indexing`** - zone occupancy sensing, the indexing state
  machine, per-tick logging. Independent of any particular scene.
- **`pick_and_place`** - UR20 + cuMotion magic-attach pick-and-place phase
  state machine. Independent of any particular scene.
- **`cameras`** - camera rig creation/capture and theia-wire-contract Zenoh
  publishing (see "Design" above). Independent of any particular scene; like
  the other two, never imports `sim_cell`.
- **`sim_cell`** - wiring for *this* cell: `5_conv_env.usd`'s prim layout
  (`layout.py`), this run's tuning (`settings.py`), stage setup
  (`stage_setup/`), camera placement/tuning (`camera_layout.py`/
  `camera_tuning.py`), and the main control loop (`runner.py`). Depends on
  all three of the above; they depend on neither it nor each other.

None of the three packages are pip-installed (see `pyproject.toml`'s own
comment) - `scripts/run.sh` puts `src/` directly on `PYTHONPATH` instead (see
"Setup").

## Files

| File | Purpose |
|---|---|
| `scripts/setup.sh` | One-time setup: generates proto bindings, installs `eclipse-zenoh` into Isaac Sim's bundled python. |
| `scripts/run.sh` | Launches the sim: pre-flight-checks for `zenoh`, sets `PYTHONPATH` (`src/` + the generated proto bindings), execs Isaac Sim's `python.sh` on `run_conveyor_indexing.py`. |
| `scripts/run_conveyor_indexing.py` | Entry point - constructs `SimulationApp`, then hands off to `sim_cell.runner.run`. |
| `scripts/download_assets.py` | Mirrors `5_conv_env.usd`'s referenced S3 assets locally (see `sim_cell.asset_paths`). |
| `scripts/camera_probe.py` | Standalone (no theia, no Isaac Sim) Zenoh client that verifies the camera contract end-to-end and dumps a frame to `.ppm`. |
| `src/conveyor_indexing/state_machine.py` | `ConveyorZoneStateMachine` - the happy-path indexing logic. |
| `src/conveyor_indexing/zone.py` | `ConveyorZone` - one zone's USD ConveyorNode + belt bbox + occupancy. |
| `src/conveyor_indexing/line_controller.py` | `ConveyorLineController` - wires a line's zones together, neighbor occupancy, hold-zone overflow. |
| `src/conveyor_indexing/parquet_logger.py` | `ConveyorIndexingLogger` - background-batched parquet writer, schema-compatible with theia's real data collection. |
| `src/conveyor_indexing/episode_recorder.py` | `EpisodeRecorder` - 30Hz synchronized image+state training rows (see "Recording training data" below). |
| `src/conveyor_indexing/{belt_geometry,occupancy,directions,telemetry,protos}.py` | Supporting atomic modules - belt-top bbox math, PhysX overlap queries, direction-correction geometry, proto message building, and the single point the generated proto bindings are imported from. |
| `src/pick_and_place/controller.py` | `MagicAttachPickPlace` - the pick-and-place phase state machine. |
| `src/pick_and_place/motion_planner.py` | cuMotion `GraphBasedMotionPlanner` construction, incl. the obstacle-scan retry loop. |
| `src/pick_and_place/trajectory.py` | `TrajectoryDriver` - plan-once-per-phase, open-loop trajectory playback. |
| `src/pick_and_place/{compat,ur20,transforms,phases,box_queries,robot_setup,obstacle_guard,ik,attachment,selection}.py` | Supporting atomic modules - the NumPy shim, UR20 constants, quaternion math, phase/tick-budget constants, box pose queries, pedestal/robot spawning, the obstacle-rotation-reset workaround, IK, magic-attach FixedJoint create/delete, and pick-candidate ranking. |
| `src/cameras/rig.py` | `CameraRig` - creates camera prims + render products + RGB annotators, GPU-first frame capture. |
| `src/cameras/zenoh_publisher.py` | `CameraZenohPublisher` - serves the latched camera list, publishes per-camera color frames with a `FrameMetadata` attachment. |
| `src/cameras/{specs,frame_meta,pose_io,protos}.py` | Supporting atomic modules - the camera spec dataclass + theia-contract proto builders, per-frame timestamp/UUIDv7/frame-counter helpers, tuned-pose JSON load/save, and the single point the generated camera proto bindings are imported from. |
| `src/sim_cell/layout.py` | Everything specific to `5_conv_env.usd`'s prim paths (`STAGE_PATH`, zone paths, robot/truck paths, camera root/hand-cam-parent paths). |
| `src/sim_cell/settings.py` | Run-time tuning (control/physics rate, per-line speed, robot placement, camera resolution/fps/placement). |
| `src/sim_cell/stage_setup/` | Opens the stage and prepares it: asset localization, frame-mesh deactivation, box/truck physics. |
| `src/sim_cell/camera_layout.py` | `build_camera_specs()` - derives all 6 camera placements from zone/robot geometry, applies any saved tuning overrides. |
| `src/sim_cell/camera_tuning.py` | `maybe_enable_camera_tuning()` - the opt-in visualize/adjust/save workflow (see "Camera tuning" above). |
| `src/sim_cell/cell.py` | `build_cell()` - builds the World, both lines, both robots, both pick-and-place controllers, the camera rig + publisher. |
| `src/sim_cell/recording.py` | Recording glue: env-var gating, camera-serial->role map, the episode key tracker, and the observation.state layout. |
| `src/sim_cell/runner.py` | `run()` - the main control loop. |
| `src/sim_cell/log_setup.py` | Attaches stdout logging handlers to each package's root logger (see "Logging" below). |
| `robot_configs/generate_ur20_spheres_lula.py` | Generates per-link collision spheres via Lula's built-in generator. |
| `robot_configs/generate_ur20_xrdf.py` | Builds `robot_configs/ur20/robot.xrdf` (+ Lula `robot_description.yaml`) from the collision spheres. |
| `robot_configs/visualize_ur20_collision_spheres.py` | Interactive viewer for the generated collision spheres. |
| `robot_configs/ur20/` | Generated UR20 cuMotion config: `robot.xrdf`, `robot_description.yaml`, `lula_spheres/`. |
| `proto/sim_conveyor_action.proto` | Sim-only action schema (see above). |
| `proto/sim_camera.proto` | Sim-only camera schema, wire-compatible with theia's real camera contract (see "Camera wire contract" above). |
| `environments/camera_poses.json` | Tuned camera transforms saved by the camera-tuning workflow (see "Camera tuning" above); absent on a fresh checkout until first tuned. |
| `gen_proto.sh` | Generates Python bindings for theia's real state schema, the sim action schema, and the sim camera schema. |
| `pyproject.toml` | Tooling config only (mypy/ruff) - see "Repo layout" above for why there's no install step. |

## Recording training data

Set `CONVEYOR_INDEXING_RECORD=1` to record synchronized 30Hz training rows
into `data/recordings/` (default off; separate from the 120Hz tick log in
`data/` so the two parquet streams never mix):

```bash
CONVEYOR_INDEXING_RECORD=1 CONVEYOR_INDEXING_HEADLESS=1 bash scripts/run.sh
```

Each row is captured in a single main-loop iteration - all 6 camera frames,
both arms' joint positions (radians, `Articulation` dof order - logged at
startup as `robot dof_names`), a suction/cups block per arm mirroring the
magic attach state, the latest `StateConveyors` snapshot, and an episode key
that increments whenever either arm starts a pick. Extra columns
(`tick`, `sim_time_s`, `phase_1`, `phase_2`) support re-segmenting episodes
at conversion time instead of re-collecting. Full schema:
`src/conveyor_indexing/episode_recorder.py`; state-vector layout and
serial->role mapping: `src/sim_cell/recording.py`.

Sizing: rows are ~5.5MB raw (6 x 640x480 RGB) at 30Hz; budget roughly
1-4GB/min on disk after parquet zstd, depending on scene content and how far
below realtime the sim runs. Files rotate at episode boundaries (~900 rows).
If the writer can't keep up it drops rows rather than stalling the sim and
logs a warning plus a final drop count - a recording with drops has time
gaps and should not be used for training.

The output is schema-compatible with theia's `dc_to_lerobot.py` converter
(dual-arm layout needs its `--arms 2` flag):

```bash
python dc_to_lerobot.py --data-dir <this repo>/data/recordings \
    --root <output dataset dir> --arms 2 --fps 30
```

## Logging

Every module logs via the standard `logging` module (`logging.getLogger(__name__)`)
rather than `print(..., flush=True)`. `sim_cell.log_setup.configure_logging()`
(called once, at the top of `scripts/run_conveyor_indexing.py`) attaches a
stdout handler to each of the four package-root loggers
(`cameras`, `conveyor_indexing`, `pick_and_place`, `sim_cell`) at `INFO` - deliberately
not `logging.basicConfig`/root propagation, since Kit reconfigures the root
logger into carb's own log system. Log lines are now prefixed with the
module path (e.g. `[sim_cell.runner]`) rather than the old
`[conveyor_indexer]`/`[pick_and_place]` prefixes.

To get the old `DEBUG_LOG_OCCUPANCY_HITS`/`DEBUG_LOG_HOLD_ZONE_STATE`
per-tick diagnostics (and the tick-counter dumps that used to print
unconditionally every 3rd control tick), set the
`CONVEYOR_INDEXING_DEBUG_LOGGERS` env var to a comma-separated list of
logger names before launching, e.g.:
```bash
CONVEYOR_INDEXING_DEBUG_LOGGERS=conveyor_indexing.occupancy,conveyor_indexing.line_controller,sim_cell.debug \
  DISPLAY=:0 bash scripts/run.sh
```
