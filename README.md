# Conveyor Indexing

A NVIDIA Isaac Sim scaffold for zone-accumulation conveyor indexing and
robotic pick-and-place. Two open (non-looping) conveyor lines are controlled
by a zone-based indexing state machine, while a UR20 arm (driven by NVIDIA
cuMotion) picks boxes off the first line and places them on the second,
which carries them into a waiting truck. Every tick logs conveyor state,
robot joint positions, and camera frames in a schema designed for training
imitation-learning and reinforcement-learning policies on the indexing and
pick-and-place tasks.

## Setup

### 1. Install Isaac Sim

[Isaac Sim](https://developer.nvidia.com/isaac/sim) 6.0, locally or via
[Isaac Automator](https://github.com/isaac-sim/IsaacAutomator).

### 2. Clone this repo

```bash
git clone --recurse-submodules git@github.com:HIRO-group/isaac_conveyor_and_robot_control.git
```

`proto/` is a submodule
([conveyor_indexing_protos](https://github.com/HIRO-group/conveyor_indexing_protos)).

### 3. Run the setup script

```bash
ISAAC_PYTHON=/path/to/isaac-sim/python.sh bash scripts/setup.sh
```

Installs `requirements.txt` into Isaac Sim's python and generates the protobuf
bindings (`PROTO_OUT`, default `/tmp/proto_gen`).

### 4. Get a scene package

The cell itself (USD stage, camera poses, `cell.yaml`) lives outside this repo.
Point `SIM_SCENE_DIR` (or `--scene`) at a directory containing:

```
cell.yaml            loops, stations, cameras, tuning (see tests/fixtures/scene/cell.yaml)
<stage>.usd          referenced from cell.yaml as `stage`
camera_poses.json    referenced as `camera_poses`; may be absent on a fresh cell
```

Remote USD assets are mirrored into `SIM_ASSET_DIR` (default `~/isaac_assets`) by
`scripts/download_assets.py`; mount that directory at `/assets` in Docker.

## What to run

```bash
ISAAC_PYTHON=/path/to/python.sh SIM_SCENE_DIR=/path/to/scene DISPLAY=:0 bash scripts/run.sh
```

Environment variables:

- `CONVEYOR_INDEXING_HEADLESS=1` (or `--headless`) — no GUI.
- `CONVEYOR_INDEXING_EXTERNAL_ACTION=1` — arms and belts are driven over Zenoh (below).
- `CONVEYOR_INDEXING_RECORD=1` — 30Hz training rows to `data/recordings/`.
- `CONVEYOR_INDEXING_RECORD_MCAP=1` — every channel to `data/mcap/`.
- `CONVEYOR_INDEXING_DATA_DIR` — output root (default `data/`).
- `CONVEYOR_INDEXING_VIEWPORT_FRAMES_DIR` — headed only: write the viewport to
  PNG on every render (`camera.fps` frames per sim second) with a
  `frames.csv` of sim times; `scripts/video/encode_frames.sh` makes the mp4.
- `CONVEYOR_INDEXING_REALTIME=1` — pace sim time to the wall clock (any factor,
  e.g. `0.5` for half speed), for screen-recording the viewport; the loop is
  otherwise unthrottled and runs several times faster than real time.
- `ZENOH_ROUTER=tcp/host:7447` — connect to a router instead of peer mode.
- `SIM_ZENOH_PREFIX` — key prefix (default `sim`).

### Capacity faults on the built-in controller

The autonomous state machine can run under a degraded capacity record
(capability-diffusion's kappas, see `src/sim_cell/faults.py`), for
demonstrations that need no external policy:

- `CONVEYOR_INDEXING_KAPPA_R_ARM1` — arm 1's reach: it only picks boxes on the
  nearest fraction of its pick zone; boxes it cannot serve pass on to arm 2.
- `CONVEYOR_INDEXING_REACH_MODE=attempt` — the capacity-blind contrast: arm 1
  selects as if nominal but physically stops at the same reach, strains there,
  retreats and tries again for as long as the box is there (needs
  `CONVEYOR_INDEXING_HOLD_WHILE_BUSY=1` so the zone keeps the box);
  `CONVEYOR_INDEXING_REACH_STALL_S` (default 1) is how long it strains at the
  limit. Default `decline` is the pass-on above.
- `CONVEYOR_INDEXING_KAPPA_H_ARM1` — arm 1's hold: the probability a grasp
  survives; a failed hold drops the box 30–50 % of the way through the swing to
  the place belt. `CONVEYOR_INDEXING_DROP_EVERY_HOLD=1` drops every hold.
- `CONVEYOR_INDEXING_KAPPA_C` — loop 1 (the belts feeding both pick zones)
  runs at this fraction of its nominal speed.
- `CONVEYOR_INDEXING_SPAWN_LAYOUT=near_far` — every wave is one pair: a box at
  arm 1's belt edge and one at the far edge; `far` spawns the far box alone.
- `CONVEYOR_INDEXING_MAX_WAVES` — stop after this many automatic waves.
- `CONVEYOR_INDEXING_HOLD_WHILE_BUSY=1` — a pick zone keeps holding its boxes
  while its arm is mid-cycle instead of overflowing them downstream.
- `CONVEYOR_INDEXING_FAULT_SEED` — fixes the drop decisions and the hesitations.
- `CONVEYOR_INDEXING_PICKS_PER_MIN` / `CONVEYOR_INDEXING_HESITATIONS` — the
  hesitant-policy imitation: each arm starts at most that many cycles a minute
  (waiting at its staging pose in between) and freezes that many times, one to
  three seconds each, on the way down to a box.

`scripts/video/` holds one launcher per take (`kappa_r_1.0.sh`,
`kappa_r_0.3.sh`, `kappa_r_0.3_attempt.sh`, `kappa_r_0.3_pass_on.sh`, `kappa_r_0.3_random.sh`, `hesitant_2pm.sh`, `kappa_h_0.3.sh`, `live_graph_r0.3.sh` (the learned policy through the eval harness),
`kappa_c_0.5.sh`), hardcoded to this
machine's Isaac python and the dual-arm scene; both are overridable with
`ISAAC_PYTHON` and `SIM_SCENE_DIR`.

## Docker

```bash
docker build -t conveyor-sim:local --build-arg SIM_GIT_SHA=$(git rev-parse --short=12 HEAD) .
docker run --gpus all --rm -v /path/to/scene:/scene:ro -e SIM_SCENE_DIR=/scene \
  -e ZENOH_ROUTER=tcp/zenoh:7447 -e CONVEYOR_INDEXING_EXTERNAL_ACTION=1 conveyor-sim:local
```

The image is headed by default: pass `-e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix` (after
`xhost +local:docker`) or set `CONVEYOR_INDEXING_HEADLESS=1`. Its healthcheck passes once
`sim/clock` is publishing.

## Zenoh contract

All keys are under `SIM_ZENOH_PREFIX` (default `sim`); schemas are in `proto/`.

| key | direction | message |
|---|---|---|
| `sim/camera/list` | pub (latched + queryable) | `CameraList` |
| `sim/camera/<id>/color` | pub | raw RGB8 bytes, `FrameMetadata` attachment |
| `sim/arm/<n>/state` | pub, control rate | `SimArmState` |
| `sim/arm/<n>/tool_pose` | pub | `ArmToolPose` |
| `sim/conveyor/state` | pub, control rate | `SimConveyorStates` |
| `sim/boxes/state` | pub | `BoxStates` |
| `sim/clock` | pub, 10 Hz | `SimClock` |
| `sim/run_metadata` | pub (latched + queryable) | `RunMetadata` |
| `sim/arm/<n>/action_command` | sub | `SimArmActionCommand` (joint setpoints, suction) |
| `sim/conveyor/command` | sub | `SimConveyorCommands` |
| `sim/boxes/command` | sub | JSON `{"op": "auto"\|"clear"\|"spawn", ...}` |

## Running an external controller

With `CONVEYOR_INDEXING_EXTERNAL_ACTION=1` nothing moves until commands arrive;
there is no autonomous fallback. Start the sim, then the monitor, then the controller:

```bash
CONVEYOR_INDEXING_EXTERNAL_ACTION=1 SIM_SCENE_DIR=... ISAAC_PYTHON=... bash scripts/run.sh
PYTHONPATH=src:/tmp/proto_gen python3 scripts/monitor_external_action.py
```

`scripts/camera_probe.py` dumps one frame to a PPM file to check the camera stream.

## Local data collection

`scripts/collect_local.py` runs a headless MCAP-recording sim and streams closed
files to `$SIM_GCS_PREFIX/<run_id>/instance_00/mcap/`:

```bash
SIM_GCS_PREFIX=gs://bucket/data_collection/sim python3 scripts/collect_local.py --sim-seconds 600 --keep-local
```

`--sweep-only --run-id <run_id>` uploads what a crashed run left behind.
`scripts/foxglove_sim_layout.json` is a Foxglove layout for the recorded MCAPs.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{briscoe_martinez_conveyor_indexing,
  author = {Briscoe-Martinez, Gilberto},
  title = {Conveyor Indexing: A Zone-Accumulation Conveyor Indexing and Pick-and-Place Scaffold for Isaac Sim},
  year = {2026},
  publisher = {HIRO Group, University of Colorado Boulder},
  url = {https://github.com/HIRO-group/isaac_conveyor_and_robot_control}
}
```
