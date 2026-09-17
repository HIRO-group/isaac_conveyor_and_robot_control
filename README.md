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

Remote USD assets are mirrored by `scripts/download_assets.py`.

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
- `ZENOH_ROUTER=tcp/host:7447` — connect to a router instead of peer mode.
- `SIM_ZENOH_PREFIX` — key prefix (default `sim`).

## Docker

```bash
docker build -t conveyor-sim:local --build-arg SIM_GIT_SHA=$(git rev-parse --short=12 HEAD) .
docker run --gpus all --rm -v /path/to/scene:/scene:ro -e SIM_SCENE_DIR=/scene \
  -e ZENOH_ROUTER=tcp/zenoh:7447 -e CONVEYOR_INDEXING_EXTERNAL_ACTION=1 conveyor-sim:local
```

The image runs headless; its healthcheck passes once `sim/clock` is publishing.

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
