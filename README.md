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

This project runs on top of NVIDIA Isaac Sim, which must be installed first:

- [Isaac Sim](https://developer.nvidia.com/isaac/sim) — install locally
  following NVIDIA's documentation, or
- [Isaac Automator](https://github.com/isaac-sim/IsaacAutomator) — to
  automatically provision a cloud instance with Isaac Sim pre-installed.

### 2. Clone this repo

```bash
git clone --recurse-submodules git@github.com:HIRO-group/isaac_conveyor_and_robot_control.git
```

The `proto/` directory is a git submodule
([conveyor_indexing_protos](https://github.com/HIRO-group/conveyor_indexing_protos))
and requires HIRO-group access to pull.

### 3. Run the setup script

```bash
bash scripts/setup.sh
```

This generates the protobuf Python bindings and installs `eclipse-zenoh`
into Isaac Sim's bundled Python interpreter (used for camera publishing).

## What to run

```bash
DISPLAY=:0 bash scripts/run.sh
```

Useful environment variables:

- `CONVEYOR_INDEXING_HEADLESS=1` — skip the GUI viewport for faster
  data-collection runs.
- `CONVEYOR_INDEXING_RECORD=1` — record synchronized 30Hz training rows to
  `data/recordings/`.
- `CONVEYOR_INDEXING_RECORD_MCAP=1` — record every raw channel to
  `data/mcap/` with no episode concept.
- `ZENOH_ROUTER=tcp/127.0.0.1:7447` — connect to a running Zenoh router
  instead of opening a peer-to-peer session.

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
