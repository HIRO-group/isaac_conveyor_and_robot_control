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

## Running a trained policy in closed loop

`CONVEYOR_INDEXING_EXTERNAL_ACTION=1` hands both arms + all 5 conveyor zones
to an external controller (e.g. a trained LeRobot checkpoint via theia's
`services/sim_bridge`) over Zenoh instead of the autonomous cuMotion/PackML
control - see `services/sim_bridge/README.md` in a sibling `theia` checkout
for the full wire contract (topics/message schemas). Nothing moves until a
real command arrives on that bus; there is no autonomous fallback while this
mode is on.

Start the three pieces **in this order** - Zenoh has no message replay, so a
subscriber only ever sees samples published after it subscribes. Starting
logging before the AI is what guarantees no early command/state transition
is missed:

**1. Start the sim** (external-action mode, GUI on display 0):

```bash
CONVEYOR_INDEXING_EXTERNAL_ACTION=1 DISPLAY=:0 bash scripts/run.sh
```

Wait for `sim_cell.runner: pick/place controllers ready, entering main loop`
in its output before continuing. Optionally add
`CONVEYOR_INDEXING_DEBUG_LOGGERS=pick_and_place.attachment,sim_cell.debug,conveyor_indexing.occupancy,conveyor_indexing.line_controller`
for verbose per-tick logging (box attach/detach with the exact box path,
pick-zone centering, occupancy hits, hold-zone state) - see
`src/sim_cell/log_setup.py` for the full list of loggers this can raise to
DEBUG. It's noisy; only turn it on when actively debugging.

**2. Start logging**, so it's already subscribed before the AI publishes
anything:

```bash
PYTHONPATH=/tmp/proto_gen /home/ubuntu/IsaacSim/python.sh \
  scripts/monitor_external_action.py
```

Confirm its `monitor: Zenoh session open ...` line has printed before moving
on. This prints every EE (suction) on/off toggle - both what was *commanded*
and what actually latched, which can differ if a box wasn't really in
reach - every conveyor command change, and a snapshot every 5s.

**3. Start the AI** (from a sibling `theia` checkout - see
`services/sim_bridge/README.md` there for full setup):

```bash
gsutil -m cp -r gs://por-theia-1/models/<checkpoint>/checkpoints/<step>/pretrained_model \
  ~/theia/services/sim_bridge/checkpoints/<checkpoint>/checkpoints/<step>/pretrained_model

cd ~/theia/services/sim_bridge
PYTHONPATH=/tmp/sim_bridge_proto_gen:$(pwd)/src .venv/bin/python3 src/main.py \
  --model-path checkpoints/<checkpoint>/checkpoints/<step>/pretrained_model \
  --num-arms 2 \
  --conveyors \
    "/World/ConveyorTrack/ConveyorBeltGraph/ConveyorNode" \
    "/World/ConveyorTrack_01/ConveyorBeltGraph/ConveyorNode" \
    "/World/ConveyorTrack_02/ConveyorBeltGraph/ConveyorNode" \
    "/World/ConveyorTrack_09/ConveyorBeltGraph/ConveyorNode" \
    "/World/ConveyorTrack_10/ConveyorBeltGraph/ConveyorNode" \
  --loop-hz 30
```

- The `--conveyors` order above is this repo's actual zone order (loop1's 3
  zones, then loop2's 2 - see `src/sim_cell/layout.py`'s
  `ZONE_NODE_PATHS_LOOP1`/`ZONE_NODE_PATHS_LOOP2`), which is also the order
  `theia/plc/state_conveyors` reports them in and what a checkpoint trained
  via this repo's recordings was trained with - get this wrong and the
  policy's actions get silently misaligned with the wrong conveyors.
- Only download `pretrained_model/`, not the sibling `training_state/` -
  that's optimizer state, not needed for inference, and is most of a
  checkpoint's size.
- `checkpoints/last` is a gcsfuse-only symlink `gsutil cp` can't follow; pick
  the highest-numbered `checkpoints/<step>` directory instead.
- Add `--conveyor-speed-multiplier <N>` (default 1.0) to rescale a
  checkpoint's raw predicted conveyor speed_pct before publishing (clamped to
  0-100) - a knob for compensating an under-scaled prediction empirically,
  not a training-data property.

To stop, kill all three in any order; re-running from step 1 is the safest
way to guarantee no stale state carries over.

## Local data collection

`scripts/collect_local.py` (stock `python3`, stdlib only) runs a headless
MCAP-recording sim and streams each closed `.mcap` file to
`gs://por-theia-1/data_collection/sim/<run_id>/instance_00/mcap/`, deleting
local copies after a verified upload so multi-hour runs never fill the disk:

```bash
# Short run, files also kept locally (e.g. to inspect in Foxglove):
python3 scripts/collect_local.py --sim-seconds 600 --keep-local

# Long unattended run (~0.23x realtime on an L4 - plan wall time accordingly):
nohup python3 scripts/collect_local.py --sim-seconds 14400 > /dev/null 2>&1 &
tail -f data/collect/<run_id>/collect.log
```

`--sweep-only --run-id <run_id>` uploads whatever a crashed/rebooted run left
behind. `scripts/foxglove_sim_layout.json` is a Foxglove layout (6 camera
panels, conveyor speeds, arm joints/phases) for viewing the recorded MCAPs.

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
