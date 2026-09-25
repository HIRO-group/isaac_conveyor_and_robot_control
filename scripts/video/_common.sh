#!/usr/bin/env bash
# Shared launcher for the video takes in this directory. Hardcoded for this
# machine (Isaac python, the dual-arm scene) but every value can be overridden
# from the environment. Each kappa_*.sh sets one fault and calls `launch`.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export ISAAC_PYTHON="${ISAAC_PYTHON:-/home/ggbrisco/isaacsim/_build/linux-x86_64/release/python.sh}"
export SIM_SCENE_DIR="${SIM_SCENE_DIR:-/media/ggbrisco/data/theia/sim/scenes/dual_arm_cell}"
# Same box sequence and the same drop decisions on every take.
export CONVEYOR_INDEXING_SPAWN_SEED="${CONVEYOR_INDEXING_SPAWN_SEED:-4242}"
export CONVEYOR_INDEXING_FAULT_SEED="${CONVEYOR_INDEXING_FAULT_SEED:-4242}"
# Frame-accurate capture (preferred over Movie Capture or a screen recorder):
# set VIDEO_FRAMES_DIR and every viewport render is written as a PNG, 30 per
# sim second, then `bash scripts/video/encode_frames.sh <run dir> 30 out.mp4`.
if [ -n "${VIDEO_FRAMES_DIR:-}" ]; then
  export CONVEYOR_INDEXING_VIEWPORT_FRAMES_DIR="$VIDEO_FRAMES_DIR"
fi
# Pace sim time to the wall clock so a screen recording of the viewport plays
# at a constant, known speed. Headed with the viewport this machine renders at
# ~0.68x real time (measured 2026-09-20), so 0.6 leaves headroom; the log
# prints the achieved factor every 5 s and it must read exactly the target,
# or the machine is not keeping up and the speed will drift between takes.
export CONVEYOR_INDEXING_REALTIME="${CONVEYOR_INDEXING_REALTIME:-0.6}"

launch() {
  echo "take: ${TAKE:-?} (kappa_r_arm1=${CONVEYOR_INDEXING_KAPPA_R_ARM1:-1} reach_mode=${CONVEYOR_INDEXING_REACH_MODE:-decline} kappa_h_arm1=${CONVEYOR_INDEXING_KAPPA_H_ARM1:-1} kappa_c=${CONVEYOR_INDEXING_KAPPA_C:-1} picks_per_min=${CONVEYOR_INDEXING_PICKS_PER_MIN:-inf} hesitations=${CONVEYOR_INDEXING_HESITATIONS:-0} layout=${CONVEYOR_INDEXING_SPAWN_LAYOUT:-waves} boxes=${CONVEYOR_INDEXING_BOX_VARIANT:-all})"
  exec bash "$REPO/scripts/run.sh" "$@"
}
