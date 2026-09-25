#!/usr/bin/env bash
# Take: LIVE INFERENCE of the full typed-graph policy (curriculum_v4_graph_hybrid_base,
# stage2_failure checkpoint) under arm-1 reach 0.3, through the paper's
# continuous-run harness (capability-diffusion/scripts/eval_continuous_live.sh:
# external-action sim, waves of 4..8, work-gated stall reset, rest start pose).
# The policy runs on the wall clock, so the sim is paced to 1.0x real time
# (the evaluations ran unthrottled at ~1.1x); the log prints the achieved
# factor every 5 s and it must stay at 1.00, or the take is not the
# evaluated behaviour. Frames: VIDEO_FRAMES_DIR as for the other takes.
#   DURATION_S=120 KAPPA=0.3 VIDEO_FRAMES_DIR=... bash scripts/video/live_graph_r0.3.sh
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
CD="${CD:-$(dirname "$REPO")/capability-diffusion}"
export SIM="$REPO"
export CKPT_DIR="${CKPT_DIR:-$CD/docs/results/a100final-20260911/curriculum_v4_graph_hybrid_base}"
export KAPPA="${KAPPA:-0.3}"
export DURATION_S="${DURATION_S:-180}"
export SPAWN_SEED="${SPAWN_SEED:-930001}"
export CONVEYOR_INDEXING_REALTIME="${CONVEYOR_INDEXING_REALTIME:-1.0}"
export RUN_DIR="${RUN_DIR:-data/video_live_graph_r${KAPPA}}"
echo "take: live graph policy (kappa_r_arm1=$KAPPA duration=${DURATION_S}s realtime=$CONVEYOR_INDEXING_REALTIME frames=${VIDEO_FRAMES_DIR:-off})"
exec bash "$CD/scripts/eval_continuous_live.sh"
