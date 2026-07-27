#!/usr/bin/env bash
# Launch the conveyor indexing sim with Isaac Sim's bundled python.
#
# Usage:
#   DISPLAY=:0 bash scripts/run.sh
#
# Requires the protobuf Python bindings already generated (see gen_proto.sh)
# at /tmp/proto_gen - see the top-level README's "Setup" section.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$REPO/src:/tmp/proto_gen${PYTHONPATH:+:$PYTHONPATH}"
exec /home/ubuntu/IsaacSim/python.sh "$REPO/scripts/run_conveyor_indexing.py" "$@"
