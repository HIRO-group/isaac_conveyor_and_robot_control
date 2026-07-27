#!/usr/bin/env bash
# Launch the conveyor indexing sim with Isaac Sim's bundled python.
#
# Usage:
#   DISPLAY=:0 bash scripts/run.sh
#
# Requires the protobuf Python bindings already generated (see gen_proto.sh)
# at /tmp/proto_gen, and eclipse-zenoh installed into Isaac Sim's bundled
# python (required for camera publishing) - run scripts/setup.sh for both.
# See the top-level README's "Setup" section.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Cheap pre-flight check - catches a missing dependency immediately instead
# of paying a full Isaac Sim startup first (cameras.zenoh_publisher's own
# SystemExit is the second line of defense, in case this is bypassed).
if ! /home/ubuntu/IsaacSim/python.sh -c "import zenoh" >/dev/null 2>&1; then
  echo "ERROR: eclipse-zenoh missing from Isaac Sim's python. Run: bash $REPO/scripts/setup.sh" >&2
  exit 1
fi

export PYTHONPATH="$REPO/src:/tmp/proto_gen${PYTHONPATH:+:$PYTHONPATH}"
exec /home/ubuntu/IsaacSim/python.sh "$REPO/scripts/run_conveyor_indexing.py" "$@"
