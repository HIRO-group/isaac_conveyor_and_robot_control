#!/usr/bin/env bash
# Launch the sim with Isaac Sim's bundled python.
#   ISAAC_PYTHON   path to Isaac Sim's python.sh (required)
#   SIM_SCENE_DIR  scene package directory (required, or pass --scene)
#   PROTO_OUT      generated bindings (default: /tmp/proto_gen; see gen_proto.sh)
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${ISAAC_PYTHON:?set ISAAC_PYTHON to Isaac Sim's python.sh}"
PROTO_OUT="${PROTO_OUT:-/tmp/proto_gen}"

if ! "$ISAAC_PYTHON" -c "import zenoh" >/dev/null 2>&1; then
  echo "ERROR: eclipse-zenoh missing from Isaac Sim's python. Run: bash $REPO/scripts/setup.sh" >&2
  exit 1
fi

export PYTHONPATH="$REPO/src:$PROTO_OUT${PYTHONPATH:+:$PYTHONPATH}"
exec "$ISAAC_PYTHON" "$REPO/scripts/run_conveyor_indexing.py" "$@"
