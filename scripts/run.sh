#!/usr/bin/env bash
# Launch the sim with Isaac Sim's bundled python.
#   ISAAC_PYTHON   path to Isaac Sim's python.sh (required)
#   SIM_SCENE_DIR  scene package directory (required, or pass --scene)
#   PROTO_OUT      generated bindings (default: /tmp/proto_gen; see gen_proto.sh)
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${ISAAC_PYTHON:?set ISAAC_PYTHON to the Isaac Sim python.sh}"
PROTO_OUT="${PROTO_OUT:-/tmp/proto_gen}"

if ! "$ISAAC_PYTHON" -c "import zenoh" >/dev/null 2>&1; then
  echo "ERROR: eclipse-zenoh missing from Isaac Sim's python. Run: bash $REPO/scripts/setup.sh" >&2
  exit 1
fi

# /tmp/proto_gen is wiped on reboot: regenerate the bindings when any is
# missing or older than its schema, instead of failing inside Isaac Sim.
for schema in "$REPO"/proto/*.proto "$REPO"/proto/foxglove/*.proto; do
  rel="${schema#"$REPO"/proto/}"
  gen="$PROTO_OUT/${rel%.proto}_pb2.py"
  if [ ! -f "$gen" ] || [ "$schema" -nt "$gen" ]; then
    echo "generating protobuf bindings in $PROTO_OUT"
    PROTOC="$ISAAC_PYTHON -m grpc_tools.protoc" PROTO_OUT="$PROTO_OUT" bash "$REPO/gen_proto.sh"
    break
  fi
done

export PYTHONPATH="$REPO/src:$PROTO_OUT${PYTHONPATH:+:$PYTHONPATH}"
exec "$ISAAC_PYTHON" "$REPO/scripts/run_conveyor_indexing.py" "$@"
