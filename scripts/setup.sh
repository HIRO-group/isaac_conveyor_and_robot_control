#!/usr/bin/env bash
# One-time setup: generate protobuf bindings and install the sim's python deps into Isaac Sim's python.
#   ISAAC_PYTHON  path to Isaac Sim's python.sh (required)
#   PROTO_OUT     generated bindings directory (default: /tmp/proto_gen)
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${ISAAC_PYTHON:?set ISAAC_PYTHON to Isaac Sim's python.sh}"

"$ISAAC_PYTHON" -m pip install -r "$REPO/requirements.txt"
PROTOC="$ISAAC_PYTHON -m grpc_tools.protoc" bash "$REPO/gen_proto.sh"

echo "Setup complete. Run the sim with:"
echo "  SIM_SCENE_DIR=/path/to/scene bash $REPO/scripts/run.sh"
