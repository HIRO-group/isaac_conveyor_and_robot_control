#!/usr/bin/env bash
# Generate Python protobuf bindings needed by scripts/run_conveyor_indexing.py
# (via conveyor_indexing.protos / cameras.protos):
#   - theia's real plc-connector.proto + common/types.proto (schema for state)
#   - this directory's sim_conveyor_action.proto (schema for actions)
#   - this directory's sim_camera.proto (schema for cameras - wire-compatible
#     with theia's real camera.proto, but generated from this repo's own copy;
#     see proto/sim_camera.proto's header comment for why)
#
# Also generated, needed only when CONVEYOR_INDEXING_RECORD_MCAP=1 (see
# conveyor_indexing.mcap_recorder):
#   - theia's real robot.proto + foxglove/raw_image.proto (MCAP channel
#     schemas mirroring theia's real data-collection service, so
#     mcap_to_lerobot.py routes sim and real captures identically)
#   - this directory's sim_state.proto (sim-only ground truth: box
#     states/events, arm phase transitions, run metadata)
#
# Usage:
#   bash /home/ubuntu/conveyor_indexing/gen_proto.sh
#
# Requires a protoc whose codegen matches the `protobuf` Python package that
# will import the result (Isaac Sim's bundled python.sh has protobuf 7.35.1
# as of this writing). apt's protoc (3.12.4 on Ubuntu 22.04) is too old for
# that - it emits descriptor code that protobuf>=4's upb backend rejects with
# "Descriptors cannot be created directly". Use `python3 -m grpc_tools.protoc`
# instead (`pip install grpcio-tools`), which bundles a protoc version that
# matches modern protobuf runtimes.

set -e

PROTOC="python3 -m grpc_tools.protoc"
PROTO_OUT=/tmp/proto_gen
THEIA_ROOT=/home/ubuntu/theia
SIM_PROTO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/proto" && pwd)"

mkdir -p "$PROTO_OUT"

$PROTOC \
  -I "$THEIA_ROOT/proto/plc-connector" \
  -I "$THEIA_ROOT/proto" \
  --python_out="$PROTO_OUT" \
  "$THEIA_ROOT/proto/plc-connector/plc-connector.proto" \
  "$THEIA_ROOT/proto/common/types.proto"

touch "$PROTO_OUT/common/__init__.py"

$PROTOC \
  -I "$SIM_PROTO_DIR" \
  --python_out="$PROTO_OUT" \
  "$SIM_PROTO_DIR/sim_conveyor_action.proto"

$PROTOC \
  -I "$SIM_PROTO_DIR" \
  --python_out="$PROTO_OUT" \
  "$SIM_PROTO_DIR/sim_camera.proto"

# MCAP-only bindings (conveyor_indexing.mcap_recorder). robot.proto and
# foxglove/raw_image.proto have no cross-imports, so each compiles standalone
# with just $THEIA_ROOT/proto on the include path.
$PROTOC \
  -I "$THEIA_ROOT/proto" \
  --python_out="$PROTO_OUT" \
  "$THEIA_ROOT/proto/robot/robot.proto" \
  "$THEIA_ROOT/proto/foxglove/raw_image.proto"

touch "$PROTO_OUT/robot/__init__.py" "$PROTO_OUT/foxglove/__init__.py"

$PROTOC \
  -I "$SIM_PROTO_DIR" \
  --python_out="$PROTO_OUT" \
  "$SIM_PROTO_DIR/sim_state.proto"

echo "Proto generated at $PROTO_OUT"
echo "scripts/run.sh already puts $PROTO_OUT on PYTHONPATH - just run:"
echo "  DISPLAY=:0 bash /home/ubuntu/conveyor_indexing/scripts/run.sh"
