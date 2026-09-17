#!/usr/bin/env bash
# One-time setup for this repo: generates the protobuf Python bindings and
# installs eclipse-zenoh (camera publishing, see src/cameras/zenoh_publisher.py)
# and warp-lang (GPU camera-capture path, see src/cameras/rig.py) into Isaac
# Sim's bundled python - see the top-level README's "Setup" section.
#
# Usage:
#   bash /home/ggbrisco/isaac_conveyor_and_robot_control/scripts/setup.sh
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ISAAC_PYTHON=/home/ggbrisco/isaacsim/_build/linux-x86_64/release/python.sh

bash "$REPO/gen_proto.sh"

# Pinned to match theia's own pin (~theia/data_collection/requirements.txt) -
# coincidental alignment, not a dependency: this keeps the wire/API version
# this sim publishes with in step with what theia's collectors expect.
"$ISAAC_PYTHON" -m pip install eclipse-zenoh==1.7.1

# Not bundled by Isaac Sim's launcher python by default; cameras.rig falls
# back to (slower) CPU capture without it, logged once, not fatal.
"$ISAAC_PYTHON" -m pip install warp-lang

# Required unconditionally: sim_cell.cell imports conveyor_indexing.mcap_recorder
# (and its protobuf-generated modules) at module load time regardless of
# whether CONVEYOR_INDEXING_RECORD_MCAP is set.
"$ISAAC_PYTHON" -m pip install protobuf==7.36.1 mcap mcap-protobuf-support pyarrow

echo "Setup complete. Run the sim with:"
echo "  DISPLAY=:1 bash $REPO/scripts/run.sh"
