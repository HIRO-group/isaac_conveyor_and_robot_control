#!/usr/bin/env bash
# One-time setup for this repo: generates the protobuf Python bindings and
# installs eclipse-zenoh into Isaac Sim's bundled python (required for camera
# publishing - see src/cameras/zenoh_publisher.py and the top-level README's
# "Setup" section).
#
# Usage:
#   bash /home/ubuntu/conveyor_indexing/scripts/setup.sh
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

bash "$REPO/gen_proto.sh"

# Pinned to match theia's own pin (~theia/data_collection/requirements.txt) -
# coincidental alignment, not a dependency: this keeps the wire/API version
# this sim publishes with in step with what theia's collectors expect.
/home/ubuntu/IsaacSim/python.sh -m pip install eclipse-zenoh==1.7.1

echo "Setup complete. Run the sim with:"
echo "  DISPLAY=:0 bash $REPO/scripts/run.sh"
