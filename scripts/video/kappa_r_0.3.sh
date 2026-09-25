#!/usr/bin/env bash
# Take: arm 1's reach degraded to kappa_r = 0.3 (it serves the nearest 30 % of
# its pick zone). Same near/far pairs as kappa_r_1.0.sh: arm 1 picks the near
# box; the far box is out of its reach, so the zone releases it and it passes
# on to arm 2. Repeats until the box pool runs dry.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
TAKE="kappa_r=0.3"
export CONVEYOR_INDEXING_KAPPA_R_ARM1=0.3
export CONVEYOR_INDEXING_SPAWN_LAYOUT=near_far
export CONVEYOR_INDEXING_HOLD_WHILE_BUSY=1
# Small (21 cm) boxes only, so every pair reads the same on camera.
export CONVEYOR_INDEXING_BOX_VARIANT=21cm
launch "$@"
