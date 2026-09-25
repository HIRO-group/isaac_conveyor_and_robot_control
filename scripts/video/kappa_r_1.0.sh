#!/usr/bin/env bash
# Take: arm 1 at full reach (kappa_r = 1.0), the contrast for kappa_r_0.3.sh.
# Every wave is a pair: one box at arm 1's belt edge, one at the far edge. The
# pick zone holds both through arm 1's cycle, so arm 1 picks the near box and
# then the far one. Repeats until the box pool runs dry.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
TAKE="kappa_r=1.0"
export CONVEYOR_INDEXING_KAPPA_R_ARM1=1.0
export CONVEYOR_INDEXING_SPAWN_LAYOUT=near_far
export CONVEYOR_INDEXING_HOLD_WHILE_BUSY=1
launch "$@"
