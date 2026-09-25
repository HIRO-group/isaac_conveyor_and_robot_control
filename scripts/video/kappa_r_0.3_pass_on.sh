#!/usr/bin/env bash
# Take: arm 1's reach degraded to kappa_r = 0.3 and the controller knows it
# (the default reach mode, "decline"). The same single far box as
# kappa_r_0.3_attempt.sh rides in and stops at the far edge of arm 1's zone;
# arm 1 does not go for it, the zone releases it, and it rides on to arm 2,
# which picks and places it. Repeats MAX_WAVES times (one box per wave, the
# next spawned as soon as the previous one has left the spawn belt); zone 2
# holds a box that arrives while arm 2 is mid-cycle instead of overflowing it.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
TAKE="kappa_r=0.3 pass on"
export CONVEYOR_INDEXING_KAPPA_R_ARM1=0.3
export CONVEYOR_INDEXING_SPAWN_LAYOUT=far
export CONVEYOR_INDEXING_MAX_WAVES="${CONVEYOR_INDEXING_MAX_WAVES:-6}"
export CONVEYOR_INDEXING_HOLD_WHILE_BUSY=1
export CONVEYOR_INDEXING_BOX_VARIANT=21cm
launch "$@"
