#!/usr/bin/env bash
# Take: arm 1's reach degraded to kappa_r = 0.3, but the controller does not
# know it (reach mode "attempt"). One box rides in on the conveyor and stops
# at the far edge of arm 1's pick zone, ~1.47 m from the base against a
# ~1.11 m reach. Arm 1 goes for it, stalls at its limit, retreats to the
# staging pose and goes again, for as long as the box is there - the
# fail-freeze contrast for kappa_r_0.3.sh, where the same box is passed on.
# MAX_WAVES=1 keeps the line to that one box; raise it to show the line
# backing up behind the stuck arm. One attempt is ~2 s of sim time at the
# default 1 s strain; REACH_STALL_S lengthens the strain.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
TAKE="kappa_r=0.3 attempt"
export CONVEYOR_INDEXING_KAPPA_R_ARM1=0.3
export CONVEYOR_INDEXING_REACH_MODE=attempt
export CONVEYOR_INDEXING_SPAWN_LAYOUT=far
export CONVEYOR_INDEXING_MAX_WAVES="${CONVEYOR_INDEXING_MAX_WAVES:-1}"
export CONVEYOR_INDEXING_HOLD_WHILE_BUSY=1
export CONVEYOR_INDEXING_REACH_STALL_S="${CONVEYOR_INDEXING_REACH_STALL_S:-1.0}"
export CONVEYOR_INDEXING_BOX_VARIANT=21cm
launch "$@"
