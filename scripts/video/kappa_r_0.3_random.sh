#!/usr/bin/env bash
# Take: arm 1's reach degraded to kappa_r = 0.3 under ordinary random waves
# (1..4 boxes, random sizes and positions across the belt). Arm 1 serves the
# nearest 30 % of its zone: it picks whatever lands there, closest first,
# and once nothing it can reach is left the zone defers the rest to arm 2,
# which picks them. Zones hold through a cycle so a box is deferred for
# reach, not for the arm being busy. MAX_WAVES caps the take.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
TAKE="kappa_r=0.3 random waves"
export CONVEYOR_INDEXING_KAPPA_R_ARM1=0.3
export CONVEYOR_INDEXING_HOLD_WHILE_BUSY=1
export CONVEYOR_INDEXING_MAX_WAVES="${CONVEYOR_INDEXING_MAX_WAVES:-6}"
launch "$@"
