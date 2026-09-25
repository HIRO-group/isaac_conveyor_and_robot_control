#!/usr/bin/env bash
# Take: the pick conveyors (loop 1, the three belts feeding both pick zones) at
# kappa_c = 0.5: half their nominal speed. The outfeed loop stays nominal.
# Ordinary random waves.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
TAKE="kappa_c=0.5"
export CONVEYOR_INDEXING_KAPPA_C=0.5
launch "$@"
