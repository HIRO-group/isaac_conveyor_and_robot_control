#!/usr/bin/env bash
# Take: both arms nominal but hesitant, like a cautious learned policy. Each
# arm starts at most PICKS_PER_MIN cycles a minute (two: one every 30 s,
# waiting at its staging pose in between) and on every descent freezes
# HESITATIONS times, one to three seconds each, before going on to the box.
# Ordinary random waves; the fault seed fixes where the stops fall.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
TAKE="hesitant, 2 picks/min"
export CONVEYOR_INDEXING_PICKS_PER_MIN="${CONVEYOR_INDEXING_PICKS_PER_MIN:-2}"
export CONVEYOR_INDEXING_HESITATIONS="${CONVEYOR_INDEXING_HESITATIONS:-2}"
launch "$@"
