#!/usr/bin/env bash
# Take: arm 1's hold degraded (kappa_h = 0.3 on the record). Ordinary random
# waves; for the clip EVERY pick drops, 30-50 % of the way through the swing
# from the pick belt to the place belt, and arm 1 finishes the cycle
# empty-handed. The fault seed fixes where in the swing each drop lands.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
TAKE="kappa_h=0.3"
export CONVEYOR_INDEXING_KAPPA_H_ARM1=0.3
export CONVEYOR_INDEXING_DROP_EVERY_HOLD=1
launch "$@"
