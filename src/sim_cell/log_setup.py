"""Logging setup for the sim - deliberately not `logging.basicConfig`.

Kit reconfigures the root logger and routes it into carb's own log system,
so relying on root propagation is not dependable; instead, each of this
repo's three package-root loggers gets its own stdout handler.
"""

from __future__ import annotations

import logging
import os
import sys

_PACKAGE_LOGGERS = ("conveyor_indexing", "pick_and_place", "sim_cell")

# Comma-separated logger names to raise to DEBUG, e.g.
#   CONVEYOR_INDEXING_DEBUG_LOGGERS=conveyor_indexing.occupancy,sim_cell.debug
# Replaces the old DEBUG_LOG_OCCUPANCY_HITS / DEBUG_LOG_HOLD_ZONE_STATE flags:
#   - conveyor_indexing.occupancy         (was DEBUG_LOG_OCCUPANCY_HITS)
#   - conveyor_indexing.line_controller   (was DEBUG_LOG_HOLD_ZONE_STATE)
#   - sim_cell.debug                      (the tick%3 diagnostic dump)
_DEBUG_LOGGERS_ENV_VAR = "CONVEYOR_INDEXING_DEBUG_LOGGERS"


def configure_logging(level: int = logging.INFO, debug_loggers: list = None) -> None:
    """Attach one stdout StreamHandler (flushing per record, like the old
    `print(..., flush=True)` calls) to each package-root logger, at `level`.
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(name)s] %(message)s"))

    for name in _PACKAGE_LOGGERS:
        pkg_logger = logging.getLogger(name)
        pkg_logger.setLevel(level)
        pkg_logger.addHandler(handler)
        pkg_logger.propagate = False

    env_value = os.environ.get(_DEBUG_LOGGERS_ENV_VAR, "")
    names = [n.strip() for n in env_value.split(",") if n.strip()]
    names.extend(debug_loggers or [])
    for name in names:
        logging.getLogger(name).setLevel(logging.DEBUG)
