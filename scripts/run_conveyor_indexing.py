"""Zone-accumulation indexing over 5_conv_env.usd's two open conveyor lines,
with per-tick logging and a UR20 pick-and-place (cuMotion RMPflow) moving
boxes from loop 1 to loop 2 and into a waiting SteelBoxTruck.

Run via scripts/run.sh (sets PYTHONPATH to src/ and the generated protobuf
bindings) - see the top-level README's "Setup" section.
"""

from __future__ import annotations

import os

if __name__ == "__main__":
    # SimulationApp must be constructed before any omni.*/carb/isaacsim/
    # cumotion import - so every import below is deferred until here, inside
    # the guard, rather than living at module scope.
    from isaacsim import SimulationApp

    # Opt-in, not default: this repo's existing workflows (camera tuning, live
    # noVNC debugging - see README "Setup"/"Camera tuning") all assume a GUI
    # window, so headless stays behind an explicit env var rather than flipping
    # the default under them.
    headless = os.environ.get("CONVEYOR_INDEXING_HEADLESS", "") == "1"
    simulation_app = SimulationApp({"headless": headless})

    from sim_cell.log_setup import configure_logging
    from sim_cell.runner import run

    configure_logging()
    run(simulation_app)
