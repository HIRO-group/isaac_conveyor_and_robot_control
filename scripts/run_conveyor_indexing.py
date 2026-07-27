"""Zone-accumulation indexing over 5_conv_env.usd's two open conveyor lines,
with per-tick logging and a UR20 pick-and-place (cuMotion RMPflow) moving
boxes from loop 1 to loop 2 and into a waiting SteelBoxTruck.

Run via scripts/run.sh (sets PYTHONPATH to src/ and the generated protobuf
bindings) - see the top-level README's "Setup" section.
"""

from __future__ import annotations

if __name__ == "__main__":
    # SimulationApp must be constructed before any omni.*/carb/isaacsim/cumotion
    # import - so every import below is deferred until here, inside the guard,
    # rather than living at module scope.
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": False})

    from sim_cell.log_setup import configure_logging
    from sim_cell.runner import run

    configure_logging()
    run(simulation_app)
