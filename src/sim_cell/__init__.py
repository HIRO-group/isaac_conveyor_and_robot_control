"""Wiring for this specific cell: `5_conv_env.usd`'s layout, this run's
tuning, stage setup, and the main control loop. Depends on both
`conveyor_indexing` and `pick_and_place`, which depend on neither it nor
each other.

Deliberately import-light, like `conveyor_indexing` (see that package's
docstring) - a `SimulationApp` must already be constructed before any
submodule here is imported.
"""
