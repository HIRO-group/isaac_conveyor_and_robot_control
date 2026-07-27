"""Conveyor zone-accumulation indexing: state machine, occupancy sensing, and
per-tick logging, independent of any particular scene.

Deliberately import-light: every submodule here imports `omni`/`isaacsim`/
`carb` (directly or via `pxr`) at module scope, which requires a
`SimulationApp` to already be constructed (see `scripts/run_conveyor_indexing.py`).
Importing this package itself must stay side-effect-free; import submodules
directly (e.g. `from conveyor_indexing.state_machine import ...`) once the app
is up, not via a re-export here.
"""
