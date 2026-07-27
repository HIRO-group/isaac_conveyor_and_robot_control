"""Sim cameras: renders camera prims via Replicator and publishes frames over
Zenoh on theia's wire contract (see this package's ``zenoh_publisher`` module
and the top-level README's "Design" section for the exact contract and why
this repo mirrors it locally instead of depending on theia).

Importing this package itself must stay side-effect-free; import submodules
directly (`from cameras.rig import CameraRig`) once the app is up, not via a
re-export here - `rig.py` imports `omni`/`pxr`, which require a live
`SimulationApp`, same convention as `conveyor_indexing`/`sim_cell`.
"""
