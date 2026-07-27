"""Workaround for an Isaac Sim bug (filed upstream): `WorldBinding.initialize()`
resets xform ops on every tracked Mesh obstacle, dropping its rotation.
"""

from __future__ import annotations

from contextlib import contextmanager

from pxr import Gf, Usd, UsdGeom

import isaacsim.core.experimental.utils.stage as stage_utils


@contextmanager
def preserve_obstacle_rotations(paths: list):
    """Snapshot each prim in `paths`'s local rotation on entry, restore it on exit -
    wrap any `WorldBinding.initialize()` call in this.
    """
    stage = stage_utils.get_current_stage()
    snapshot = {}
    for guard_path in paths:
        local_matrix = UsdGeom.Xformable(stage.GetPrimAtPath(guard_path)).GetLocalTransformation(Usd.TimeCode.Default())
        snapshot[guard_path] = Gf.Transform(local_matrix).GetRotation().GetQuat()

    try:
        yield
    finally:
        for guard_path, original_local_quat in snapshot.items():
            guard_prim = stage.GetPrimAtPath(guard_path)
            if guard_prim.IsValid() and "xformOp:orient" in guard_prim.GetPropertyNames():
                # Match the attribute's actual authored precision (Quatf vs Quatd) -
                # Set() raises a type-mismatch error otherwise on single-precision prims.
                orient_attr = guard_prim.GetAttribute("xformOp:orient")
                quat_type = type(orient_attr.Get()) if orient_attr.Get() is not None else Gf.Quatd
                orient_attr.Set(quat_type(original_local_quat))
