"""NumPy compatibility shim - must be imported before `cumotion`.

`isaacsim.robot_motion.cumotion` calls `np.reshape(arr, shape=[...])`, a NumPy
2.1+ kwarg not present in this install's bundled NumPy 1.26.4. Patched here,
process-local only. Must capture the real `np.reshape` before reassigning it
below, or the wrapper recurses.
"""

from __future__ import annotations

import numpy as np

_np_reshape = np.reshape


def _reshape_shape_kwarg_compat(a, *args, **kwargs):
    if "shape" in kwargs:
        kwargs["newshape"] = kwargs.pop("shape")
    return _np_reshape(a, *args, **kwargs)


np.reshape = _reshape_shape_kwarg_compat
