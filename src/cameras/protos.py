"""Single place the generated camera protobuf bindings are imported from -
same convention as ``conveyor_indexing.protos``. Depends on
``sim_camera_pb2``, generated from ``proto/sim_camera.proto`` (see the
top-level README's "Setup" section) and made importable via ``PYTHONPATH``
(see ``scripts/run.sh``).
"""

from __future__ import annotations

import sim_camera_pb2 as camera

__all__ = ["camera"]
