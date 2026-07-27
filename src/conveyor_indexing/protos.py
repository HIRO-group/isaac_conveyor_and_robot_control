"""Single place the generated protobuf bindings are imported from.

Depends on ``plc_connector_pb2`` / ``sim_conveyor_action_pb2`` / (``common.``)
``types_pb2``, generated from theia's real proto (see the top-level README's
"Setup" section for the generation step) and made importable via
``PYTHONPATH`` (see ``scripts/run.sh``) - not via a relative/package import,
since these are flat top-level module names off the generated output
directory, not part of this package.
"""

from __future__ import annotations

import plc_connector_pb2 as plc
import sim_conveyor_action_pb2 as sim_action

try:
    from common import types_pb2 as common_types
except ModuleNotFoundError:
    import types_pb2 as common_types

__all__ = ["plc", "sim_action", "common_types"]
