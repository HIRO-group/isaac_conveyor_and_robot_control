"""Single place the generated protobuf bindings for the live external-control
channels are imported from - same convention as ``conveyor_indexing.protos``
and ``cameras.protos``. Depends on ``sim_robot_state_pb2`` / ``sim_arm_action_pb2``,
generated from ``proto/sim_robot_state.proto`` / ``proto/sim_arm_action.proto``
(see gen_proto.sh) and made importable via ``PYTHONPATH`` (see ``scripts/run.sh``).
"""

from __future__ import annotations

import sim_arm_action_pb2 as arm_action
import sim_robot_state_pb2 as robot_state
import sim_state_pb2 as sim_state

__all__ = ["arm_action", "robot_state", "sim_state"]
