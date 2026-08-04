"""Subscribes to externally-supplied per-arm action commands and conveyor
commands over Zenoh, for CONVEYOR_INDEXING_EXTERNAL_ACTION mode (see
sim_cell.runner) - the live counterpart of the autonomous
pick_and_place.controller.MagicAttachPickPlace / ConveyorLineController
control this mode bypasses.

Only ever keeps the most recently received message per topic (never a
backlog - only the newest command ever matters for a PD-drive-style control
loop), guarded by one lock since Zenoh delivers subscriber callbacks off the
main sim-loop thread.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import ClassVar

from conveyor_indexing.protos import sim_action
from sim_cell.protos import arm_action

logger = logging.getLogger(__name__)

try:
    import zenoh
except ImportError as exc:
    raise SystemExit(
        "eclipse-zenoh is required for CONVEYOR_INDEXING_EXTERNAL_ACTION mode but is not "
        "installed in this interpreter. Install it into Isaac Sim's bundled python:\n"
        "  /home/ubuntu/IsaacSim/python.sh -m pip install eclipse-zenoh==1.7.1\n"
        "(or run scripts/setup.sh, which does this for you - see the "
        "top-level README's 'Setup' section)."
    ) from exc


def _open_session() -> zenoh.Session:
    conf = zenoh.Config()
    router = os.environ.get("ZENOH_ROUTER")
    if router:
        conf.insert_json5("connect/endpoints", f'["{router}"]')
        logger.info("connecting to Zenoh router at %s", router)
    else:
        logger.warning("ZENOH_ROUTER not set; opening Zenoh session in peer-to-peer mode")
    return zenoh.open(conf)


def _payload_bytes(sample) -> bytes:
    payload = sample.payload
    return payload.to_bytes() if hasattr(payload, "to_bytes") else bytes(payload)


class ExternalCommandBridge:
    """Owns one Zenoh session subscribing to the externally-driven arm/conveyor
    command topics. `latest()` is the only thing sim_cell.runner needs to call,
    once per physics tick - a cheap lock+read, no Zenoh I/O on the hot path.
    """

    ARM_TOPICS: ClassVar[dict] = {1: "sim/arm/1/action_command", 2: "sim/arm/2/action_command"}
    CONVEYOR_TOPIC = "sim/conveyor/command"

    def __init__(self) -> None:
        self._session = _open_session()
        self._lock = threading.Lock()
        self._latest_arm: dict = {1: None, 2: None}
        self._latest_conveyors: sim_action.SimConveyorCommands | None = None

        self._arm_subs = {
            arm: self._session.declare_subscriber(topic, self._make_arm_handler(arm))
            for arm, topic in self.ARM_TOPICS.items()
        }
        self._conveyor_sub = self._session.declare_subscriber(self.CONVEYOR_TOPIC, self._on_conveyors)
        logger.info(
            "external-command subscribers ready: %s, %s", list(self.ARM_TOPICS.values()), self.CONVEYOR_TOPIC
        )

    def _make_arm_handler(self, arm: int):
        def _on_sample(sample) -> None:
            msg = arm_action.SimArmActionCommand()
            msg.ParseFromString(_payload_bytes(sample))
            with self._lock:
                self._latest_arm[arm] = msg

        return _on_sample

    def _on_conveyors(self, sample) -> None:
        msg = sim_action.SimConveyorCommands()
        msg.ParseFromString(_payload_bytes(sample))
        with self._lock:
            self._latest_conveyors = msg

    def latest(self):
        """Returns (arm1_cmd, arm2_cmd, conveyor_cmds) - any may be None if
        nothing has been received yet on that topic.
        """
        with self._lock:
            return self._latest_arm[1], self._latest_arm[2], self._latest_conveyors

    def close(self) -> None:
        for sub in self._arm_subs.values():
            sub.undeclare()
        self._conveyor_sub.undeclare()
        self._session.close()
        logger.info("external-command Zenoh session closed")
