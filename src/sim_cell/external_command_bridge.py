"""Subscribes to external arm, conveyor and box commands (CONVEYOR_INDEXING_EXTERNAL_ACTION=1).

Keeps only the newest message per topic. Arm commands with a stale `seq` are
dropped: peer-to-peer Zenoh has been seen to deliver out of order under load.
"""

from __future__ import annotations

import json
import logging
import threading

from conveyor_indexing.protos import sim_action
from conveyor_indexing.topics import Topics
from conveyor_indexing.zenoh_session import open_session, payload_bytes
from sim_cell.protos import arm_action

logger = logging.getLogger(__name__)

BOX_OPS = ("auto", "clear", "spawn")


class ExternalCommandBridge:
    """`latest()` is the only hot-path call: a lock and a read, no Zenoh I/O."""

    def __init__(self, arms: tuple[int, ...] = (1, 2), topics: Topics | None = None) -> None:
        self.topics = topics or Topics.from_env()
        self._session = open_session()
        self._lock = threading.Lock()
        self._latest_arm: dict = {arm: None for arm in arms}
        self._latest_conveyors: sim_action.SimConveyorCommands | None = None
        self._box_commands: list = []

        self._arm_subs = {
            arm: self._session.declare_subscriber(self.topics.arm_action(arm), self._make_arm_handler(arm))
            for arm in arms
        }
        self._conveyor_sub = self._session.declare_subscriber(self.topics.conveyor_command, self._on_conveyors)
        self._box_sub = self._session.declare_subscriber(self.topics.boxes_command, self._on_box_command)
        logger.info("external-command subscribers ready for arms %s", list(arms))

    def _on_box_command(self, sample) -> None:
        # JSON: {"op": "auto", "enabled": bool} | {"op": "clear"} | {"op": "spawn", "x", "y", "yaw", "variant"}
        try:
            cmd = json.loads(payload_bytes(sample).decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as e:
            logger.warning("box command: bad payload (%s)", e)
            return
        if not isinstance(cmd, dict) or cmd.get("op") not in BOX_OPS:
            logger.warning("box command: unknown %r", cmd)
            return
        with self._lock:
            self._box_commands.append(cmd)

    def drain_box_commands(self) -> list:
        with self._lock:
            out, self._box_commands = self._box_commands, []
        return out

    def _make_arm_handler(self, arm: int):
        def _on_sample(sample) -> None:
            msg = arm_action.SimArmActionCommand()
            msg.ParseFromString(payload_bytes(sample))
            with self._lock:
                current = self._latest_arm[arm]
                if current is not None and msg.seq <= current.seq:
                    logger.warning("arm %d: dropping out-of-order command seq=%d (have %d)", arm, msg.seq, current.seq)
                    return
                self._latest_arm[arm] = msg

        return _on_sample

    def _on_conveyors(self, sample) -> None:
        msg = sim_action.SimConveyorCommands()
        msg.ParseFromString(payload_bytes(sample))
        with self._lock:
            self._latest_conveyors = msg

    def reset(self) -> None:
        """Forget cached commands so a stale one never drives a freshly entered external mode."""
        with self._lock:
            for arm in self._latest_arm:
                self._latest_arm[arm] = None
            self._latest_conveyors = None

    def latest(self):
        """(arm1_cmd, arm2_cmd, conveyor_cmds); any may be None before its first message."""
        with self._lock:
            return self._latest_arm.get(1), self._latest_arm.get(2), self._latest_conveyors

    def close(self) -> None:
        for sub in self._arm_subs.values():
            sub.undeclare()
        self._conveyor_sub.undeclare()
        self._box_sub.undeclare()
        self._session.close()
        logger.info("external-command Zenoh session closed")
