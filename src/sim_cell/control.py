"""Runtime control of who drives the cell: the built-in state machine or an external controller.

`ControlChannel` is the Zenoh side (`sim/control` requests, latched `sim/status`);
`ModeController` applies a mode switch to the running cell.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

from conveyor_indexing.topics import Topics
from conveyor_indexing.zenoh_session import open_session, payload_bytes

logger = logging.getLogger(__name__)


class ControlMode(str, Enum):
    AUTONOMOUS = "autonomous"  # sim_cell's own pick-and-place + conveyor indexing
    EXTERNAL = "external"  # arms and belts driven over Zenoh (sim/arm/<n>/action_command, sim/conveyor/command)


@dataclass(frozen=True)
class ControlRequest:
    op: str  # set_mode | status
    mode: ControlMode | None = None


def parse_request(payload: bytes) -> ControlRequest:
    """JSON: {"op": "set_mode", "mode": "autonomous"|"external"} or {"op": "status"}."""
    try:
        data = json.loads(payload.decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as exc:
        raise ValueError(f"control request is not JSON: {exc}") from exc
    if not isinstance(data, dict) or data.get("op") not in ("set_mode", "status"):
        raise ValueError(f"unknown control request {data!r}")
    mode = None
    if data["op"] == "set_mode":
        try:
            mode = ControlMode(data.get("mode"))
        except ValueError as exc:
            raise ValueError(f"unknown mode {data.get('mode')!r}") from exc
    return ControlRequest(data["op"], mode)


class ControlChannel:
    """Zenoh queryable + subscriber on the control key; replies with the latest status JSON."""

    def __init__(self, initial_mode: ControlMode, topics: Topics | None = None) -> None:
        self.topics = topics or Topics.from_env()
        self._session = open_session()
        self._lock = threading.Lock()
        self._pending: ControlMode | None = None
        self._status: dict = {"mode": initial_mode.value, "sim_time_us": 0}
        self._status_pub = self._session.declare_publisher(self.topics.status)
        self._status_queryable = self._session.declare_queryable(self.topics.status, self._reply_status)
        self._control_queryable = self._session.declare_queryable(self.topics.control, self._on_query)
        self._control_sub = self._session.declare_subscriber(self.topics.control, self._on_sample)
        self._status_pub.put(self.status_bytes())
        logger.info("control channel on %s, status on %s", self.topics.control, self.topics.status)

    # -- inbound --------------------------------------------------------------

    def _handle(self, payload: bytes) -> str | None:
        try:
            req = parse_request(payload)
        except ValueError as exc:
            logger.warning("control: %s", exc)
            return str(exc)
        if req.op == "set_mode":
            with self._lock:
                self._pending = req.mode
            logger.info("control: mode %s requested", req.mode.value)
        return None

    def _on_query(self, query) -> None:
        error = self._handle(payload_bytes(query)) if query.payload is not None else None
        body = dict(self._status, error=error) if error else self._status
        query.reply(self.topics.control, json.dumps(body).encode("utf-8"))

    def _on_sample(self, sample) -> None:
        self._handle(payload_bytes(sample))

    def _reply_status(self, query) -> None:
        query.reply(self.topics.status, self.status_bytes())

    # -- runner side ----------------------------------------------------------

    def take_pending(self) -> ControlMode | None:
        with self._lock:
            mode, self._pending = self._pending, None
        return mode

    def publish_status(self, mode: ControlMode, sim_time_us: int, **extra) -> None:
        with self._lock:
            self._status = {"mode": mode.value, "sim_time_us": sim_time_us, **extra}
        self._status_pub.put(self.status_bytes())

    def status_bytes(self) -> bytes:
        with self._lock:
            return json.dumps(self._status).encode("utf-8")

    def close(self) -> None:
        self._control_sub.undeclare()
        self._control_queryable.undeclare()
        self._status_queryable.undeclare()
        self._status_pub.undeclare()
        self._session.close()


@dataclass
class ModeController:
    """Applies a control mode to the live cell and tracks external-mode arm state.

    `stations` maps arm -> (line controller, pick zone index, pick-and-place controller).
    `release` detaches an externally held box: release(arm, held_box_path) -> None.
    `block_external` returns a reason string when external mode must be refused (e.g. the
    episode recorder is on), else None.

    Autonomous-mode hold-zone readiness is "the arm is WAITING", with two demonstration
    hooks (sim_cell.faults): `hold_while_busy` keeps the zone holding through the arm's
    cycle, and `defer_checks[arm]()` returning True vetoes readiness so the zone passes its
    boxes downstream (the arm cannot serve what is there).
    """

    stations: dict
    bridge: object  # ExternalCommandBridge (needs .reset())
    release: Callable[[int, str], None]
    mode: ControlMode = ControlMode.AUTONOMOUS
    block_external: Callable[[], str | None] = lambda: None
    hold_while_busy: bool = False
    defer_checks: dict = field(default_factory=dict)  # arm -> () -> bool
    held: dict = field(default_factory=dict)  # arm -> held box path in external mode
    holding: dict = field(default_factory=dict)  # arm -> bool, for hold-zone readiness

    def __post_init__(self) -> None:
        for arm in self.stations:
            self.held.setdefault(arm, None)
            self.holding.setdefault(arm, False)
        self._install(self.mode)

    @property
    def external(self) -> bool:
        return self.mode is ControlMode.EXTERNAL

    def apply(self, mode: ControlMode) -> bool:
        """Switch modes. Returns False (and logs) when the switch is refused."""
        if mode is self.mode:
            return True
        if mode is ControlMode.EXTERNAL:
            reason = self.block_external()
            if reason:
                logger.warning("refusing external mode: %s", reason)
                return False
        self._leave(self.mode)
        self.mode = mode
        self._install(mode)
        logger.info("control mode -> %s", mode.value)
        return True

    def _leave(self, mode: ControlMode) -> None:
        if mode is ControlMode.EXTERNAL:
            for arm, path in self.held.items():
                if path is not None:
                    self.release(arm, path)
                self.held[arm] = None
                self.holding[arm] = False
        else:
            for _line, _zone, pick_place in self.stations.values():
                pick_place.reset()  # drops any box mid-cycle and returns to WAITING

    def _autonomous_ready_check(self, arm: int, pick_place) -> Callable[[], bool]:
        defer = self.defer_checks.get(arm)

        def ready() -> bool:
            if defer is not None and defer():
                return False
            return self.hold_while_busy or pick_place.phase_name == "WAITING"

        return ready

    def _install(self, mode: ControlMode) -> None:
        external = mode is ControlMode.EXTERNAL
        if external:
            self.bridge.reset()
        for arm, (line, zone_index, pick_place) in self.stations.items():
            if external:
                line.set_hold_zone_ready_check(zone_index, lambda arm=arm: not self.holding[arm])
            else:
                line.set_hold_zone_ready_check(zone_index, self._autonomous_ready_check(arm, pick_place))
            line.apply_belt_commands = not external
