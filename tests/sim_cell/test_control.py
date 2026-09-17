"""Control requests, mode switching on a fake cell, and the Zenoh control channel."""

from __future__ import annotations

import json
import time
from types import SimpleNamespace

import pytest
import zenoh

from sim_cell.control import ControlChannel, ControlMode, ModeController, parse_request


def test_parse_request():
    assert parse_request(b'{"op": "set_mode", "mode": "external"}').mode is ControlMode.EXTERNAL
    assert parse_request(b'{"op": "status"}').op == "status"
    for bad in (b"nope", b'{"op": "fly"}', b'{"op": "set_mode", "mode": "manual"}'):
        with pytest.raises(ValueError):
            parse_request(bad)


class FakeLine:
    def __init__(self):
        self.ready = {}
        self.apply_belt_commands = True

    def set_hold_zone_ready_check(self, zone, fn):
        self.ready[zone] = fn


class FakePickPlace:
    def __init__(self):
        self.phase_name = "WAITING"
        self.resets = 0

    def reset(self):
        self.resets += 1
        self.phase_name = "WAITING"


class FakeBridge:
    def __init__(self):
        self.resets = 0

    def reset(self):
        self.resets += 1


def _controller(mode=ControlMode.AUTONOMOUS, block=None):
    line, pp1, pp2, bridge = FakeLine(), FakePickPlace(), FakePickPlace(), FakeBridge()
    released = []
    ctl = ModeController(
        stations={1: (line, 1, pp1), 2: (line, 2, pp2)},
        bridge=bridge,
        release=lambda arm, path: released.append((arm, path)),
        mode=mode,
        block_external=lambda: block,
    )
    return ctl, SimpleNamespace(line=line, pp1=pp1, pp2=pp2, bridge=bridge, released=released)


def test_autonomous_readiness_follows_phase():
    ctl, f = _controller()
    assert f.line.apply_belt_commands is True
    assert f.line.ready[1]() is True
    f.pp1.phase_name = "ATTACH"
    assert f.line.ready[1]() is False


def test_switch_to_external_resets_and_owns_belts():
    ctl, f = _controller()
    f.pp1.phase_name = "LIFT"
    assert ctl.apply(ControlMode.EXTERNAL)
    assert ctl.external and f.line.apply_belt_commands is False
    assert f.pp1.resets == 1 and f.bridge.resets == 1
    ctl.holding[1] = True
    assert f.line.ready[1]() is False and f.line.ready[2]() is True


def test_switch_back_releases_external_boxes():
    ctl, f = _controller(mode=ControlMode.EXTERNAL)
    ctl.held[2] = "/World/CubeBox_7"
    ctl.holding[2] = True
    assert ctl.apply(ControlMode.AUTONOMOUS)
    assert f.released == [(2, "/World/CubeBox_7")]
    assert ctl.held[2] is None and ctl.holding[2] is False
    assert f.line.apply_belt_commands is True and f.pp2.resets == 0  # never ran autonomously


def test_external_refused_when_blocked():
    ctl, f = _controller(block="recorder on")
    assert not ctl.apply(ControlMode.EXTERNAL)
    assert ctl.mode is ControlMode.AUTONOMOUS and f.bridge.resets == 0


def test_same_mode_is_noop():
    ctl, f = _controller()
    assert ctl.apply(ControlMode.AUTONOMOUS) and f.pp1.resets == 0


def test_control_channel_round_trip():
    chan = ControlChannel(ControlMode.AUTONOMOUS)
    client = zenoh.open(zenoh.Config())
    try:
        time.sleep(0.3)
        req = json.dumps({"op": "set_mode", "mode": "external"}).encode()
        replies = list(client.get(chan.topics.control, payload=req, timeout=5.0))
        assert replies
        body = json.loads(replies[0].ok.payload.to_bytes())
        assert body["mode"] == "autonomous" and "error" not in body
        assert chan.take_pending() is ControlMode.EXTERNAL
        assert chan.take_pending() is None
        chan.publish_status(ControlMode.EXTERNAL, 42)
        status = json.loads(list(client.get(chan.topics.status, timeout=5.0))[0].ok.payload.to_bytes())
        assert status == {"mode": "external", "sim_time_us": 42}
        bad = list(client.get(chan.topics.control, payload=b'{"op":"fly"}', timeout=5.0))
        assert "error" in json.loads(bad[0].ok.payload.to_bytes())
    finally:
        client.close()
        chan.close()
