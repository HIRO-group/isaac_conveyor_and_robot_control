"""Unit tests for conveyor_indexing.telemetry - the P8 "external conveyor
override doesn't reflect Direction back into state msg" fix. No Isaac Sim
needed: append_conveyor_state/append_conveyor_command only touch
`zone.node_path`/`zone.index` (plain attributes) plus real protobuf
messages, so a lightweight duck-typed fake zone is enough.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from conveyor_indexing import telemetry
from conveyor_indexing.protos import plc, sim_action
from conveyor_indexing.state_machine import DIRECTION_FORWARD, DIRECTION_UNDEFINED, ZoneCommand, ZoneObservation

Machine = plc.ConveyorStateMachineCode


def _fake_zone(node_path: str = "/World/ConveyorTrack_01/ConveyorBeltGraph/ConveyorNode", index: int = 1):
    return SimpleNamespace(node_path=node_path, index=index)


# -- resolve_override_speed_direction (the actual bug fix) -------------------


def test_resolve_override_speed_direction_running_reflects_commanded_values():
    speed, direction = telemetry.resolve_override_speed_direction(run=True, speed=55, direction=2)
    assert (speed, direction) == (55, 2)


def test_resolve_override_speed_direction_not_running_is_zero_and_undefined():
    """Before the fix, only Speed was zeroed on the "no command yet"/not-
    running path - Direction silently kept whatever the autonomous state
    machine had last decided (always DIRECTION_FORWARD - see
    ConveyorZoneStateMachine.step). This is the regression guard.
    """
    speed, direction = telemetry.resolve_override_speed_direction(run=False, speed=55, direction=2)
    assert (speed, direction) == (0, DIRECTION_UNDEFINED)


def test_resolve_override_speed_direction_not_running_ignores_stale_inputs():
    # Even nonsense/stale speed+direction inputs are zeroed when not running.
    speed, direction = telemetry.resolve_override_speed_direction(run=False, speed=999, direction=DIRECTION_FORWARD)
    assert (speed, direction) == (0, DIRECTION_UNDEFINED)


# -- append_conveyor_state / append_conveyor_command (existing behavior,     -
# -- regression-guarded so the fix above composes correctly with them)      --


def test_append_conveyor_state_then_override_reflects_both_fields():
    """End-to-end: append_conveyor_state's autonomous Speed/Direction (always
    DIRECTION_FORWARD while running), then the external-override
    reassignment sim_cell.runner performs - both Speed and Direction must
    reflect the externally-commanded values afterward, not just Speed.
    """
    state_msg = plc.StateConveyors()
    zone = _fake_zone()
    observation = ZoneObservation(machine=Machine.CONVEYOR_STATE_MACHINE_INDUCTING, occupied=True)
    autonomous_command = ZoneCommand(run=True, speed_pct=55, direction=DIRECTION_FORWARD)
    telemetry.append_conveyor_state(state_msg, zone, observation, autonomous_command)

    item = state_msg.Conveyors[0]
    assert item.Speed == 55
    assert item.Direction == DIRECTION_FORWARD

    # External controller commands a slower reverse run instead.
    item.Speed, item.Direction = telemetry.resolve_override_speed_direction(run=True, speed=20, direction=2)
    assert item.Speed == 20
    assert item.Direction == 2


def test_append_conveyor_command_carries_direction():
    commands_msg = sim_action.SimConveyorCommands()
    zone = _fake_zone(index=3)
    command = ZoneCommand(run=True, speed_pct=55, direction=DIRECTION_FORWARD)
    telemetry.append_conveyor_command(commands_msg, zone, command)

    cmd = commands_msg.commands[0]
    assert cmd.zone_index == 3
    assert cmd.run is True
    assert cmd.speed == 55
    assert cmd.direction == DIRECTION_FORWARD


@pytest.mark.parametrize("run", [True, False])
def test_resolve_override_speed_direction_run_flag_alone_determines_branch(run):
    speed, direction = telemetry.resolve_override_speed_direction(run, 42, 1)
    if run:
        assert (speed, direction) == (42, 1)
    else:
        assert (speed, direction) == (0, DIRECTION_UNDEFINED)
