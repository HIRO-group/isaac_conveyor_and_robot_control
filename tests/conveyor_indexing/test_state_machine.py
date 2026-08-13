"""Unit tests for conveyor_indexing.state_machine - includes the P8
line_controller.py:103 dead-conditional cleanup
(`stop_fraction = 0.8 if is_last else 0.8` -> HOLD_ZONE_STOP_FRACTION).

conveyor_indexing.line_controller/zone/occupancy themselves cannot be
imported without Isaac Sim (they import carb/omni.physics.core - see
tests/conftest.py) - so this only tests the extracted constant + the
existing state-machine behavior directly, not ConveyorLineController.step()
itself. The stop_fraction call site (line_controller.py) is verified by
reading the diff: both ternary branches evaluated to 0.8 before the fix, so
collapsing to one named constant is a behavior-preserving no-op by
construction, not just by this test.
"""

from __future__ import annotations

from conveyor_indexing import state_machine as sm


def test_hold_zone_stop_fraction_is_defined_and_matches_prior_dead_ternary_value():
    """Both branches of the removed `0.8 if is_last else 0.8` ternary
    evaluated to 0.8 - the constant must preserve that exact value, so the
    line_controller.py cleanup is behavior-preserving.
    """
    assert sm.HOLD_ZONE_STOP_FRACTION == 0.8


def test_direction_constants_match_udt_convention():
    """Per ~/theia/docs/PLC/UDT.md Conveyor_Direction - regression guard for
    telemetry.resolve_override_speed_direction's DIRECTION_UNDEFINED usage.
    """
    assert sm.DIRECTION_UNDEFINED == 0
    assert sm.DIRECTION_FORWARD == 1
    assert sm.DIRECTION_REVERSE == 2


def test_zone_command_direction_is_always_forward_from_step():
    """Documents existing behavior the P8 Direction fix depends on:
    ConveyorZoneStateMachine.step always returns DIRECTION_FORWARD
    regardless of running state - so an external override reflecting
    DIRECTION_UNDEFINED while stopped is a real, visible change versus the
    autonomous default, not a no-op.
    """
    machine = sm.ConveyorZoneStateMachine(name="test-zone", run_speed_pct=55)
    machine.start()
    _, command = machine.step(occupied=False, upstream_occupied=False, downstream_clear=True)
    assert command.direction == sm.DIRECTION_FORWARD
    assert command.run is False
