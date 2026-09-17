"""PackML projection and per-tick message building for one zone."""

from __future__ import annotations

from conveyor_indexing.protos import telemetry
from conveyor_indexing.state_machine import DIRECTION_UNDEFINED, ZoneCommand, ZoneObservation

Machine = telemetry.ConveyorStateMachineCode
PackML = telemetry.PackMLState

_RUNNING_STATES = {
    Machine.CONVEYOR_STATE_MACHINE_INDUCTING,
    Machine.CONVEYOR_STATE_MACHINE_ADVANCE_ITEM,
    Machine.CONVEYOR_STATE_MACHINE_PASSTHROUGH,
}


def machine_to_packml(machine) -> int:
    """Two-bucket projection: running zones are EXECUTE, everything else IDLE."""
    return PackML.PACKML_EXECUTE if machine in _RUNNING_STATES else PackML.PACKML_IDLE


def append_conveyor_state(state_msg, zone, observation: ZoneObservation, command: ZoneCommand) -> None:
    """Append one `SimConveyorState` for `zone` this tick."""
    item = state_msg.conveyors.add()
    item.name = zone.node_path
    item.zone_index = zone.index
    item.run = command.run
    item.speed = command.speed_pct
    item.direction = command.direction
    item.machine = observation.machine
    item.packml = machine_to_packml(observation.machine)
    item.occupied = observation.occupied


def append_conveyor_command(commands_msg, zone, command: ZoneCommand) -> None:
    """Append one `SimConveyorCommand` for `zone` this tick."""
    cmd = commands_msg.commands.add()
    cmd.zone_index = zone.index
    cmd.conveyor_node_path = zone.node_path
    cmd.run = command.run
    cmd.speed = command.speed_pct
    cmd.direction = command.direction


def resolve_override_speed_direction(run: bool, speed: int, direction: int) -> tuple[int, int]:
    """Speed/direction to report for an externally commanded zone; stopped has no direction."""
    if not run:
        return 0, DIRECTION_UNDEFINED
    return speed, direction
