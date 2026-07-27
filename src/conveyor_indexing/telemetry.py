"""PackML projection and per-tick proto message building for one zone."""

from __future__ import annotations

from conveyor_indexing.protos import common_types, plc
from conveyor_indexing.state_machine import ZoneCommand, ZoneObservation

Machine = plc.ConveyorStateMachineCode

_RUNNING_STATES = {
    Machine.CONVEYOR_STATE_MACHINE_INDUCTING,
    Machine.CONVEYOR_STATE_MACHINE_ADVANCE_ITEM,
    Machine.CONVEYOR_STATE_MACHINE_PASSTHROUGH,
}


def machine_to_packml(machine) -> int:
    """Coarse, approximate PackML projection - see README for real PackML nuance
    (Starting/Stopping/Held/Aborted) not modeled by this two-bucket mapping.
    """
    return common_types.PACKML_EXECUTE if machine in _RUNNING_STATES else common_types.PACKML_IDLE


def append_conveyor_state(state_msg, zone, observation: ZoneObservation, command: ZoneCommand) -> None:
    """Append one `StateConveyors_ConveyorsItem` for `zone` this tick."""
    item = state_msg.Conveyors.add()
    item.Name = zone.node_path
    item.Type = plc.ConveyorTypeCode.CONVEYOR_TYPE_BUFFER  # TODO: set real per-zone type
    item.Fault = plc.ConveyorFaultCode.CONVEYOR_FAULT_UNSPECIFIED  # not modeled yet, see README
    item.PackML = machine_to_packml(observation.machine)
    item.Speed = command.speed_pct
    item.Direction = command.direction
    item.Machine = observation.machine


def append_conveyor_command(commands_msg, zone, command: ZoneCommand) -> None:
    """Append one `SimConveyorCommand` for `zone` this tick."""
    cmd = commands_msg.commands.add()
    cmd.zone_index = zone.index
    cmd.conveyor_node_path = zone.node_path
    cmd.run = command.run
    cmd.speed = command.speed_pct
    cmd.direction = command.direction
