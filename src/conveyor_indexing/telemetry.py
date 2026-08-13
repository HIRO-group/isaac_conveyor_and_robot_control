"""PackML projection and per-tick proto message building for one zone."""

from __future__ import annotations

from conveyor_indexing.protos import common_types, plc
from conveyor_indexing.state_machine import DIRECTION_UNDEFINED, ZoneCommand, ZoneObservation

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


def resolve_override_speed_direction(run: bool, speed: int, direction: int) -> tuple[int, int]:
    """What Speed/Direction an externally-commanded conveyor override
    (CONVEYOR_INDEXING_EXTERNAL_ACTION=1 - see sim_cell.runner) should
    reflect back into a StateConveyors_ConveyorsItem already appended by
    append_conveyor_state for this zone/tick.

    sim_cell.runner's override loop re-points Speed at what actually got
    commanded so theia/plc/state_conveyors doesn't silently report the
    autonomous state machine's stale decision once an external command owns
    the real belt - but was missing the same reflection for Direction. This
    mirrors append_conveyor_state's own field semantics: not running has no
    meaningful direction, hence DIRECTION_UNDEFINED (matching the "stop every
    zone" fallback's Speed=0 when no external command has ever arrived).
    """
    if not run:
        return 0, DIRECTION_UNDEFINED
    return speed, direction
