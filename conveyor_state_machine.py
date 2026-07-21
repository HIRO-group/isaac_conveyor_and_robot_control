"""Per-zone conveyor indexing state machine.

Implements a best-effort "happy path" subset of the real
``ConveyorStateMachineCode`` enum (see
``~/theia/proto/plc-connector/plc-connector.proto``). Only the states needed
to move a single item through a zone under normal conditions are implemented:

    EMPTY -> READY_TO_RECEIVE -> WAITING_TO_INDUCT -> INDUCTING
          -> IDLE (holding, occupied, downstream busy)
          -> ADVANCE_ITEM -> PASSTHROUGH -> EMPTY (loop)

The exception/reject states (WAITING_TO_REJECT, REJECT_SINGLE, REJECT_STUCK,
REJECT_FULL, REJECT_SPUR, SHIFTED_ITEM, PLACE_UNEXPECTED_ITEM, PURGE,
AWAITING_DECISION, CLEAR_FOR_PLACE, READY_FOR_PLACEMENT) are NOT implemented.
Their real transition conditions live in the physical PLC's ladder logic,
which isn't available in this repo - ~/theia/docs/PLC/UDT.md doesn't even
document the `Machine` field, and ~/theia/proto/plc-connector/plc-connector.proto
leaves `Conveyor_Fault` marked "pending definition". Guessing at those
transitions here would bake fabricated behavior into training data whose
whole point is to reflect real indexing decisions - see
``_handle_exception_states`` below, which is an explicit, never-called stub
for that future work.

This module depends on ``plc_connector_pb2``, generated from theia's real
proto (see the top-level README in this directory for the generation step).
"""

from __future__ import annotations

from dataclasses import dataclass

import plc_connector_pb2 as plc

Machine = plc.ConveyorStateMachineCode

# Convention per ~/theia/docs/PLC/UDT.md `Conveyor_Direction`. The wire field
# itself is a raw sint32 (see plc-connector.proto), not a proto enum.
DIRECTION_UNDEFINED = 0
DIRECTION_FORWARD = 1
DIRECTION_REVERSE = 2

# Number of control ticks to hold WAITING_TO_INDUCT before starting to run.
# Stands in for a real upstream/downstream handshake signal we don't have in
# sim; tune once real handshake timing is known.
INDUCT_HANDSHAKE_TICKS = 1


@dataclass
class ZoneCommand:
    """What the zone controller decided to do this tick."""

    run: bool
    speed_pct: int
    direction: int


@dataclass
class ZoneObservation:
    """What the zone controller observed this tick, for logging."""

    machine: "plc.ConveyorStateMachineCode.V"
    occupied: bool


class ConveyorZoneStateMachine:
    """Happy-path state machine for a single conveyor zone.

    One instance per physical zone. ``ConveyorLineController`` (see
    ``conveyor_indexer.py``) wires instances together in belt order and
    supplies the upstream/downstream signals each needs.
    """

    def __init__(self, name: str, run_speed_pct: int = 100) -> None:
        self.name = name
        self._run_speed_pct = run_speed_pct
        self._state: "plc.ConveyorStateMachineCode.V" = Machine.CONVEYOR_STATE_MACHINE_STOPPED
        self._handshake_ticks_remaining = 0

    def start(self) -> None:
        """Enable the zone (master run-enable for the whole line turned on)."""
        if self._state == Machine.CONVEYOR_STATE_MACHINE_STOPPED:
            self._state = Machine.CONVEYOR_STATE_MACHINE_EMPTY

    def stop(self) -> None:
        """Disable the zone unconditionally (master run-enable turned off)."""
        self._state = Machine.CONVEYOR_STATE_MACHINE_STOPPED

    def step(
        self,
        occupied: bool,
        upstream_occupied: bool,
        downstream_clear: bool,
    ) -> tuple[ZoneObservation, ZoneCommand]:
        """Advance the zone's state machine by one control tick.

        Args:
            occupied: Whether this zone's own occupancy sensor currently
                detects a part.
            upstream_occupied: Whether the neighboring upstream zone is
                occupied (i.e. has a part ready to hand off). For the first
                zone in a line, this should be driven by whatever upstream
                infeed/spawn signal exists.
            downstream_clear: Whether the neighboring downstream zone can
                accept a handoff (unoccupied). For the last zone in a line,
                this should be True (handing off to an outfeed outside the
                modeled system).

        Returns:
            A tuple of (observation, command) for this tick - the observation
            is what gets logged as state, the command is what gets applied to
            the sim's ConveyorNode and logged as the action.
        """
        s = self._state

        if s == Machine.CONVEYOR_STATE_MACHINE_STOPPED:
            pass  # stays stopped until start() is called externally

        elif s == Machine.CONVEYOR_STATE_MACHINE_EMPTY:
            if occupied:
                # Something appeared without going through the induct
                # handshake (e.g. spawned directly onto the zone). Treat it
                # as already inducted rather than getting stuck.
                self._state = Machine.CONVEYOR_STATE_MACHINE_IDLE
            elif upstream_occupied:
                self._state = Machine.CONVEYOR_STATE_MACHINE_READY_TO_RECEIVE

        elif s == Machine.CONVEYOR_STATE_MACHINE_READY_TO_RECEIVE:
            if not upstream_occupied:
                self._state = Machine.CONVEYOR_STATE_MACHINE_EMPTY
            else:
                self._handshake_ticks_remaining = INDUCT_HANDSHAKE_TICKS
                self._state = Machine.CONVEYOR_STATE_MACHINE_WAITING_TO_INDUCT

        elif s == Machine.CONVEYOR_STATE_MACHINE_WAITING_TO_INDUCT:
            if not upstream_occupied:
                self._state = Machine.CONVEYOR_STATE_MACHINE_EMPTY
            elif self._handshake_ticks_remaining <= 0:
                self._state = Machine.CONVEYOR_STATE_MACHINE_INDUCTING
            else:
                self._handshake_ticks_remaining -= 1

        elif s == Machine.CONVEYOR_STATE_MACHINE_INDUCTING:
            if occupied:
                self._state = Machine.CONVEYOR_STATE_MACHINE_IDLE

        elif s == Machine.CONVEYOR_STATE_MACHINE_IDLE:
            # Overloaded on purpose: IDLE + occupied=True means "holding a
            # part, waiting for downstream to clear". Cross-reference the
            # logged `occupied` field to disambiguate from the empty/idle
            # case, which this scaffold routes through EMPTY instead.
            if not occupied:
                self._state = Machine.CONVEYOR_STATE_MACHINE_EMPTY
            elif downstream_clear:
                self._state = Machine.CONVEYOR_STATE_MACHINE_ADVANCE_ITEM

        elif s == Machine.CONVEYOR_STATE_MACHINE_ADVANCE_ITEM:
            if not occupied:
                # Left this zone without the downstream zone ever reporting
                # occupied (e.g. sensor gap at the boundary) - treat as a
                # completed handoff rather than getting stuck.
                self._state = Machine.CONVEYOR_STATE_MACHINE_EMPTY
            elif not downstream_clear:
                # Downstream just became occupied - that's this item
                # starting to cross the boundary, i.e. the handoff beginning,
                # not some unrelated obstruction. Proceed, don't abort.
                self._state = Machine.CONVEYOR_STATE_MACHINE_PASSTHROUGH
            # else: downstream still clear, keep advancing (state unchanged).

        elif s == Machine.CONVEYOR_STATE_MACHINE_PASSTHROUGH:
            if not occupied:
                self._state = Machine.CONVEYOR_STATE_MACHINE_EMPTY

        else:
            # Any exception/reject state (or CONVEYOR_STATE_MACHINE_UNDEFINED)
            # is out of scope for this scaffold. Surface loudly rather than
            # silently running the belt in an unknown state.
            raise NotImplementedError(
                f"Zone '{self.name}' is in unimplemented Machine state {s!r}; "
                "see conveyor_state_machine.py module docstring."
            )

        running = self._state in (
            Machine.CONVEYOR_STATE_MACHINE_INDUCTING,
            Machine.CONVEYOR_STATE_MACHINE_ADVANCE_ITEM,
            Machine.CONVEYOR_STATE_MACHINE_PASSTHROUGH,
        )
        command = ZoneCommand(
            run=running,
            speed_pct=self._run_speed_pct if running else 0,
            direction=DIRECTION_FORWARD,
        )
        observation = ZoneObservation(machine=self._state, occupied=occupied)
        return observation, command

    def _handle_exception_states(self) -> "plc.ConveyorStateMachineCode.V":
        """Placeholder for reject/fault decision logic. Never called.

        Real transition conditions for WAITING_TO_REJECT, REJECT_SINGLE,
        REJECT_STUCK, REJECT_FULL, REJECT_SPUR, SHIFTED_ITEM,
        PLACE_UNEXPECTED_ITEM, and PURGE live in the physical PLC's ladder
        logic, which is not available in this repo. Do not guess at these
        transitions - wire this up once the real logic is provided (see the
        module docstring).
        """
        raise NotImplementedError(
            "Reject/fault state transitions are not implemented; see "
            "conveyor_state_machine.py module docstring."
        )
