"""Owns one line's ordered zones, wires neighbor occupancy, applies commands."""

from __future__ import annotations

import logging

from pxr import Usd

from conveyor_indexing.directions import fix_zone_directions
from conveyor_indexing.occupancy import leading_occupant_path
from conveyor_indexing.state_machine import HOLD_ZONE_STOP_FRACTION
from conveyor_indexing.telemetry import append_conveyor_command, append_conveyor_state
from conveyor_indexing.zone import ConveyorZone

logger = logging.getLogger(__name__)


class ConveyorLineController:
    """Owns one line's ordered zones, wires neighbor occupancy, applies commands.

    Supports both a closed loop (neighbor indices wrap) and an open line (first
    zone has no upstream, last has no downstream) via `closed_loop`.
    """

    def __init__(
        self,
        stage: Usd.Stage,
        node_paths: list,
        excluded_roots: tuple,
        hold_zone_indices: frozenset = frozenset(),
        closed_loop: bool = False,
        run_speed_pct: int = 100,
    ) -> None:
        self.zones = [
            ConveyorZone(i, path, stage, excluded_roots, run_speed_pct=run_speed_pct)
            for i, path in enumerate(node_paths)
        ]
        self.hold_zone_indices = hold_zone_indices
        self.closed_loop = closed_loop
        self.occupied: list = [False] * len(self.zones)
        self.machine_states: list = [None] * len(self.zones)
        self._box_rigid_prims: dict | None = None
        self._hold_zone_ready_checks: dict = {}  # zone_index -> ready_fn; absent = always-ready
        fix_zone_directions(self.zones, self.closed_loop)

    def set_hold_zone_ready_check(self, zone_index: int, ready_fn) -> None:
        """While ready_fn() is False (that zone's robot is busy), hold zone zone_index
        behaves like an ordinary pass-through zone, so a box overflows to the next
        pick station instead of backing up behind a busy robot.
        """
        self._hold_zone_ready_checks[zone_index] = ready_fn

    def set_box_rigid_prims(self, box_rigid_prims: dict) -> None:
        """Inject the live RigidPrim for every known box (built in sim_cell.cell only
        after world.reset()) so hold zones can check is_past_center() against real position.
        """
        self._box_rigid_prims = box_rigid_prims

    def step(self, state_msg, commands_msg, box_positions: dict) -> None:
        """Advance every zone by one control tick, appending into shared log messages.

        `box_positions` is {path: (x, y, z)}, precomputed once per control tick from a
        single batched RigidPrim read (see sim_cell.runner) rather than queried per-box.
        """
        for zone in self.zones:
            zone.invalidate_occupancy_cache()
        self.occupied = [zone.check_occupied() for zone in self.zones]

        n = len(self.zones)
        for i, zone in enumerate(self.zones):
            # Open line: zone 0 has no real upstream, so it's always treated as having
            # more infeed available; the last zone's downstream (open end/truck) is
            # always clear, unless it's a held zone, which never reports clear.
            if i == 0 and not self.closed_loop:
                upstream_occupied = True
            else:
                upstream_occupied = self.occupied[(i - 1) % n]

            # See set_hold_zone_ready_check: a hold zone overflows only while its robot
            # is busy, and never if it's the line's last zone (nowhere to overflow to).
            is_last = i == n - 1 and not self.closed_loop
            robot_ready = i in self.hold_zone_indices and self._hold_zone_ready_checks.get(i, lambda: True)()
            holding = i in self.hold_zone_indices and (robot_ready or is_last)
            if holding:
                downstream_clear = False
            elif is_last:
                downstream_clear = True
            else:
                downstream_clear = not self.occupied[(i + 1) % n]

            # Any hold zone defines a stop position, whether or not it's currently
            # holding for its own robot - it should still index a box up to that point
            # (maximizing its own occupancy) even while overflowing because its robot is
            # busy or its downstream neighbor is full; only non-hold zones default to
            # True (stop-as-soon-as-occupied).
            at_stop_position = True
            if i in self.hold_zone_indices and self._box_rigid_prims is not None and self.occupied[i]:
                occupying_paths = zone.get_occupying_prim_paths()
                leading_path = leading_occupant_path(
                    zone.world_travel_direction, occupying_paths, box_positions, zone_name=zone.node_path
                )
                if leading_path is not None:
                    at_stop_position = zone.is_past_center(
                        box_positions[leading_path], stop_fraction=HOLD_ZONE_STOP_FRACTION
                    )

            observation, command = zone.state_machine.step(
                occupied=self.occupied[i],
                upstream_occupied=upstream_occupied,
                downstream_clear=downstream_clear,
                at_stop_position=at_stop_position,
            )
            if i in self.hold_zone_indices:
                logger.debug(
                    "hold zone: %s occupied=%s holding=%s downstream_clear=%s at_stop_position=%s machine=%s run=%s",
                    zone.node_path, self.occupied[i], holding, downstream_clear, at_stop_position,
                    observation.machine, command.run,
                )
            zone.apply_command(command.run, command.speed_pct)
            self.machine_states[i] = observation.machine

            append_conveyor_state(state_msg, zone, observation, command)
            append_conveyor_command(commands_msg, zone, command)
