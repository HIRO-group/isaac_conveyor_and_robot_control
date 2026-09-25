"""Capacity faults for the built-in state machine, so a demonstration run can
show what a degraded capacity record looks like without an external policy.

The three kappas follow capability-diffusion's `sim_control/degradation.py`
(paper Sec. III-A), realised here on the autonomous controller:

- kappa_r (arm reach, `CONVEYOR_INDEXING_KAPPA_R_ARM1`): the fraction of arm
  1's pick zone the arm can still serve. In the default reach mode ("decline",
  `CONVEYOR_INDEXING_REACH_MODE`) pick selection only considers boxes within
  `reach_for_kappa`; a box the arm cannot serve is passed on to the next
  station instead of being held (`ReachGate`). In reach mode "attempt" the
  controller is capacity-blind: it selects as if nominal, but the arm
  physically cannot extend past the same radius, so it reaches for a box
  beyond it as far as it can, stalls at the limit, retreats to the staging
  pose and tries again while the box is there (`ReachFault`) - the
  fail-freeze contrast take.
- kappa_h (hold, `CONVEYOR_INDEXING_KAPPA_H_ARM1`): the probability a grasp
  survives, drawn once per hold. A decided drop releases the box part-way
  through the arm's swing to the place side (`HoldFault`);
  `CONVEYOR_INDEXING_DROP_EVERY_HOLD=1` makes every hold drop regardless.
- kappa_c (conveyor, `CONVEYOR_INDEXING_KAPPA_C`): a speed cap on the supply
  line (loop 1, the belts feeding both pick zones) as a fraction of nominal.

Demonstration-only helpers live here too: the near/far and far-only spawn
layouts (`CONVEYOR_INDEXING_SPAWN_LAYOUT=near_far|far`, see
`sim_cell.box_spawner`) and
`CONVEYOR_INDEXING_HOLD_WHILE_BUSY=1`, which keeps a pick zone holding its
boxes while its arm is mid-cycle instead of overflowing them downstream, and
a hesitant-policy imitation (`Hesitation`): `CONVEYOR_INDEXING_PICKS_PER_MIN`
rate-limits each arm's cycles (it waits at the staging pose in between) and
`CONVEYOR_INDEXING_HESITATIONS` freezes each descent that many times for a
second or three, so the arm visibly stops and hesitates on its way to a box.

Import-light on purpose (no Isaac): everything here is unit-testable.
"""

from __future__ import annotations

import logging
import math
import os
import random
from dataclasses import dataclass

logger = logging.getLogger(__name__)

KAPPA_R_ARM1_ENV_VAR = "CONVEYOR_INDEXING_KAPPA_R_ARM1"
KAPPA_H_ARM1_ENV_VAR = "CONVEYOR_INDEXING_KAPPA_H_ARM1"
KAPPA_C_ENV_VAR = "CONVEYOR_INDEXING_KAPPA_C"
SPAWN_LAYOUT_ENV_VAR = "CONVEYOR_INDEXING_SPAWN_LAYOUT"
HOLD_WHILE_BUSY_ENV_VAR = "CONVEYOR_INDEXING_HOLD_WHILE_BUSY"
FAULT_SEED_ENV_VAR = "CONVEYOR_INDEXING_FAULT_SEED"
DROP_EVERY_HOLD_ENV_VAR = "CONVEYOR_INDEXING_DROP_EVERY_HOLD"
REACH_MODE_ENV_VAR = "CONVEYOR_INDEXING_REACH_MODE"
REACH_STALL_ENV_VAR = "CONVEYOR_INDEXING_REACH_STALL_S"
PICKS_PER_MIN_ENV_VAR = "CONVEYOR_INDEXING_PICKS_PER_MIN"
HESITATIONS_ENV_VAR = "CONVEYOR_INDEXING_HESITATIONS"

SPAWN_LAYOUTS = ("waves", "near_far", "far")
REACH_MODES = ("decline", "attempt")

# Reach mode "attempt": how long the arm strains at its reach limit before it
# gives the descent up and retreats (REACH_STALL_ENV_VAR overrides).
REACH_STALL_S = 1.0

# UR20 spec reach, the nominal selection radius (pick_and_place.selection's
# PICK_MAX_REACH_M; duplicated so this module stays free of that package's
# Isaac-only imports).
NOMINAL_MAX_REACH_M = 1.75

# A hesitation freezes the descent for a random length in this range, at a
# random point in the middle of the planned motion.
HESITATION_STOP_S = (1.0, 3.0)
HESITATION_WINDOW = (0.15, 0.85)

# A decided drop fires this far (fraction of the planned duration) into the
# arm's swing from the pick side to the place side, so the box falls mid-air
# between the belts.
DROP_WINDOW = (0.3, 0.5)


def _kappa_from_env(env, name: str) -> float:
    raw = env.get(name)
    if raw is None or raw == "":
        return 1.0
    value = float(raw)
    if not 0.0 < value <= 1.0:
        raise ValueError(f"{name}={raw}: a capacity must be in (0, 1]")
    return value


@dataclass(frozen=True)
class FaultConfig:
    kappa_r_arm1: float = 1.0
    kappa_h_arm1: float = 1.0
    kappa_c: float = 1.0
    spawn_layout: str = "waves"
    hold_while_busy: bool = False
    seed: int | None = None
    drop_every_hold: bool = False
    reach_mode: str = "decline"
    reach_stall_s: float = REACH_STALL_S
    picks_per_min: float | None = None
    hesitations: int = 0

    @classmethod
    def from_env(cls, environ=None) -> "FaultConfig":
        env = os.environ if environ is None else environ
        layout = env.get(SPAWN_LAYOUT_ENV_VAR, "waves")
        if layout not in SPAWN_LAYOUTS:
            raise ValueError(f"{SPAWN_LAYOUT_ENV_VAR}={layout!r}: expected one of {SPAWN_LAYOUTS}")
        reach_mode = env.get(REACH_MODE_ENV_VAR, "decline")
        if reach_mode not in REACH_MODES:
            raise ValueError(f"{REACH_MODE_ENV_VAR}={reach_mode!r}: expected one of {REACH_MODES}")
        picks_raw = env.get(PICKS_PER_MIN_ENV_VAR, "")
        picks_per_min = float(picks_raw) if picks_raw else None
        if picks_per_min is not None and picks_per_min <= 0.0:
            raise ValueError(f"{PICKS_PER_MIN_ENV_VAR}={picks_raw}: must be > 0")
        hesitations = int(env.get(HESITATIONS_ENV_VAR, "0") or 0)
        if hesitations < 0:
            raise ValueError(f"{HESITATIONS_ENV_VAR}={hesitations}: must be >= 0")
        stall_raw = env.get(REACH_STALL_ENV_VAR, "")
        reach_stall_s = float(stall_raw) if stall_raw else REACH_STALL_S
        if reach_stall_s <= 0.0:
            raise ValueError(f"{REACH_STALL_ENV_VAR}={stall_raw}: must be > 0")
        seed_raw = env.get(FAULT_SEED_ENV_VAR)
        return cls(
            kappa_r_arm1=_kappa_from_env(env, KAPPA_R_ARM1_ENV_VAR),
            kappa_h_arm1=_kappa_from_env(env, KAPPA_H_ARM1_ENV_VAR),
            kappa_c=_kappa_from_env(env, KAPPA_C_ENV_VAR),
            spawn_layout=layout,
            hold_while_busy=env.get(HOLD_WHILE_BUSY_ENV_VAR, "") == "1",
            seed=int(seed_raw) if seed_raw else None,
            drop_every_hold=env.get(DROP_EVERY_HOLD_ENV_VAR, "") == "1",
            reach_mode=reach_mode,
            reach_stall_s=reach_stall_s,
            picks_per_min=picks_per_min,
            hesitations=hesitations,
        )

    @property
    def nominal(self) -> bool:
        return self.kappa_r_arm1 >= 1.0 and self.kappa_h_arm1 >= 1.0 and self.kappa_c >= 1.0

    def loop1_run_speed_pct(self, nominal_pct: int) -> int:
        """kappa_c as a cap on the supply line's run speed, in the command's own percent units."""
        return int(round(nominal_pct * self.kappa_c))

    @property
    def attempts_unreachable(self) -> bool:
        """Reach mode "attempt" with a real reach fault: selection is nominal and
        the limit is physical (see ReachFault)."""
        return self.reach_mode == "attempt" and self.kappa_r_arm1 < 1.0

    def arm1_reach_fault(self, max_reach_m: float, robot_xy, physics_dt: float) -> "ReachFault | None":
        if not self.attempts_unreachable:
            return None
        return ReachFault(max_reach_m, robot_xy, max(1, int(round(self.reach_stall_s / physics_dt))))

    def hesitation(self, arm: int, physics_dt: float) -> "Hesitation | None":
        """The hesitant-policy imitation for one arm, or None when neither knob is set."""
        if self.picks_per_min is None and self.hesitations == 0:
            return None
        rng = random.Random(self.seed + arm) if self.seed is not None else random.Random()
        period_s = 60.0 / self.picks_per_min if self.picks_per_min else 0.0
        return Hesitation(period_s, self.hesitations, rng, physics_dt)

    def arm1_hold_fault(self) -> "HoldFault | None":
        if self.kappa_h_arm1 >= 1.0 and not self.drop_every_hold:
            return None
        rng = random.Random(self.seed) if self.seed is not None else random.Random()
        return HoldFault(self.kappa_h_arm1, rng, always_drop=self.drop_every_hold)

    def describe(self) -> str:
        parts = [f"kappa_r_arm1={self.kappa_r_arm1:g}", f"kappa_h_arm1={self.kappa_h_arm1:g}", f"kappa_c={self.kappa_c:g}"]
        if self.reach_mode != "decline":
            parts.append(f"reach_mode={self.reach_mode} reach_stall_s={self.reach_stall_s:g}")
        if self.spawn_layout != "waves":
            parts.append(f"spawn_layout={self.spawn_layout}")
        if self.hold_while_busy:
            parts.append("hold_while_busy")
        if self.drop_every_hold:
            parts.append("drop_every_hold")
        if self.picks_per_min is not None:
            parts.append(f"picks_per_min={self.picks_per_min:g}")
        if self.hesitations:
            parts.append(f"hesitations={self.hesitations}")
        if self.seed is not None:
            parts.append(f"fault_seed={self.seed}")
        return " ".join(parts)


# -- kappa_r: reach ------------------------------------------------------------


def pick_zone_reach_span(
    zone_center, zone_half_extent, travel_axis: int, travel_sign: float, stop_fraction: float, robot_xy
) -> tuple[float, float]:
    """(nearest, farthest) distance from the robot base to the zone's stop line.

    The stop line is the lateral line across the belt where a hold zone parks
    a box (`stop_fraction` of the way through, see
    conveyor_indexing.state_machine.HOLD_ZONE_STOP_FRACTION). A parked box's
    distance from the robot lies between these two bounds.
    """
    lateral_axis = 1 - travel_axis
    stop = zone_center[travel_axis] + travel_sign * zone_half_extent[travel_axis] * (2 * stop_fraction - 1)
    lo = zone_center[lateral_axis] - zone_half_extent[lateral_axis]
    hi = zone_center[lateral_axis] + zone_half_extent[lateral_axis]
    point = [0.0, 0.0]
    distances = []
    for lateral in (lo, hi):
        point[travel_axis] = stop
        point[lateral_axis] = lateral
        distances.append(math.dist(point, robot_xy[:2]))
    return min(distances), max(distances)


def reach_for_kappa(kappa_r: float, d_near: float, d_far: float, nominal_max_reach_m: float = NOMINAL_MAX_REACH_M) -> float:
    """Selection radius realising kappa_r: the arm serves the nearest `kappa_r`
    of the zone's stop line, never more than its nominal reach."""
    if kappa_r >= 1.0:
        return nominal_max_reach_m
    return min(nominal_max_reach_m, d_near + kappa_r * (d_far - d_near))


class ReachGate:
    """Hold-zone readiness veto: True while the zone holds boxes the arm cannot
    serve (so the line passes them on). Updated by the runner each control tick."""

    def __init__(self) -> None:
        self.unreachable = False

    def __call__(self) -> bool:
        return self.unreachable


class ReachFault:
    """kappa_r as a physical limit on the autonomous pick cycle (reach mode
    "attempt"): the tool cannot get farther than `max_reach_m` (horizontal
    distance) from the base. For a box beyond it the pick controller descends
    to `clip` of the pick point - as far along the line to the box as the arm
    goes, at the box's height - and, there or the tick the tool strays past
    the limit (`beyond`), holds the arm for `stall_ticks` (`begin_stall`,
    `stalling`), then abandons the cycle and retries from the staging pose
    (the box stays, so the retries never end).
    """

    def __init__(self, max_reach_m: float, robot_xy, stall_ticks: int) -> None:
        if max_reach_m <= 0.0 or stall_ticks < 1:
            raise ValueError(f"need max_reach_m > 0 and stall_ticks >= 1, got {max_reach_m}, {stall_ticks}")
        self.max_reach_m = max_reach_m
        self.robot_xy = (float(robot_xy[0]), float(robot_xy[1]))
        self.stall_ticks = stall_ticks
        self._stalled = 0
        self.attempts = 0  # descents that ended at the limit
        self.last_distance_m: float | None = None

    def distance_m(self, position) -> float:
        return math.dist((float(position[0]), float(position[1])), self.robot_xy)

    def beyond(self, position) -> bool:
        return self.distance_m(position) > self.max_reach_m

    def clip(self, target):
        """`target` (x, y, z) if within reach, else the point at the same height
        `max_reach_m` along the base-to-target line - the closest the arm gets."""
        d = self.distance_m(target)
        if d <= self.max_reach_m:
            return target
        f = self.max_reach_m / d
        return (
            self.robot_xy[0] + (float(target[0]) - self.robot_xy[0]) * f,
            self.robot_xy[1] + (float(target[1]) - self.robot_xy[1]) * f,
            float(target[2]),
        )

    def begin_stall(self, tool_position) -> int:
        """The descent has ended at the limit: count the attempt, start the stall."""
        self.last_distance_m = self.distance_m(tool_position)
        self.attempts += 1
        self._stalled = 0
        return self.attempts

    def stalling(self) -> bool:
        """Call once per tick while stalled: True until the stall has run its course."""
        self._stalled += 1
        return self._stalled < self.stall_ticks

    def reset(self) -> None:
        self._stalled = 0


# -- kappa_h: hold ---------------------------------------------------------------


class HoldFault:
    """kappa_h as a physical fault on the autonomous pick cycle: at every attach a
    drop is decided with probability `1 - holding_fraction` (or always, with
    `always_drop`); a decided drop fires once the swing to the place side is a
    random fraction of the way through, drawn from `window`.
    """

    def __init__(
        self,
        holding_fraction: float,
        rng: random.Random,
        always_drop: bool = False,
        window: tuple[float, float] = DROP_WINDOW,
    ) -> None:
        if not 0.0 < holding_fraction <= 1.0:
            raise ValueError(f"holding_fraction must be in (0, 1], got {holding_fraction}")
        if not 0.0 <= window[0] <= window[1] <= 1.0:
            raise ValueError(f"drop window must satisfy 0 <= lo <= hi <= 1, got {window}")
        self.holding_fraction = holding_fraction
        self.always_drop = always_drop
        self._rng = rng
        self._window = window
        self._drop_at: float | None = None  # swing progress at which this hold drops
        self.holds = 0
        self.drops = 0

    def on_attach(self) -> bool:
        """Decide this hold's fate. Returns True when it will drop."""
        self.holds += 1
        will_drop = self.always_drop or self._rng.random() >= self.holding_fraction
        self._drop_at = self._rng.uniform(*self._window) if will_drop else None
        return will_drop

    def should_drop(self, swing_progress: float) -> bool:
        """Call every physics step of the swing with its progress in [0, 1]; True
        exactly once, when the decided drop point is reached."""
        if self._drop_at is None or swing_progress < self._drop_at:
            return False
        self._drop_at = None
        self.drops += 1
        return True

    def reset(self) -> None:
        self._drop_at = None


# -- hesitant policy imitation -----------------------------------------------------


class Hesitation:
    """Makes the autonomous pick cycle look like a hesitant learned policy: at
    most one cycle per `period_s` (the arm waits at the staging pose until the
    period since the last start has passed), and `stops` freezes per descent,
    each HESITATION_STOP_S long at a random point in HESITATION_WINDOW of the
    planned motion. Ticks are physics steps; the controller supplies them.
    """

    def __init__(self, period_s: float, stops: int, rng: random.Random, physics_dt: float,
                 stop_s: tuple = HESITATION_STOP_S, window: tuple = HESITATION_WINDOW) -> None:
        if period_s < 0.0 or stops < 0 or physics_dt <= 0.0:
            raise ValueError(f"need period_s >= 0, stops >= 0, physics_dt > 0; got {period_s}, {stops}, {physics_dt}")
        self.period_ticks = int(round(period_s / physics_dt))
        self.stops = stops
        self._rng = rng
        self._physics_dt = physics_dt
        self._stop_s = stop_s
        self._window = window
        self._last_start_tick: int | None = None
        self._pending: list = []  # (progress, ticks) still to fire, ascending
        self._frozen_ticks = 0
        self.cycles = 0
        self.freezes = 0

    def can_start(self, tick: int) -> bool:
        return self._last_start_tick is None or tick - self._last_start_tick >= self.period_ticks

    def start_cycle(self, tick: int) -> None:
        """A cycle begins now: draw where and how long this descent will freeze."""
        self._last_start_tick = tick
        self.cycles += 1
        points = sorted(self._rng.uniform(*self._window) for _ in range(self.stops))
        self._pending = [(p, max(1, int(round(self._rng.uniform(*self._stop_s) / self._physics_dt)))) for p in points]
        self._frozen_ticks = 0

    def frozen(self, progress: float) -> bool:
        """Call once per descent tick with the plan's progress: True while the arm
        must hold still (the caller then skips advancing the trajectory)."""
        if self._frozen_ticks > 0:
            self._frozen_ticks -= 1
            return True
        if self._pending and progress >= self._pending[0][0]:
            _, ticks = self._pending.pop(0)
            self._frozen_ticks = ticks - 1
            self.freezes += 1
            return True
        return False

    def reset(self) -> None:
        self._pending = []
        self._frozen_ticks = 0


# -- near/far spawn layout -------------------------------------------------------

# Lateral position of each box of a near/far pair, as a signed fraction of the
# room left across the belt once the box and a lane clearance are subtracted;
# positive is the robot's side. 0.9 puts the near box a hair off the robot's
# belt edge, -0.8 keeps the far box inside the nominal reach.
NEAR_FAR_LATERAL_FRACTIONS = (0.9, -0.8)


def near_far_lateral_offsets(
    lateral_half_extent: float, box_half_extents: tuple, near_side_sign: float, clearance_m: float,
    fractions: tuple = NEAR_FAR_LATERAL_FRACTIONS,
) -> list[float]:
    """Lateral offsets from the belt centre for the (near, far) pair, yaw 0; one
    box per entry of `fractions` (the far box alone is `fractions[1:]`)."""
    offsets = []
    for fraction, half in zip(fractions, box_half_extents):
        room = max(0.0, lateral_half_extent - max(half[0], half[1]) - clearance_m)
        offsets.append(near_side_sign * fraction * room)
    return offsets
