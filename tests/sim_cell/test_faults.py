"""sim_cell.faults: env parsing, the kappa_r reach mapping, the kappa_h drop
decision, and the near/far pair geometry. Pure, no Isaac."""

from __future__ import annotations

import math

import pytest

from sim_cell import faults
from sim_cell.faults import (
    FaultConfig,
    Hesitation,
    HoldFault,
    ReachFault,
    ReachGate,
    near_far_lateral_offsets,
    pick_zone_reach_span,
    reach_for_kappa,
)


def test_from_env_defaults_are_nominal():
    cfg = FaultConfig.from_env({})
    assert cfg.nominal and cfg.spawn_layout == "waves" and not cfg.hold_while_busy and cfg.seed is None
    assert cfg.reach_mode == "decline" and not cfg.attempts_unreachable
    assert cfg.arm1_hold_fault() is None
    assert cfg.arm1_reach_fault(1.1, (0.0, 0.0), 1 / 120) is None
    assert cfg.loop1_run_speed_pct(55) == 55


def test_from_env_reads_every_knob():
    cfg = FaultConfig.from_env({
        faults.KAPPA_R_ARM1_ENV_VAR: "0.3", faults.KAPPA_H_ARM1_ENV_VAR: "0.3", faults.KAPPA_C_ENV_VAR: "0.5",
        faults.SPAWN_LAYOUT_ENV_VAR: "near_far", faults.HOLD_WHILE_BUSY_ENV_VAR: "1", faults.FAULT_SEED_ENV_VAR: "7",
    })
    assert (cfg.kappa_r_arm1, cfg.kappa_h_arm1, cfg.kappa_c) == (0.3, 0.3, 0.5)
    assert cfg.spawn_layout == "near_far" and cfg.hold_while_busy and cfg.seed == 7
    assert not cfg.nominal
    assert cfg.loop1_run_speed_pct(55) == 28
    assert isinstance(cfg.arm1_hold_fault(), HoldFault)
    assert "kappa_c=0.5" in cfg.describe() and "near_far" in cfg.describe()


@pytest.mark.parametrize("bad", [
    {faults.KAPPA_C_ENV_VAR: "0"}, {faults.KAPPA_R_ARM1_ENV_VAR: "1.5"}, {faults.SPAWN_LAYOUT_ENV_VAR: "grid"},
    {faults.REACH_MODE_ENV_VAR: "ignore"}, {faults.REACH_STALL_ENV_VAR: "0"},
    {faults.PICKS_PER_MIN_ENV_VAR: "0"}, {faults.HESITATIONS_ENV_VAR: "-1"},
])
def test_from_env_rejects_bad_values(bad):
    with pytest.raises(ValueError):
        FaultConfig.from_env(bad)


def test_reach_span_is_the_stop_line_edges():
    # travel along -x, the zone is x in [-4, -2], y in [-0.65, 0.25]; stop 80 % through -> x = -3.6
    near, far = pick_zone_reach_span((-3.0, -0.2), (1.0, 0.45), 0, -1.0, 0.8, (-3.0, 1.0928))
    assert near == pytest.approx(math.hypot(0.6, 1.0928 - 0.25))
    assert far == pytest.approx(math.hypot(0.6, 1.0928 + 0.65))


def test_reach_for_kappa_interpolates_and_caps():
    assert reach_for_kappa(1.0, 1.0, 2.0) == faults.NOMINAL_MAX_REACH_M
    assert reach_for_kappa(0.5, 1.0, 1.4) == pytest.approx(1.2)
    assert reach_for_kappa(0.9, 1.0, 3.0) == faults.NOMINAL_MAX_REACH_M  # never beyond the arm's spec reach
    # the demonstration's numbers: near box ~1.2 m is served at 0.3, the far one ~1.66 m is not
    reach = reach_for_kappa(0.3, 1.04, 1.86)
    assert 1.2 < reach < 1.66


def test_reach_mode_attempt_needs_a_real_reach_fault():
    nominal = FaultConfig.from_env({faults.REACH_MODE_ENV_VAR: "attempt"})
    assert not nominal.attempts_unreachable and nominal.arm1_reach_fault(1.75, (0, 0), 1 / 120) is None
    cfg = FaultConfig.from_env({faults.REACH_MODE_ENV_VAR: "attempt", faults.KAPPA_R_ARM1_ENV_VAR: "0.3"})
    assert cfg.attempts_unreachable and "reach_mode=attempt" in cfg.describe()
    fault = cfg.arm1_reach_fault(1.11, (-3.0, 1.0928), 1 / 120)
    assert isinstance(fault, ReachFault) and fault.max_reach_m == 1.11
    assert fault.stall_ticks == round(faults.REACH_STALL_S * 120)
    slow = FaultConfig.from_env({faults.REACH_MODE_ENV_VAR: "attempt", faults.KAPPA_R_ARM1_ENV_VAR: "0.3", faults.REACH_STALL_ENV_VAR: "2.5"})
    assert slow.arm1_reach_fault(1.11, (0, 0), 1 / 120).stall_ticks == 300 and "reach_stall_s=2.5" in slow.describe()


def test_reach_fault_clips_a_far_target_to_the_limit_and_stalls_for_its_ticks():
    fault = ReachFault(1.11, (-3.0, 1.0928), stall_ticks=3)
    near = (-3.6, 0.28, 1.99)  # ~1.01 m: served as is
    assert not fault.beyond(near) and fault.clip(near) is near
    far = (-3.6, -0.252, 1.99)  # the far box at the stop line, ~1.47 m
    assert fault.beyond(far)
    clipped = fault.clip(far)
    assert fault.distance_m(clipped) == pytest.approx(1.11) and clipped[2] == 1.99
    # on the base-to-box line, short of the box
    assert (clipped[0] + 3.0) / (far[0] + 3.0) == pytest.approx((clipped[1] - 1.0928) / (far[1] - 1.0928))
    assert fault.begin_stall(clipped) == 1 and fault.last_distance_m == pytest.approx(1.11)
    assert [fault.stalling() for _ in range(3)] == [True, True, False]
    assert fault.begin_stall((-3.4, -0.05, 2.2)) == 2  # the next descent counts again
    with pytest.raises(ValueError):
        ReachFault(0.0, (0, 0), 1)


def test_hesitation_rate_limits_cycles_and_freezes_the_descent():
    assert FaultConfig.from_env({}).hesitation(1, 1 / 120) is None
    cfg = FaultConfig.from_env({faults.PICKS_PER_MIN_ENV_VAR: "2", faults.HESITATIONS_ENV_VAR: "2", faults.FAULT_SEED_ENV_VAR: "5"})
    assert "picks_per_min=2" in cfg.describe() and "hesitations=2" in cfg.describe()
    h = cfg.hesitation(1, 1 / 120)
    assert isinstance(h, Hesitation) and h.period_ticks == 3600 and h.stops == 2
    # one cycle per 30 s of ticks
    assert h.can_start(0)
    h.start_cycle(100)
    assert not h.can_start(3699) and h.can_start(3700)
    # the descent freezes twice, each for 120..360 ticks, never before the window;
    # the plan's progress only advances on the ticks the arm moves
    progress, frozen = 0.0, []
    while progress < 1.0:
        f = h.frozen(progress)
        frozen.append(f)
        if not f:
            progress += 1 / 1000
    assert h.freezes == 2 and not any(frozen[:150])
    runs = []
    for i, f in enumerate(frozen):
        if f and (i == 0 or not frozen[i - 1]):
            runs.append(0)
        if f:
            runs[-1] += 1
    assert len(runs) == 2 and all(120 <= r <= 360 for r in runs)
    h.reset()
    assert not any(h.frozen(p / 10) for p in range(11))
    # hesitations without a rate limit start every cycle at once
    free = FaultConfig.from_env({faults.HESITATIONS_ENV_VAR: "1"}).hesitation(2, 1 / 120)
    free.start_cycle(0)
    assert free.can_start(1)
    with pytest.raises(ValueError):
        Hesitation(-1.0, 0, faults.random.Random(0), 1 / 120)


def test_reach_gate_is_a_plain_flag():
    gate = ReachGate()
    assert gate() is False
    gate.unreachable = True
    assert gate() is True


class ScriptedRng:
    def __init__(self, randoms, uniforms):
        self._randoms = list(randoms)
        self._uniforms = list(uniforms)

    def random(self):
        return self._randoms.pop(0)

    def uniform(self, lo, hi):
        v = self._uniforms.pop(0)
        assert lo <= v <= hi
        return v


def test_hold_fault_fires_once_at_its_swing_fraction():
    fault = HoldFault(0.3, ScriptedRng([0.9], [0.4]))
    assert fault.on_attach() is True
    assert [fault.should_drop(p) for p in (0.0, 0.2, 0.39, 0.4, 0.6, 1.0)] == [False, False, False, True, False, False]
    assert (fault.holds, fault.drops) == (1, 1)


def test_hold_fault_survives_when_the_draw_is_under_kappa():
    fault = HoldFault(0.3, ScriptedRng([0.1], []))
    assert fault.on_attach() is False
    assert not any(fault.should_drop(i / 100) for i in range(101))
    assert (fault.holds, fault.drops) == (1, 0)


def test_hold_fault_always_drop_ignores_the_draw_and_stays_in_the_window():
    fault = HoldFault(1.0, ScriptedRng([], [0.3, 0.5]), always_drop=True)
    assert fault.on_attach() is True and fault.should_drop(0.3)
    assert fault.on_attach() is True and not fault.should_drop(0.49) and fault.should_drop(0.5)
    assert FaultConfig.from_env({faults.DROP_EVERY_HOLD_ENV_VAR: "1"}).arm1_hold_fault().always_drop


def test_hold_fault_reset_cancels_a_pending_drop():
    fault = HoldFault(0.3, ScriptedRng([0.9], [0.4]))
    fault.on_attach()
    fault.reset()
    assert not fault.should_drop(1.0)


def test_near_far_pair_fits_the_belt_without_overlap():
    lateral_half, clearance = 0.45, 0.03
    halves = [(0.13, 0.13, 0.13), (0.13, 0.13, 0.13)]
    near, far = near_far_lateral_offsets(lateral_half, halves, 1.0, clearance)
    assert near > 0 > far  # near is on the robot's side (positive)
    assert abs(near) + 0.13 <= lateral_half and abs(far) + 0.13 <= lateral_half
    assert near - far > 2 * 0.13 + clearance
    # flipping the robot's side mirrors the pair
    assert near_far_lateral_offsets(lateral_half, halves, -1.0, clearance) == [-near, -far]
    # the far box alone lands where it does in the pair
    assert near_far_lateral_offsets(lateral_half, halves[1:], 1.0, clearance, fractions=faults.NEAR_FAR_LATERAL_FRACTIONS[1:]) == [far]
