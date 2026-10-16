"""Guyot KM reconstruction (C1c-1): simulated exponential data with censoring, exact and coarsely digitised curves."""

import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from heor_tree_engine.nma_km_reconstruct import interval_counts, kaplan_meier, reconstruct_arm  # noqa: E402

RISK_TIMES = [0, 5, 10, 15, 20, 25]


def _simulate(seed=3, n=200, rate=0.08):
    rng = random.Random(seed)
    latent = [rng.expovariate(rate) for _ in range(n)]
    censor = [rng.uniform(5, 30) for _ in range(n)]
    time = [min(a, b) for a, b in zip(latent, censor)]
    event = [1 if a <= b else 0 for a, b in zip(latent, censor)]
    return time, event


def _at_risk(time):
    return [sum(1 for x in time if x >= t) for t in RISK_TIMES]


def test_exact_curve_recovers_events_and_rate():
    time, event = _simulate()
    arm = reconstruct_arm(kaplan_meier(time, event), RISK_TIMES, _at_risk(time), sum(event))
    assert arm.n == 200 and arm.warnings == []
    assert arm.n_events == sum(event)
    true_rate = sum(event) / sum(time)
    assert arm.n_events / sum(arm.times) == pytest.approx(true_rate, rel=0.03)
    counts = interval_counts(arm, [0, 5, 10, 15, 20, 25, 30])
    truth = [sum(1 for t, e in zip(time, event) if e and a < t <= a + 5) for a in (0, 5, 10, 15, 20, 25)]
    assert [c[2] for c in counts] == pytest.approx(truth, abs=2)


def test_reconstructed_curve_reproduces_the_input_curve():
    time, event = _simulate(seed=5)
    curve = kaplan_meier(time, event)
    arm = reconstruct_arm(curve, RISK_TIMES, _at_risk(time), sum(event))
    rebuilt = kaplan_meier(arm.times, arm.events)

    def at(c, t):
        return [s for (tt, s) in c if tt <= t][-1]

    for t in (2, 5, 8, 12, 16, 20):
        assert at(rebuilt, t) == pytest.approx(at(curve, t), abs=0.02)


def test_coarsely_digitised_curve_is_close():
    time, event = _simulate(seed=11)
    curve = kaplan_meier(time, event)
    coarse = [(round(t, 1), round(s, 2)) for (t, s) in curve[::4]]  # every 4th step, 2-decimal survival
    arm = reconstruct_arm(coarse, RISK_TIMES, _at_risk(time), sum(event))
    true_rate = sum(event) / sum(time)
    assert arm.n_events / sum(arm.times) == pytest.approx(true_rate, rel=0.12)
    assert arm.warnings  # event total differs from the reported one: the user is warned


def test_bad_risk_table_is_reported():
    arm = reconstruct_arm([(0, 1.0), (5, 0.9)], [1, 5], [100, 80])
    assert arm.warnings and arm.n == 0
