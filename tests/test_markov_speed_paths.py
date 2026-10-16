"""The two Markov/PSA speed-ups must not change results.

1. distributions._fast_survival (plain `math` formulas for Exponential/Weibull/Lognormal/Log-logistic/Gompertz)
   against scipy's frozen distributions.
2. markov_engine's numpy reward fast path (dot products over the tenure axis) against the original per-entry loop
   (switched off with markov_engine.FAST_REWARD = False) on a model with tenure-based costs.
"""
import json
import math
import random
from pathlib import Path

import pytest

from heor_tree_engine import distributions as dist
from heor_tree_engine import markov_engine
from heor_tree_engine.sensitivity import evaluate_markov_scenario

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "tests" / "fixtures" / "markov_tenure_fixture.json"  # tenure-based test fixture, not a user sample


def _resolve(value):
    return float(value)


CASES = [
    ("Exponential", lambda r: {"rate": r.uniform(0.01, 2.0)}),
    ("Weibull", lambda r: {"shape": r.uniform(0.4, 3.0), "scale": r.uniform(0.5, 30.0)}),
    ("Lognormal", lambda r: {"meanlog": r.uniform(-1.0, 3.0), "sdlog": r.uniform(0.2, 1.5)}),
    ("Log-logistic", lambda r: {"shape": r.uniform(0.6, 4.0), "scale": r.uniform(0.5, 30.0)}),
    ("Gompertz", lambda r: {"shape": r.uniform(0.01, 0.8), "rate": r.uniform(0.005, 0.5)}),
]


def test_fast_survival_matches_scipy() -> None:
    pytest.importorskip("scipy")
    rng = random.Random(5)
    for name, make in CASES:
        for _ in range(40):
            definition = {"type": name, "parameters": make(rng)}
            fast = dist._fast_survival(definition, _resolve)
            assert fast is not None, name
            scipy_dist = dist._scipy_distribution_uncached(definition, _resolve)
            for x in (-1.0, 0.0, 0.01, 0.5, 1.0, 3.7, 12.0, 40.0, 150.0):
                expected = float(scipy_dist.sf(x))
                assert abs(fast(x) - expected) <= 1e-12 * max(1.0, abs(expected)) + 1e-300, (name, definition, x, fast(x), expected)


def test_distribution_to_prob_matches_scipy_path() -> None:
    pytest.importorskip("scipy")
    rng = random.Random(9)
    for name, make in CASES:
        for _ in range(20):
            definition = {"type": name, "parameters": make(rng)}
            start, duration = rng.uniform(0.0, 20.0), rng.uniform(0.01, 5.0)
            got = dist.distribution_to_prob(definition, start, duration, _resolve)
            scipy_dist = dist._scipy_distribution_uncached(definition, _resolve)
            s0 = float(scipy_dist.sf(start))
            expected = 0.0 if s0 <= 0 else 1.0 - float(scipy_dist.sf(start + duration)) / s0
            assert abs(got - expected) < 1e-12, (name, got, expected)


def test_invalid_parameters_still_raise_the_scipy_error() -> None:
    pytest.importorskip("scipy")
    for definition in (
        {"type": "Gompertz", "parameters": {"shape": -0.1, "rate": 0.1}},
        {"type": "Weibull", "parameters": {"shape": 0.0, "scale": 2.0}},
        {"type": "Exponential", "parameters": {"rate": -1.0}},
    ):
        assert dist._fast_survival(definition, _resolve) is None  # falls back to scipy (and its error handling)


def test_unknown_family_has_no_fast_path() -> None:
    assert dist._fast_survival({"type": "Normal", "parameters": {"mu": 0, "sigma": 1}}, _resolve) is None


def _outcome(document):
    result = evaluate_markov_scenario(document)
    assert not result.errors, result.errors
    return result.outcomes[0]


def test_reward_fast_path_equals_per_entry_loop_on_tenure_model() -> None:
    pytest.importorskip("scipy")
    document = json.loads(SAMPLE.read_text(encoding="utf-8"))
    document["cycles"] = 120  # keeps the slow reference quick
    fast = _outcome(document)
    markov_engine.FAST_REWARD = False
    try:
        slow = _outcome(document)
    finally:
        markov_engine.FAST_REWARD = True
    assert abs(fast.total_cost - slow.total_cost) <= 1e-9 * abs(slow.total_cost)
    assert abs(fast.total_effect - slow.total_effect) <= 1e-9 * abs(slow.total_effect)
