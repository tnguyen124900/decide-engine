"""Tests for nma_voi (Value of Information from an NMA posterior)."""

from __future__ import annotations

import numpy as np
import pytest

from heor_tree_engine.nma_bayes import BayesSettings, run_nma_bayesian
from heor_tree_engine.nma_parser import resolve_nma
from heor_tree_engine.nma_voi import run_nma_voi


def _network() -> dict:
    """A 3-treatment network with a clear winner (C much better)."""
    studies = []
    for study, t1, t2, e1, n1, e2, n2 in [
        ("S1", "A", "B", 40, 100, 30, 100),
        ("S2", "A", "C", 42, 102, 15, 100),
        ("S3", "B", "C", 35, 100, 18, 101),
        ("S4", "A", "B", 38, 99, 32, 100),
        ("S5", "A", "C", 45, 105, 14, 99),
        ("S6", "B", "C", 33, 98, 16, 100),
    ]:
        studies.append({"id": study, "arms": [
            {"treatment": t1, "events": e1, "n": n1},
            {"treatment": t2, "events": e2, "n": n2},
        ]})
    return {
        "spec": {"outcome": "binary", "measure": "OR", "reference": "A", "small_values": "good"},
        "studies": studies,
    }


def _bayes() -> "object":
    model = resolve_nma(_network())
    return model, run_nma_bayesian(model, BayesSettings(effects="random", chains=4, draws_per_chain=2000, seed=7))


def test_voi_runs_effect_scale() -> None:
    model, bayes = _bayes()
    result = run_nma_voi(model, bayes)
    assert result.payoff_scale == "effect"
    assert result.n_draws == bayes.draws.shape[0]
    assert result.evpi >= 0.0
    assert result.optimal_treatment in model.treatment_names


def test_voi_net_benefit_scale() -> None:
    model, bayes = _bayes()
    costs = {"A": 100.0, "B": 200.0, "C": 150.0}
    result = run_nma_voi(model, bayes, willingness_to_pay=1000.0, costs=costs)
    assert result.payoff_scale == "net_benefit"
    assert result.evpi >= 0.0


def test_evpi_zero_when_one_treatment_dominates() -> None:
    """A network where C wins on essentially every draw has ~no decision
    uncertainty, so EVPI should be near zero."""
    # Make C overwhelmingly better.
    document = _network()
    for study in document["studies"]:
        for arm in study["arms"]:
            if arm["treatment"] == "C":
                arm["events"] = 2.0  # almost no events on C
    model = resolve_nma(document)
    bayes = run_nma_bayesian(model, BayesSettings(effects="random", chains=4, draws_per_chain=2000, seed=11))
    result = run_nma_voi(model, bayes)
    assert result.evpi < 0.05  # small on the effect scale


def test_evppi_sorted_and_bounded() -> None:
    model, bayes = _bayes()
    result = run_nma_voi(model, bayes)
    if result.evpi > 0 and result.evppi:
        evppi_values = [row.evppi for row in result.evppi]
        assert evppi_values == sorted(evppi_values, reverse=True)
        for row in result.evppi:
            assert 0.0 <= row.evppi <= result.evpi + 1e-9
            if row.percent_of_evpi is not None:
                assert 0.0 <= row.percent_of_evpi <= 100.0 + 1e-6


def test_voi_warns_without_draws() -> None:
    model, bayes = _bayes()
    bayes.draws = None
    result = run_nma_voi(model, bayes)
    assert result.warnings != []


def test_voi_warns_on_nonpositive_wtp() -> None:
    model, bayes = _bayes()
    result = run_nma_voi(model, bayes, willingness_to_pay=0.0)
    assert result.warnings != []
