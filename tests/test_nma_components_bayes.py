"""Tests for nma_components_bayes (Bayesian component NMA)."""

from __future__ import annotations

import numpy as np
import pytest

from heor_tree_engine.nma_components_bayes import (
    BayesComponentSettings,
    run_component_nma_bayes,
)
from heor_tree_engine.nma_bayes import BayesSettings, run_nma_bayesian
from heor_tree_engine.nma_parser import resolve_nma


def _network() -> dict:
    """A 4-treatment network where two treatments share a component.

    A = reference (no components), B = drug1, C = drug2,
    D = drug1 + counselling -- so "counselling" is the add-on component the
    additive model should recover from the B-vs-D contrast.
    """

    studies = []
    for i, (study, t1, t2, e1, n1, e2, n2) in enumerate([
        ("S1", "A", "B", 30, 100, 20, 100),
        ("S2", "A", "B", 28, 98, 22, 102),
        ("S3", "A", "C", 32, 105, 25, 100),
        ("S4", "B", "D", 20, 100, 12, 100),
        ("S5", "C", "D", 24, 98, 15, 101),
        ("S6", "A", "D", 35, 110, 18, 108),
    ]):
        studies.append({"id": study, "arms": [
            {"treatment": t1, "events": e1, "n": n1},
            {"treatment": t2, "events": e2, "n": n2},
        ]})
    return {
        "spec": {"outcome": "binary", "measure": "OR", "reference": "A", "small_values": "good"},
        "studies": studies,
    }


def _settings(**over) -> BayesComponentSettings:
    bayes = BayesSettings(effects="random", chains=4, draws_per_chain=2000, seed=2026)
    return BayesComponentSettings(bayes=bayes, **over)


def test_additive_runs_and_recovers_components() -> None:
    components = {"B": "drug1", "C": "drug2", "D": ["drug1", "counselling"]}
    result = run_component_nma_bayes(resolve_nma(_network()), components, _settings())
    assert result.errors == []
    # drug1, drug2, counselling = 3 components; no interactions.
    assert set(result.components) == {"drug1", "drug2", "counselling"}
    assert result.component_draws is not None
    assert result.component_draws.shape[1] == 3
    assert result.treatment_draws is not None
    # Implied treatment effects: 3 treatments vs reference A.
    assert result.treatment_draws.shape[1] == 3


def test_membership_defaults_to_own_name() -> None:
    """A treatment with no components entered becomes its own component."""
    result = run_component_nma_bayes(resolve_nma(_network()), {"B": "drug1"}, _settings())
    assert result.errors == []
    # drug1 (B), plus C and D as their own components.
    assert "drug1" in result.components


def test_implied_effects_match_standard_when_no_sharing() -> None:
    """With no shared components (every treatment its own component), the
    additive model IS the standard NMA -- implied treatment effects should
    land close to the standard Bayesian NMA's posterior means."""

    document = resolve_nma(_network())
    components = {"B": "B", "C": "C", "D": "D"}
    comp = run_component_nma_bayes(document, components, _settings())
    std = run_nma_bayesian(document, BayesSettings(effects="random", chains=4, draws_per_chain=2000, seed=2026))
    assert comp.errors == [] and std.errors == []
    comp_means = comp.treatment_draws.mean(axis=0)
    std_means = std.draws.mean(axis=0)
    # Same model, same data, same seed family: posterior means within MCMC
    # tolerance (5% rel or 0.05 abs on the log-OR scale).
    np.testing.assert_allclose(comp_means, std_means, rtol=0.05, atol=0.05)


def test_interaction_adds_a_column() -> None:
    components = {"B": "drug1", "C": "drug2", "D": ["drug1", "drug2"]}
    settings = _settings(interactions=[("drug1", "drug2")])
    result = run_component_nma_bayes(resolve_nma(_network()), components, settings)
    assert result.errors == []
    # 2 base components + 1 interaction column.
    assert "drug1:drug2" in result.components
    assert result.component_draws.shape[1] == 3


def test_unidentifiable_components_flagged() -> None:
    """Two components that ALWAYS occur together can't be told apart."""
    document = {
        "spec": {"outcome": "binary", "measure": "OR", "reference": "A", "small_values": "good"},
        "studies": [
            {"id": "S1", "arms": [{"treatment": "A", "events": 30, "n": 100},
                                  {"treatment": "B", "events": 20, "n": 100}]},
            {"id": "S2", "arms": [{"treatment": "A", "events": 28, "n": 98},
                                  {"treatment": "B", "events": 22, "n": 102}]},
        ],
    }
    # B always = x+y together, never separately -> not identifiable.
    components = {"B": ["x", "y"]}
    result = run_component_nma_bayes(resolve_nma(document), components, _settings())
    assert any(getattr(e, "code", "") == "components_not_identified" for e in result.errors)


def test_sucra_present() -> None:
    components = {"B": "drug1", "C": "drug2", "D": ["drug1", "counselling"]}
    result = run_component_nma_bayes(resolve_nma(_network()), components, _settings())
    assert result.errors == []
    assert set(result.sucra) == {"A", "B", "C", "D"}
    for value in result.sucra.values():
        assert 0.0 <= value <= 1.0


def test_fixed_effect_posterior_matches_frequentist_component_effects() -> None:
    """With fixed effects and a vague prior the posterior of the component effects is the GLS
    estimate (the exact conjugate result), up to Monte-Carlo error."""

    from heor_tree_engine.nma_components import run_component_nma

    model = resolve_nma(_network())
    comps = {"D": "B + counselling"}
    freq = run_component_nma(model, comps, "fixed")
    bayes = run_component_nma_bayes(model, comps, BayesComponentSettings(
        bayes=BayesSettings(effects="fixed", chains=4, draws_per_chain=4000, seed=7)))
    assert not freq.errors and not bayes.errors
    f = {e.treatment: e for e in freq.component_effects}
    b = {e.treatment: e for e in bayes.component_effects}
    for name in f:
        assert b[name].estimate == pytest.approx(f[name].estimate, abs=0.03), name
        assert b[name].se == pytest.approx(f[name].se, rel=0.1), name
