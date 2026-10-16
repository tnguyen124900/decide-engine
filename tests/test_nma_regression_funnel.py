import math

import numpy as np
import pytest

from heor_tree_engine.nma_funnel import comparison_adjusted_funnel
from heor_tree_engine.nma_parser import resolve_nma
from heor_tree_engine.nma_regression import (
    BASELINE_RISK,
    baseline_risk_values,
    effect_curve,
    effects_at,
    run_meta_regression,
)


def _contrast_doc(rows, reference="A"):
    return {"spec": {"outcome": "contrast", "measure": "HR", "reference": reference},
            "studies": [{"id": sid, "contrasts": [{"treatment": t, "versus": v, "estimate": e, "se": se}]}
                        for sid, t, v, e, se in rows]}


def test_pairwise_fixed_effect_regression_equals_weighted_least_squares() -> None:
    rows = [("s1", "B", "A", -0.5, 0.20), ("s2", "B", "A", -0.2, 0.25), ("s3", "B", "A", -0.8, 0.15),
            ("s4", "B", "A", -0.1, 0.30), ("s5", "B", "A", -0.6, 0.22)]
    x = {"s1": 60, "s2": 70, "s3": 55, "s4": 75, "s5": 62}
    result = run_meta_regression(resolve_nma(_contrast_doc(rows)), "age", x, "shared", "fixed")

    y = np.array([r[3] for r in rows])
    w = np.array([1 / r[4] ** 2 for r in rows])
    xc = np.array([x[r[0]] for r in rows]) - np.mean(list(x.values()))
    D = np.column_stack([np.ones(5), xc])
    cov = np.linalg.inv(D.T @ (w[:, None] * D))
    beta = cov @ D.T @ (w * y)
    assert result.errors == []
    assert result.coefficients[0].estimate == pytest.approx(beta[1], rel=1e-10)
    assert result.coefficients[0].se == pytest.approx(math.sqrt(cov[1, 1]), rel=1e-10)
    assert effects_at(result, result.centre)[0].estimate == pytest.approx(beta[0], rel=1e-10)


def test_exact_linear_covariate_effect_is_recovered_in_a_network() -> None:
    d = {"A": 0.0, "B": -0.4, "C": -0.1}
    beta = 0.02
    ages = {f"s{i}": 50 + 3 * i for i in range(8)}
    mean_age = np.mean(list(ages.values()))
    pairs = [("B", "A"), ("C", "A"), ("C", "B"), ("B", "A"), ("C", "A"), ("C", "B"), ("B", "A"), ("C", "A")]
    rows = []
    for i, (t, v) in enumerate(pairs):
        xc = ages[f"s{i}"] - mean_age
        effect = (d[t] + (beta * xc if t != "A" else 0)) - (d[v] + (beta * xc if v != "A" else 0))
        rows.append((f"s{i}", t, v, effect, 0.1 + 0.02 * i))
    result = run_meta_regression(resolve_nma(_contrast_doc(rows)), "age", ages, "shared", "random")

    assert result.coefficients[0].estimate == pytest.approx(beta, abs=1e-10)
    assert result.q_residual == pytest.approx(0.0, abs=1e-18)
    at_mean = {e.treatment: e.estimate for e in effects_at(result, mean_age)}
    assert at_mean == pytest.approx({"B": -0.4, "C": -0.1}, abs=1e-10)
    older = {e.treatment: e.estimate for e in effects_at(result, mean_age + 10)}
    assert older["B"] == pytest.approx(-0.4 + 10 * beta, abs=1e-10)


def test_specific_interaction_and_non_estimable_cases() -> None:
    rows = [("s1", "B", "A", -0.5, 0.2), ("s2", "C", "A", -0.1, 0.2), ("s3", "C", "B", 0.3, 0.2),
            ("s4", "B", "A", -0.3, 0.2), ("s5", "C", "A", 0.0, 0.2)]
    model = resolve_nma(_contrast_doc(rows))
    x = {"s1": 1, "s2": 2, "s3": 3, "s4": 4, "s5": 5}
    specific = run_meta_regression(model, "x", x, "specific", "fixed")
    assert specific.errors == [] and [c.treatment for c in specific.coefficients] == ["B", "C"]

    constant = run_meta_regression(model, "x", {k: 1.0 for k in x}, "shared", "fixed")
    assert any(e.code == "beta_not_estimable" for e in constant.errors)
    too_few = run_meta_regression(model, "x", {"s1": 1, "s2": 2}, "shared", "fixed")
    assert any(e.code == "too_few_studies" for e in too_few.errors)
    partial = run_meta_regression(model, "x", {"s1": 1, "s2": 2, "s3": 3, "s4": 5}, "shared", "fixed")
    assert partial.excluded_studies == ["s5"]
    curve = effect_curve(specific, "B", points=5)
    assert len(curve) == 5 and curve[0].at == 1 and curve[-1].at == 5


def test_baseline_risk_uses_reference_arm_log_odds() -> None:
    doc = {"spec": {"outcome": "binary", "reference": "P"}, "studies": [
        {"id": "s1", "arms": [{"treatment": "P", "events": 20, "n": 100}, {"treatment": "A", "events": 10, "n": 100}]},
        {"id": "s2", "arms": [{"treatment": "A", "events": 8, "n": 50}, {"treatment": "B", "events": 9, "n": 50}]},
        {"id": "s3", "arms": [{"treatment": "P", "events": 0, "n": 40}, {"treatment": "B", "events": 3, "n": 40}]},
    ]}
    values = baseline_risk_values(resolve_nma(doc))
    assert values["s1"] == pytest.approx(math.log(20 / 80))
    assert "s2" not in values
    assert values["s3"] == pytest.approx(math.log(0.5 / 40.5))  # zero-cell corrected
    result = run_meta_regression(resolve_nma(doc), BASELINE_RISK, None, "shared", "fixed")
    assert any(e.code == "too_few_studies" for e in result.errors)


def _funnel_doc():
    rows = [("s1", "B", "A", -0.5, 0.10), ("s2", "B", "A", -0.9, 0.35), ("s3", "B", "A", -0.4, 0.12),
            ("s4", "C", "A", -0.2, 0.15), ("s5", "C", "A", -0.7, 0.40), ("s6", "C", "B", 0.3, 0.2)]
    return _contrast_doc(rows)


def test_funnel_points_are_centred_and_oriented_by_order() -> None:
    model = resolve_nma(_funnel_doc())
    funnel = comparison_adjusted_funnel(model)

    assert len(funnel.points) == 6
    for key, mean in funnel.comparison_means.items():
        pts = [p for p in funnel.points if (p.newer, p.older) == key]
        weights = [1 / p.se ** 2 for p in pts]
        assert sum(w * p.centred for w, p in zip(weights, pts)) == pytest.approx(0.0, abs=1e-12)
    assert {(p.newer, p.older) for p in funnel.points} == {("B", "A"), ("C", "A"), ("C", "B")}

    reversed_funnel = comparison_adjusted_funnel(model, order=["C", "B", "A"])
    assert {(p.newer, p.older) for p in reversed_funnel.points} == {("A", "B"), ("A", "C"), ("B", "C")}
    assert sorted(p.centred for p in reversed_funnel.points) == pytest.approx(
        sorted(-p.centred for p in funnel.points))
    assert any("little power" in w for w in funnel.warnings)


def test_egger_intercept_matches_direct_regression_and_detects_asymmetry() -> None:
    rng = np.random.default_rng(4)
    rows = []
    for i in range(40):
        se = rng.uniform(0.05, 0.5)
        bias = 1.5 * se  # small (high-SE) studies look more favourable to B
        rows.append((f"s{i}", "B", "A", -0.3 + bias + rng.normal(0, se), se))
    funnel = comparison_adjusted_funnel(resolve_nma(_contrast_doc(rows)))

    se = np.array([p.se for p in funnel.points])
    z = np.array([p.centred for p in funnel.points]) / se
    slope, intercept = np.polyfit(1 / se, z, 1)
    assert funnel.egger_intercept == pytest.approx(intercept, rel=1e-8)
    assert funnel.egger_p < 0.01

    symmetric = [(f"s{i}", "B", "A", -0.3 + rng.normal(0, s), s) for i, s in enumerate(rng.uniform(0.05, 0.5, 40))]
    assert comparison_adjusted_funnel(resolve_nma(_contrast_doc(symmetric))).egger_p > 0.01
