"""Exact-likelihood Bayesian NMA (MCMC) and the UME model.

Checks against closed forms / numerical integration rather than other
software: a one-study posterior on a 2-D grid (binomial and Poisson), the
normal-likelihood model against the exact DIRECT sampler, and the UME model
picking up a planted inconsistency.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import nma_editor_support as sup  # noqa: E402
from heor_tree_engine.nma_bayes import BayesSettings, run_nma_bayesian  # noqa: E402
from heor_tree_engine.nma_bayes_exact import exact_likelihood, run_nma_bayesian_exact  # noqa: E402
from heor_tree_engine.nma_parser import resolve_nma  # noqa: E402

FAST = dict(chains=3, draws_per_chain=1200, seed=3)


def _grid_posterior(loglik):
    mu = np.linspace(-9, 4, 1301)
    d = np.linspace(-6, 8, 1401)
    M, D = np.meshgrid(mu, d, indexing="ij")
    lp = loglik(M, D) - 0.5 * (M / 100) ** 2 - 0.5 * (D / 100) ** 2
    w = np.exp(lp - lp.max()).sum(axis=0)
    w /= w.sum()
    mean = float((w * d).sum())
    return mean, math.sqrt(float((w * (d - mean) ** 2).sum()))


def test_binomial_one_study_matches_numerical_integration() -> None:
    model = resolve_nma({"spec": {"outcome": "binary", "measure": "OR", "reference": "P"},
                         "treatments": [{"name": "P"}, {"name": "A"}],
                         "studies": [{"id": "s", "arms": [{"treatment": "P", "events": 3, "n": 20},
                                                          {"treatment": "A", "events": 9, "n": 21}]}]})
    result = run_nma_bayesian_exact(model, BayesSettings(effects="fixed", **FAST), warmup=500)
    mean, sd = _grid_posterior(lambda M, D: 3 * M - 20 * np.logaddexp(0, M) + 9 * (M + D) - 21 * np.logaddexp(0, M + D))
    assert result.errors == []
    assert result.draws[:, 0].mean() == pytest.approx(mean, abs=0.06)
    assert result.draws[:, 0].std() == pytest.approx(sd, rel=0.06)
    assert result.diagnostics[0].rhat < 1.02


def test_poisson_one_study_matches_numerical_integration() -> None:
    model = resolve_nma({"spec": {"outcome": "rate", "measure": "RATE_RATIO", "reference": "P"},
                         "treatments": [{"name": "P"}, {"name": "A"}],
                         "studies": [{"id": "s", "arms": [{"treatment": "P", "events": 12, "exposure": 100},
                                                          {"treatment": "A", "events": 5, "exposure": 90}]}]})
    result = run_nma_bayesian_exact(model, BayesSettings(effects="fixed", **FAST), warmup=500)
    mean, sd = _grid_posterior(lambda M, D: 12 * M - 100 * np.exp(M) + 5 * (M + D) - 90 * np.exp(M + D))
    assert result.draws[:, 0].mean() == pytest.approx(mean, abs=0.05)
    assert result.draws[:, 0].std() == pytest.approx(sd, rel=0.06)


def _continuous_network():
    rng = np.random.default_rng(5)
    true = {"P": 0.0, "A": -2.0, "B": -1.0, "C": -3.0}
    designs = [("P", "A"), ("P", "A"), ("P", "B"), ("P", "B"), ("A", "B"), ("P", "C"), ("B", "C"), ("P", "A", "B"),
               ("A", "C")]
    studies = []
    for i, design in enumerate(designs):
        base = rng.normal(20, 3)
        studies.append({"id": f"s{i}", "arms": [{"treatment": t, "mean": base + true[t] + rng.normal(0, 0.8), "sd": 6.0,
                                                 "n": int(rng.integers(30, 80))} for t in design]})
    return resolve_nma({"spec": {"outcome": "continuous", "measure": "MD", "reference": "P"},
                        "treatments": [{"name": n} for n in "PABC"], "studies": studies})


def _compare_with_direct(effects: str) -> None:
    model = _continuous_network()
    settings = BayesSettings(effects=effects, chains=4, draws_per_chain=1000, seed=11)
    exact = run_nma_bayesian_exact(model, settings, warmup=500)
    direct = run_nma_bayesian(model, settings)
    for a, b in zip(exact.fit.effects, direct.fit.effects):
        assert a.estimate == pytest.approx(b.estimate, abs=0.12)
        assert a.se == pytest.approx(b.se, rel=0.12)
    if effects == "random":
        assert exact.tau.median == pytest.approx(direct.tau.median, abs=0.15)
    assert max(d.rhat for d in exact.diagnostics) < 1.05


def test_normal_likelihood_agrees_with_the_direct_sampler_fixed() -> None:
    _compare_with_direct("fixed")


def test_normal_likelihood_agrees_with_the_direct_sampler_random() -> None:
    _compare_with_direct("random")


def test_zero_cell_needs_no_correction_and_deviance_bookkeeping() -> None:
    document = sup.default_document()
    document["studies"][8]["arms"][1]["events"] = 0  # Drug D arm with no events
    model = resolve_nma(document)
    result = run_nma_bayesian_exact(model, BayesSettings(effects="fixed", **FAST), warmup=500)

    assert result.errors == [] and result.warnings == []
    assert result.n_data_points == 19 and len(result.data_points) == 19  # one per arm
    assert result.chain_draws.shape == (3, 1200, 4) and result.chain_tau is None  # fixed effect: no tau chain
    assert result.parameter_names == ["d[Drug A]", "d[Drug B]", "d[Drug C]", "d[Drug D]"]
    assert sum(point.leverage for point in result.data_points) == pytest.approx(result.p_d)
    assert result.p_d == pytest.approx(13, abs=1.0)  # 9 study baselines + 4 effects
    assert result.dic == pytest.approx(result.residual_deviance + result.p_d)
    assert result.fit.effects[-1].treatment == "Drug D" and result.fit.effects[-1].natural < 0.5


def test_ume_model_detects_a_planted_inconsistency() -> None:
    document = sup.default_document()
    document["studies"][6]["arms"] = [{"treatment": "Drug A", "events": 40, "n": 90},
                                      {"treatment": "Drug B", "events": 5, "n": 92},
                                      {"treatment": "Drug C", "events": 17, "n": 91}]
    model = resolve_nma(document)
    settings = BayesSettings(effects="fixed", **FAST)
    consistency = run_nma_bayesian_exact(model, settings, warmup=500)
    ume = run_nma_bayesian_exact(model, settings, model_type="ume", warmup=500)

    assert consistency.dic - ume.dic > 10
    assert ume.p_d == pytest.approx(consistency.p_d + 3, abs=1.0)  # 7 comparisons vs 4 effects
    assert set(ume.ume_effects) == {("Drug A", "Placebo"), ("Drug B", "Placebo"), ("Drug C", "Placebo"),
                                    ("Drug D", "Placebo"), ("Drug B", "Drug A"), ("Drug C", "Drug A"),
                                    ("Drug D", "Drug B")}
    assert ume.ume_effects[("Drug B", "Drug A")].natural < 0.2  # study 07's own (conflicting) estimate


def test_exact_likelihood_only_for_supported_measures() -> None:
    document = sup.default_document()
    assert exact_likelihood(resolve_nma(document)) == "binomial"
    for measure, kind in (("RR", "binomial-log"), ("RD", "binomial-identity")):
        document["spec"]["measure"] = measure
        assert exact_likelihood(resolve_nma(document)) == kind
    document["spec"]["measure"] = "HR"  # not an arm-level binary measure
    model = resolve_nma(document)
    assert exact_likelihood(model) is None
    result = run_nma_bayesian_exact(model, BayesSettings(**FAST))
    assert "no_exact_likelihood" in [error.code for error in result.errors]


def test_bayesian_node_split_agrees_with_side_and_flags_conflict() -> None:
    from heor_tree_engine.nma_inconsistency import run_inconsistency

    document = sup.default_document()
    document["studies"][6]["arms"] = [{"treatment": "Drug A", "events": 40, "n": 90},
                                      {"treatment": "Drug B", "events": 5, "n": 92},
                                      {"treatment": "Drug C", "events": 17, "n": 91}]
    model = resolve_nma(document)
    settings = BayesSettings(effects="fixed", **FAST)
    run = run_nma_bayesian_exact(model, settings, model_type="nodesplit", warmup=500, split=("Drug B", "Drug A"))
    split = run.split
    frequentist = next(x for x in run_inconsistency(model).splits_fixed if {x.treatment, x.versus} == {"Drug B", "Drug A"})

    assert run.errors == [] and split.n_direct_studies == 1
    assert split.direct.natural == pytest.approx(frequentist.natural_direct, rel=0.25)
    assert split.indirect.natural == pytest.approx(frequentist.natural_indirect, rel=0.2)
    assert split.p_value < 0.01 and split.dic is not None
    # Without the conflict the split shows nothing.
    calm = run_nma_bayesian_exact(resolve_nma(sup.default_document()), settings, model_type="nodesplit", warmup=500,
                                  split=("Drug B", "Drug A"))
    assert calm.split.p_value > 0.3
    bad = run_nma_bayesian_exact(model, settings, model_type="nodesplit", warmup=500)
    assert [e.code for e in bad.errors] == ["missing_split"]


def _one_study(outcome, measure, arms):
    return resolve_nma({"spec": {"outcome": outcome, "measure": measure, "reference": "P"},
                        "treatments": [{"name": "P"}, {"name": "A"}],
                        "studies": [{"id": "s", "arms": [{"treatment": t, **fields} for t, fields in arms]}]})


def test_risk_ratio_log_link_matches_numerical_integration() -> None:
    model = _one_study("binary", "RR", [("P", {"events": 6, "n": 40}), ("A", {"events": 14, "n": 42})])
    assert exact_likelihood(model) == "binomial-log"
    result = run_nma_bayesian_exact(model, BayesSettings(effects="fixed", **FAST), warmup=500)

    def loglik(M, D):  # p = exp(eta), eta < 0 for both arms
        with np.errstate(divide="ignore", invalid="ignore"):
            ll = 6 * M + 34 * np.log1p(-np.exp(M)) + 14 * (M + D) + 28 * np.log1p(-np.exp(M + D))
        return np.where((M < 0) & (M + D < 0), ll, -np.inf)

    mean, sd = _grid_posterior(loglik)
    assert result.errors == []
    assert result.draws[:, 0].mean() == pytest.approx(mean, abs=0.06)
    assert result.draws[:, 0].std() == pytest.approx(sd, rel=0.08)
    assert result.fit.effects[0].natural == pytest.approx(math.exp(mean), rel=0.08)  # RR ≈ 2.2
    assert result.diagnostics[0].rhat < 1.03


def test_risk_difference_identity_link_matches_numerical_integration() -> None:
    model = _one_study("binary", "RD", [("P", {"events": 12, "n": 50}), ("A", {"events": 5, "n": 48})])
    assert exact_likelihood(model) == "binomial-identity"
    result = run_nma_bayesian_exact(model, BayesSettings(effects="fixed", **FAST), warmup=500)

    def loglik(M, D):  # p = eta in (0, 1) for both arms
        with np.errstate(divide="ignore", invalid="ignore"):
            ll = 12 * np.log(M) + 38 * np.log1p(-M) + 5 * np.log(M + D) + 43 * np.log1p(-(M + D))
        return np.where((M > 0) & (M < 1) & (M + D > 0) & (M + D < 1), ll, -np.inf)

    mean, sd = _grid_posterior(loglik)
    assert result.errors == []
    assert result.draws[:, 0].mean() == pytest.approx(mean, abs=0.012)
    assert result.draws[:, 0].std() == pytest.approx(sd, rel=0.08)
    assert result.fit.effects[0].natural == pytest.approx(mean, abs=0.012)  # RD ≈ -0.14
    assert result.diagnostics[0].rhat < 1.03


def test_zero_cells_are_fine_for_risk_difference() -> None:
    document = sup.default_document()
    document["spec"]["measure"] = "RD"
    document["studies"][8]["arms"][1]["events"] = 0
    result = run_nma_bayesian_exact(resolve_nma(document), BayesSettings(effects="fixed", **FAST), warmup=500)
    assert result.errors == [] and result.likelihood == "binomial-identity"
    assert -1 < result.fit.effects[-1].natural < 0  # Drug D lowers the risk
    assert all(-1 <= e.natural <= 1 for e in result.fit.effects)


def _compare_with_direct_measure(measure: str, effects: str, tolerance: float) -> None:
    from heor_tree_engine.nma_bayes import run_nma_bayesian as direct_sampler

    if measure == "SMD":
        model = _continuous_network()
        document = None
    else:
        document = sup.default_document()
        document["spec"]["measure"] = measure
        model = resolve_nma(document)
    if measure == "SMD":
        doc = {"spec": {"outcome": "continuous", "measure": "SMD", "reference": "P"},
               "treatments": [{"name": n} for n in "PABC"],
               "studies": [{"id": f"s{i}", "arms": [{"treatment": arm.treatment, "mean": arm.mean, "sd": arm.sd, "n": arm.n}
                                                    for arm in study.arms]} for i, study in enumerate(model.studies)]}
        model = resolve_nma(doc)
    settings = BayesSettings(effects=effects, chains=4, draws_per_chain=1000, seed=11)
    exact = run_nma_bayesian_exact(model, settings, warmup=500)
    direct = direct_sampler(model, settings)
    assert exact.errors == [] and direct.errors == []
    for a, b in zip(exact.fit.effects, direct.fit.effects):
        assert a.treatment == b.treatment
        assert a.estimate == pytest.approx(b.estimate, abs=tolerance)
        assert a.se == pytest.approx(b.se, rel=0.35)
    assert max(d.rhat for d in exact.diagnostics) < 1.06


def test_risk_ratio_network_agrees_with_the_normal_approximation() -> None:
    _compare_with_direct_measure("RR", "fixed", 0.12)


def test_risk_difference_network_agrees_with_the_normal_approximation() -> None:
    _compare_with_direct_measure("RD", "fixed", 0.03)


def test_smd_network_agrees_with_the_normal_approximation_fixed_and_random() -> None:
    _compare_with_direct_measure("SMD", "fixed", 0.12)
    _compare_with_direct_measure("SMD", "random", 0.15)
