import math

import numpy as np
import pytest

from heor_tree_engine.nma_evidence import (
    APPLY_FORMULAS,
    build_evidence_set,
    merge_into_document,
    parameter_name,
)
from heor_tree_engine.nma_freq import run_nma_frequentist
from heor_tree_engine.nma_parser import resolve_nma
from heor_tree_engine.nma_sensitivity import influence_summary, leave_one_out
from heor_tree_engine.psa import _lognormal_from_mean_se


def _network(measure_outcome="binary"):
    return {
        "spec": {"outcome": measure_outcome, "reference": "Placebo"},
        "studies": [
            {"id": "S1", "arms": [{"treatment": "Placebo", "events": 18, "n": 120}, {"treatment": "Drug A", "events": 11, "n": 118}]},
            {"id": "S2", "arms": [{"treatment": "Placebo", "events": 25, "n": 200}, {"treatment": "Drug A", "events": 14, "n": 196}]},
            {"id": "S3", "arms": [{"treatment": "Placebo", "events": 30, "n": 150}, {"treatment": "Drug B", "events": 20, "n": 148}]},
            {"id": "S4", "arms": [{"treatment": "Drug A", "events": 12, "n": 90}, {"treatment": "Drug B", "events": 15, "n": 92},
                                  {"treatment": "Drug C", "events": 17, "n": 91}]},
            {"id": "S5", "arms": [{"treatment": "Placebo", "events": 16, "n": 90}, {"treatment": "Drug C", "events": 13, "n": 92}]},
        ],
    }


def test_parameter_names_are_valid_identifiers() -> None:
    assert parameter_name("HR", "Drug A", "Placebo") == "hr_drug_a_vs_placebo"
    assert parameter_name("OR", "5-FU + LV", "SoC", prefix="nma_") == "nma_or_5_fu_lv_vs_soc"
    assert parameter_name("MD", "2", "1").isidentifier()


def test_evidence_set_matches_nma_effects_and_covariance() -> None:
    result = run_nma_frequentist(resolve_nma(_network()))
    evidence = build_evidence_set(result, "random")

    fit = {effect.treatment: effect for effect in result.random.effects}
    assert [p.treatment for p in evidence.parameters] == ["Drug A", "Drug B", "Drug C"]
    for parameter in evidence.parameters:
        effect = fit[parameter.treatment]
        assert parameter.base == pytest.approx(effect.natural, rel=1e-12)
        assert parameter.low == pytest.approx(effect.natural_lower, rel=1e-12)
        assert parameter.high == pytest.approx(effect.natural_upper, rel=1e-12)
        # DECIDE's own Lognormal(mean, se) conversion gives back (mu, se) exactly.
        mu, sigma = _lognormal_from_mean_se(parameter.psa_values["mean"], parameter.psa_values["se"])
        assert mu == pytest.approx(effect.estimate, rel=1e-10, abs=1e-12)
        assert sigma == pytest.approx(effect.se, rel=1e-10)
    cov = np.array(result.random.covariance)
    names = result.random.parameter_names
    rho_ab = cov[0, 1] / math.sqrt(cov[0, 0] * cov[1, 1])
    correlations = {(a, b): rho for a, b, rho in evidence.correlations}
    assert correlations[("or_drug_a_vs_placebo", "or_drug_b_vs_placebo")] == pytest.approx(rho_ab, rel=1e-10)
    assert names == ["Drug A", "Drug B", "Drug C"]
    assert evidence.apply_formula == APPLY_FORMULAS["OR"]


def test_evidence_against_another_anchor_uses_league_effects() -> None:
    result = run_nma_frequentist(resolve_nma(_network()))
    evidence = build_evidence_set(result, "fixed", anchor="Drug A", treatments=["Drug B", "Drug C"])

    for parameter in evidence.parameters:
        league = result.fixed.league[(parameter.treatment, "Drug A")]
        assert parameter.mu == pytest.approx(league.estimate, rel=1e-10, abs=1e-12)
        assert parameter.se == pytest.approx(league.se, rel=1e-10)
        assert parameter.versus == "Drug A"


def test_merge_into_document_upserts_parameters_psa_rows_and_correlations() -> None:
    result = run_nma_frequentist(resolve_nma(_network()))
    evidence = build_evidence_set(result, "random")
    document = {
        "parameters": [{"name": "p_base", "base": 0.2}, {"name": "or_drug_a_vs_placebo", "base": 1.0}],
        "psa_parameters": [{"target_id": "param:p_base", "psa_type": "Beta", "values": {"mean": 0.2, "se": 0.02}},
                           {"target_id": "param:or_drug_a_vs_placebo", "psa_type": "Gamma", "values": {}}],
        "psa_correlations": [{"target_a": "param:p_base", "target_b": "param:or_drug_a_vs_placebo", "rho": 0.5}],
    }

    summary = merge_into_document(document, evidence)

    assert summary == {"added": ["or_drug_b_vs_placebo", "or_drug_c_vs_placebo"], "updated": ["or_drug_a_vs_placebo"]}
    names = [row["name"] for row in document["parameters"]]
    assert names == ["p_base", "or_drug_a_vs_placebo", "or_drug_b_vs_placebo", "or_drug_c_vs_placebo"]
    assert document["parameters"][1]["base"] == pytest.approx(evidence.parameters[0].base)
    psa = {row["target_id"]: row for row in document["psa_parameters"]}
    assert psa["param:p_base"]["psa_type"] == "Beta"  # untouched
    assert psa["param:or_drug_a_vs_placebo"]["psa_type"] == "Lognormal"
    pairs = {(row["target_a"], row["target_b"]) for row in document["psa_correlations"]}
    assert ("param:p_base", "param:or_drug_a_vs_placebo") not in pairs  # stale row replaced
    assert len(pairs) == 3  # A-B, A-C, B-C


def test_merged_document_runs_through_decide_psa_with_nma_correlation() -> None:
    from heor_tree_engine.psa import make_sampler, _iman_conover_reorder  # noqa: F401

    result = run_nma_frequentist(resolve_nma(_network()))
    evidence = build_evidence_set(result, "random")
    rng = np.random.default_rng(7)
    draws = np.array([[make_sampler(p.psa_type, p.psa_values)(rng) for p in evidence.parameters] for _ in range(4000)])
    logs = np.log(draws)
    for column, parameter in enumerate(evidence.parameters):
        assert logs[:, column].mean() == pytest.approx(parameter.mu, abs=0.02)
        assert logs[:, column].std() == pytest.approx(parameter.se, rel=0.05)


def test_mean_difference_uses_normal() -> None:
    result = run_nma_frequentist(resolve_nma({
        "spec": {"outcome": "continuous", "reference": "P"},
        "studies": [{"id": "S1", "arms": [{"treatment": "P", "mean": 10.0, "sd": 2.0, "n": 50},
                                          {"treatment": "A", "mean": 8.5, "sd": 2.5, "n": 40}]}],
    }))
    parameter = build_evidence_set(result, "fixed").parameters[0]
    assert parameter.psa_type == "Normal"
    assert parameter.base == pytest.approx(-1.5)
    assert parameter.psa_values == pytest.approx({"mean": -1.5, "se": math.sqrt(4 / 50 + 6.25 / 40)})


def test_leave_one_out_refits_without_each_study() -> None:
    model = resolve_nma(_network())
    full = run_nma_frequentist(model)
    rows = leave_one_out(model, full, "fixed")

    assert [row.study_id for row in rows] == ["S1", "S2", "S3", "S4", "S5"]
    # Dropping S1 equals fitting the network without it.
    without_s1 = _network()
    without_s1["studies"] = without_s1["studies"][1:]
    expected = {e.treatment: e.natural for e in run_nma_frequentist(resolve_nma(without_s1)).fixed.effects}
    assert {t: v[0] for t, v in rows[0].effects.items()} == pytest.approx(expected, rel=1e-12)
    # S3 is the only Placebo-B study, but B stays connected through S4.
    assert "Drug B" in rows[2].effects
    summary = influence_summary(rows, full, "Drug A", "fixed")
    assert summary["min"] <= summary["max"] and summary["max_shift"] >= 0


def test_leave_one_out_reports_not_estimable() -> None:
    doc = {"spec": {"outcome": "binary", "reference": "P"}, "studies": [
        {"id": "S1", "arms": [{"treatment": "P", "events": 10, "n": 100}, {"treatment": "A", "events": 5, "n": 100}]},
        {"id": "S2", "arms": [{"treatment": "A", "events": 8, "n": 100}, {"treatment": "B", "events": 4, "n": 100}]},
    ]}
    rows = leave_one_out(resolve_nma(doc))
    assert rows[0].error and "reference" in rows[0].error
    assert rows[1].error is None and set(rows[1].effects) == {"A"}


def test_exclude_studies_refits_without_them() -> None:
    from heor_tree_engine.nma_sensitivity import exclude_studies

    model = resolve_nma(_network())
    reduced = exclude_studies(model, ["S1"])
    without = _network()
    without["studies"] = without["studies"][1:]
    expected = run_nma_frequentist(resolve_nma(without))
    assert [e.estimate for e in reduced.fixed.effects] == pytest.approx([e.estimate for e in expected.fixed.effects])
    assert exclude_studies(model, ["S1", "S2", "S3", "S5"]).errors  # reference loses every study


def _fake_scenario_evaluator():
    """evaluate_scenario stand-in: cost = 100 * effect parameter A, effect = parameter B."""

    from types import SimpleNamespace

    def evaluate(document, overrides=None, **_):
        a = overrides.get("or_drug_a_vs_placebo", 1.0)
        b = overrides.get("or_drug_b_vs_placebo", 1.0)
        outcome = SimpleNamespace(strategy_id="s", strategy_name="s", total_cost=100 * a, total_effect=b,
                                  patient_cost_variance=None, patient_effect_variance=None,
                                  patient_cost_effect_covariance=None, n_patients=None)
        return SimpleNamespace(errors=[], outcomes=[outcome])

    return evaluate


def test_posterior_samples_are_exported_and_drawn_jointly_in_psa() -> None:
    from heor_tree_engine.nma_bayes import BayesSettings, run_nma_bayesian
    from heor_tree_engine.psa import run_psa

    model = resolve_nma(_network())
    result = run_nma_frequentist(model)
    bayes = run_nma_bayesian(model, BayesSettings(effects="random", chains=2, draws_per_chain=1500, seed=3))
    proxy = type(result)(reference=result.reference, measure=result.measure, treatments=list(bayes.treatments),
                         fixed=bayes.fit, random=bayes.fit)
    evidence = build_evidence_set(proxy, "random", treatments=["Drug A", "Drug B"], draws=bayes.draws, max_samples=1000)

    a, b = evidence.parameters
    assert len(a.samples) == 1000 and len(b.samples) == 1000
    assert np.median(a.samples) == pytest.approx(a.base, rel=0.03)
    assert np.log(b.samples).std() == pytest.approx(b.se, rel=0.1)

    document: dict = {"parameters": [], "psa_parameters": [], "psa_correlations": []}
    merge_into_document(document, evidence)
    assert set(document["psa_empirical"]) == {f"param:{a.name}", f"param:{b.name}"}
    assert document["psa_empirical"][f"param:{a.name}"]["joint"] == document["psa_empirical"][f"param:{b.name}"]["joint"]

    psa = run_psa(document, document["psa_parameters"], iterations=800, evaluate_scenario=_fake_scenario_evaluator(),
                  seed=1, correlations=document["psa_correlations"])
    assert psa.errors == []
    sampled_a, sampled_b = psa.samples[:, 0], psa.samples[:, 1]
    assert set(np.round(sampled_a, 12)) <= set(np.round(a.samples, 12))  # values come from the posterior draws
    # joint index: every (a, b) pair is one of the exported pairs
    pairs = {(round(x, 9), round(y, 9)) for x, y in zip(a.samples, b.samples)}
    assert all((round(x, 9), round(y, 9)) in pairs for x, y in zip(sampled_a, sampled_b))

    # editing the PSA row by hand switches back to the row's own distribution
    document["psa_parameters"][0]["values"]["mean"] = 0.9
    psa2 = run_psa(document, document["psa_parameters"], iterations=200, evaluate_scenario=_fake_scenario_evaluator(), seed=1)
    assert not set(np.round(psa2.samples[:, 0], 12)) <= set(np.round(a.samples, 12))
    assert set(np.round(psa2.samples[:, 1], 12)) <= set(np.round(b.samples, 12))
