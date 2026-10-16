import math

import numpy as np
import pytest

from heor_tree_engine.nma_freq import chi2_sf, normal_cdf, run_nma_frequentist
from heor_tree_engine.nma_parser import resolve_nma

# BCG vaccine trials (Colditz et al. 1994; metafor's dat.bcg):
# (vaccinated TB+, vaccinated TB-, control TB+, control TB-)
BCG = [
    (4, 119, 11, 128), (6, 300, 29, 274), (3, 228, 11, 209), (62, 13536, 248, 12619),
    (33, 5036, 47, 5761), (180, 1361, 372, 1079), (8, 2537, 10, 619), (505, 87886, 499, 87892),
    (29, 7470, 45, 7232), (17, 1699, 65, 1600), (186, 50448, 141, 27197), (5, 2493, 3, 2338),
    (27, 16886, 29, 17825),
]


def _bcg_document() -> dict:
    studies = []
    for index, (a, b, c, d) in enumerate(BCG):
        log_rr = math.log(a / (a + b)) - math.log(c / (c + d))
        se = math.sqrt(1 / a - 1 / (a + b) + 1 / c - 1 / (c + d))
        studies.append({"id": f"T{index + 1}", "contrasts": [
            {"treatment": "BCG", "versus": "Control", "estimate": log_rr, "se": se}]})
    return {"spec": {"outcome": "contrast", "measure": "RR", "reference": "Control"}, "studies": studies}


def test_pairwise_matches_published_metafor_results_for_bcg() -> None:
    # Reference values: metafor rma(measure="RR", method="FE"/"DL") on dat.bcg.
    result = run_nma_frequentist(resolve_nma(_bcg_document()))

    assert result.errors == []
    fixed, random = result.fixed.effects[0], result.random.effects[0]
    assert fixed.estimate == pytest.approx(-0.4303, abs=5e-5)
    assert fixed.se == pytest.approx(0.0405, abs=5e-5)
    assert random.estimate == pytest.approx(-0.7141, abs=5e-5)
    assert random.se == pytest.approx(0.1787, abs=5e-5)
    assert result.heterogeneity.tau2 == pytest.approx(0.3088, abs=5e-5)
    assert result.heterogeneity.q == pytest.approx(152.2330, abs=5e-4)
    assert result.heterogeneity.df == 12
    assert result.heterogeneity.i2 == pytest.approx(0.9212, abs=5e-5)
    assert random.natural == pytest.approx(math.exp(-0.7141172), rel=1e-6)


def _binary_pairwise(data):
    studies = [{"id": f"S{i}", "arms": [
        {"treatment": "Placebo", "events": a, "n": b},
        {"treatment": "Drug A", "events": c, "n": d}]} for i, (a, b, c, d) in enumerate(data)]
    return {"spec": {"outcome": "binary", "reference": "Placebo"}, "studies": studies}


def test_binary_pairwise_equals_closed_form_dersimonian_laird() -> None:
    data = [(12, 100, 7, 98), (25, 200, 14, 196), (9, 80, 6, 82), (30, 150, 11, 148), (5, 60, 9, 61)]
    result = run_nma_frequentist(resolve_nma(_binary_pairwise(data)))

    y = np.array([math.log(c / (d - c)) - math.log(a / (b - a)) for a, b, c, d in data])
    v = np.array([1 / a + 1 / (b - a) + 1 / c + 1 / (d - c) for a, b, c, d in data])
    w = 1 / v
    fixed = (w * y).sum() / w.sum()
    q = (w * (y - fixed) ** 2).sum()
    tau2 = max(0.0, (q - (len(y) - 1)) / (w.sum() - (w ** 2).sum() / w.sum()))
    w_re = 1 / (v + tau2)
    random = (w_re * y).sum() / w_re.sum()

    assert result.fixed.effects[0].estimate == pytest.approx(fixed, rel=1e-10)
    assert result.heterogeneity.tau2 == pytest.approx(tau2, rel=1e-10)
    assert result.random.effects[0].estimate == pytest.approx(random, rel=1e-10)
    assert result.random.effects[0].se == pytest.approx(math.sqrt(1 / w_re.sum()), rel=1e-10)


def _network_document(reference: str = "Placebo", multi_arm_order=("Placebo", "Drug A", "Drug B")) -> dict:
    arms_multi = {
        "Placebo": {"events": 20, "n": 100},
        "Drug A": {"events": 12, "n": 101},
        "Drug B": {"events": 15, "n": 99},
    }
    return {
        "spec": {"outcome": "binary", "reference": reference},
        "studies": [
            {"id": "S1", "arms": [{"treatment": "Placebo", "events": 18, "n": 120}, {"treatment": "Drug A", "events": 11, "n": 118}]},
            {"id": "S2", "arms": [{"treatment": "Placebo", "events": 25, "n": 200}, {"treatment": "Drug A", "events": 14, "n": 196}]},
            {"id": "S3", "arms": [{"treatment": "Placebo", "events": 30, "n": 150}, {"treatment": "Drug B", "events": 20, "n": 148}]},
            {"id": "S4", "arms": [{"treatment": "Drug A", "events": 12, "n": 90}, {"treatment": "Drug B", "events": 15, "n": 92}]},
            {"id": "S5", "arms": [{"treatment": "Drug B", "events": 9, "n": 80}, {"treatment": "Drug C", "events": 14, "n": 79}]},
            {"id": "S6", "arms": [{"treatment": name, **arms_multi[name]} for name in multi_arm_order]},
            {"id": "S7", "arms": [{"treatment": "Placebo", "events": 40, "n": 210}, {"treatment": "Drug C", "events": 33, "n": 205}]},
        ],
    }


def test_multi_arm_result_does_not_depend_on_arm_order_or_baseline() -> None:
    first = run_nma_frequentist(resolve_nma(_network_document()))
    # Same data; the multi-arm study lists Drug B first and the reference is
    # changed, so a different baseline arm is used inside the 3-arm study.
    second = run_nma_frequentist(resolve_nma(_network_document("Drug B", ("Drug B", "Drug A", "Placebo"))))

    assert first.errors == [] and second.errors == []
    assert first.heterogeneity.q == pytest.approx(second.heterogeneity.q, rel=1e-10)
    assert first.heterogeneity.tau2 == pytest.approx(second.heterogeneity.tau2, rel=1e-10)
    for pair, effect in first.random.league.items():
        other = second.random.league[pair]
        assert effect.estimate == pytest.approx(other.estimate, rel=1e-9, abs=1e-12)
        assert effect.se == pytest.approx(other.se, rel=1e-9)


def test_league_table_is_antisymmetric_and_consistent() -> None:
    result = run_nma_frequentist(resolve_nma(_network_document()))
    league = result.fixed.league

    a_vs_p = league[("Drug A", "Placebo")].estimate
    b_vs_p = league[("Drug B", "Placebo")].estimate
    assert league[("Drug A", "Drug B")].estimate == pytest.approx(a_vs_p - b_vs_p, rel=1e-12)
    assert league[("Placebo", "Drug A")].estimate == pytest.approx(-a_vs_p, rel=1e-12)
    assert league[("Placebo", "Drug A")].se == pytest.approx(league[("Drug A", "Placebo")].se, rel=1e-12)
    assert len(league) == 4 * 3


def test_exactly_consistent_data_recovers_effects_with_zero_q() -> None:
    # Contrast-level data generated from d = (A: -0.5, B: -0.2, C: 0.1) vs P.
    d = {"Placebo": 0.0, "A": -0.5, "B": -0.2, "C": 0.1}
    pairs = [("A", "Placebo"), ("B", "Placebo"), ("B", "A"), ("C", "B"), ("C", "Placebo"), ("A", "Placebo")]
    studies = [{"id": f"S{i}", "contrasts": [{"treatment": t, "versus": v, "estimate": d[t] - d[v], "se": 0.1 + 0.05 * i}]}
               for i, (t, v) in enumerate(pairs)]
    result = run_nma_frequentist(resolve_nma({"spec": {"outcome": "contrast", "measure": "HR", "reference": "Placebo"}, "studies": studies}))

    assert result.heterogeneity.q == pytest.approx(0.0, abs=1e-20)
    assert result.heterogeneity.tau2 == 0.0
    estimates = {effect.treatment: effect.estimate for effect in result.random.effects}
    assert estimates == pytest.approx({"A": -0.5, "B": -0.2, "C": 0.1}, abs=1e-12)


def test_arm_level_and_equivalent_contrast_level_multi_arm_agree() -> None:
    arms = [("Placebo", 20, 100), ("A", 12, 101), ("B", 15, 99)]
    logit = {t: math.log(r / (n - r)) for t, r, n in arms}
    var = {t: 1 / r + 1 / (n - r) for t, r, n in arms}
    two_arm = [{"id": "S2", "arms": [{"treatment": "Placebo", "events": 30, "n": 150}, {"treatment": "B", "events": 20, "n": 148}]}]
    arm_doc = {"spec": {"outcome": "binary", "reference": "Placebo"},
               "studies": [{"id": "S1", "arms": [{"treatment": t, "events": r, "n": n} for t, r, n in arms]}, *two_arm]}
    two_arm_contrast = {"id": "S2", "contrasts": [{
        "treatment": "B", "versus": "Placebo",
        "estimate": math.log(20 / 128) - math.log(30 / 120),
        "se": math.sqrt(1 / 20 + 1 / 128 + 1 / 30 + 1 / 120)}]}
    contrast_doc = {"spec": {"outcome": "contrast", "measure": "OR", "reference": "Placebo"},
                    "studies": [{"id": "S1", "baseline_se": math.sqrt(var["Placebo"]), "contrasts": [
                        {"treatment": t, "versus": "Placebo", "estimate": logit[t] - logit["Placebo"],
                         "se": math.sqrt(var[t] + var["Placebo"])} for t in ("A", "B")]}, two_arm_contrast]}

    from_arms = run_nma_frequentist(resolve_nma(arm_doc))
    from_contrasts = run_nma_frequentist(resolve_nma(contrast_doc))

    assert from_arms.errors == [] and from_contrasts.errors == []
    for a, b in zip(from_arms.random.effects, from_contrasts.random.effects):
        assert a.estimate == pytest.approx(b.estimate, rel=1e-10)
        assert a.se == pytest.approx(b.se, rel=1e-10)


def test_zero_cell_study_is_corrected_and_double_zero_study_excluded() -> None:
    document = _binary_pairwise([(12, 100, 7, 98), (0, 50, 3, 52), (0, 40, 0, 41), (9, 80, 6, 82)])
    result = run_nma_frequentist(resolve_nma(document))

    assert result.errors == []
    assert result.excluded_studies == ["S2"]
    assert any("S1: zero cell" in warning for warning in result.warnings)
    assert len(result.observations) == 3


def test_p_scores_rank_best_treatment_highest_and_average_half() -> None:
    result = run_nma_frequentist(resolve_nma(_network_document()))
    scores = result.random.p_scores

    assert sum(scores.values()) / len(scores) == pytest.approx(0.5, abs=1e-12)
    effects = {effect.treatment: effect.estimate for effect in result.random.effects}
    best = min(effects, key=effects.get)  # small values good (fewer events)
    assert max(scores, key=scores.get) == best

    flipped = resolve_nma({**_network_document(), "spec": {"outcome": "binary", "reference": "Placebo", "small_values": "bad"}})
    flipped_scores = run_nma_frequentist(flipped).random.p_scores
    for name in scores:
        assert flipped_scores[name] == pytest.approx(1 - scores[name], abs=1e-12)


def test_continuous_and_rate_outcomes_use_md_and_log_rate_ratio() -> None:
    continuous = run_nma_frequentist(resolve_nma({
        "spec": {"outcome": "continuous", "reference": "P"},
        "studies": [{"id": "S1", "arms": [{"treatment": "P", "mean": 10.0, "sd": 2.0, "n": 50},
                                          {"treatment": "A", "mean": 8.5, "sd": 2.5, "n": 40}]}],
    }))
    effect = continuous.fixed.effects[0]
    assert continuous.measure == "MD"
    assert effect.estimate == pytest.approx(-1.5)
    assert effect.se == pytest.approx(math.sqrt(4 / 50 + 6.25 / 40))
    assert effect.natural == effect.estimate

    rate = run_nma_frequentist(resolve_nma({
        "spec": {"outcome": "rate", "reference": "P"},
        "studies": [{"id": "S1", "arms": [{"treatment": "P", "events": 30, "exposure": 1000.0},
                                          {"treatment": "A", "events": 15, "exposure": 800.0}]}],
    }))
    effect = rate.fixed.effects[0]
    assert rate.measure == "RATE_RATIO"
    assert effect.estimate == pytest.approx(math.log((15 / 800) / (30 / 1000)))
    assert effect.se == pytest.approx(math.sqrt(1 / 30 + 1 / 15))
    assert effect.natural == pytest.approx((15 / 800) / (30 / 1000))


def test_single_multi_arm_study_has_no_heterogeneity_df() -> None:
    result = run_nma_frequentist(resolve_nma({
        "spec": {"outcome": "binary", "reference": "P"},
        "studies": [{"id": "S1", "arms": [{"treatment": "P", "events": 20, "n": 100},
                                          {"treatment": "A", "events": 12, "n": 100},
                                          {"treatment": "B", "events": 15, "n": 100}]}],
    }))
    assert result.heterogeneity.df == 0
    assert result.heterogeneity.p_value is None
    assert result.random.effects[0].estimate == pytest.approx(result.fixed.effects[0].estimate)
    assert any("No degrees of freedom" in warning for warning in result.warnings)


def test_validation_errors_stop_the_run() -> None:
    result = run_nma_frequentist(resolve_nma({
        "spec": {"outcome": "binary", "reference": "P"},
        "studies": [{"id": "S1", "arms": [{"treatment": "P", "events": 20, "n": 100},
                                          {"treatment": "A", "events": 120, "n": 100}]}],
    }))
    assert result.fixed is None and result.random is None
    assert any(error.code == "events_exceed_n" for error in result.errors)


def test_confidence_interval_uses_requested_level() -> None:
    document = _bcg_document()
    document["spec"]["ci_level"] = 0.90
    effect = run_nma_frequentist(resolve_nma(document)).fixed.effects[0]
    assert effect.upper - effect.estimate == pytest.approx(1.6448536 * effect.se, rel=1e-6)


def test_numeric_helpers() -> None:
    assert normal_cdf(1.959963984540054) == pytest.approx(0.975, rel=1e-12)
    assert chi2_sf(3.841458820694124, 1) == pytest.approx(0.05, rel=1e-9)
    assert chi2_sf(18.307038053275146, 10) == pytest.approx(0.05, rel=1e-9)
    assert chi2_sf(0.0, 3) == 1.0
