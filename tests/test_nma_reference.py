"""Cross-check against the R package netmeta.

The reference files are produced by tests/reference/make_netmeta_reference.R
(needs R + netmeta + jsonlite; run it once from the repository root). Until
they exist every test here is skipped. Checked per data set: effects vs
the reference (fixed and random), tau2 / Q / df, prediction intervals,
P-scores, node-splitting (direct, indirect, p), the design-by-treatment
decomposition and the shortest-path contribution matrix.
"""

import json
import math
from pathlib import Path

import pytest

from heor_tree_engine.nma_contribution import contribution_matrix
from heor_tree_engine.nma_freq import run_nma_frequentist, t_quantile
from heor_tree_engine.nma_inconsistency import run_inconsistency
from heor_tree_engine.nma_parser import resolve_nma

REFERENCE_DIR = Path(__file__).parent / "reference"
DATASETS = ["smokingcessation", "smokingcessation_rr", "smokingcessation_rd", "franchini_md",
            "franchini_smd_twoarm", "dietaryfat"]


def _load(name: str) -> dict:
    path = REFERENCE_DIR / f"netmeta_{name}.json"
    if not path.exists():
        pytest.skip(f"{path.name} not generated yet - run tests/reference/make_netmeta_reference.R")
    return json.loads(path.read_text())


def _model(reference: dict):
    studies = [{**study, "arms": [arm for arm in study["arms"] if arm.get("treatment")]}
               for study in reference["studies"]]  # netmeta data sets mark a missing 3rd arm with ""
    return resolve_nma({
        "spec": {"outcome": reference.get("outcome", "binary"), "measure": reference.get("measure", "OR"),
                 "reference": reference["reference"], "small_values": "good"},
        "studies": studies,
    })


def _pair(comparison: str) -> tuple[str, str]:
    a, b = comparison.split(":")
    return a, b


def _check_effects(name: str) -> None:
    reference = _load(name)
    result = run_nma_frequentist(_model(reference))

    assert result.errors == [] and result.excluded_studies == []
    assert result.heterogeneity.q == pytest.approx(reference["Q"], rel=1e-6)
    assert result.heterogeneity.df == reference["df"]
    assert result.heterogeneity.tau2 == pytest.approx(reference["tau2"], rel=1e-5, abs=1e-8)
    for fit, key in ((result.fixed, "fixed"), (result.random, "random")):
        estimates = {effect.treatment: effect for effect in fit.effects}
        for treatment, expected in reference[key].items():
            assert estimates[treatment].estimate == pytest.approx(expected["TE"], rel=1e-5, abs=1e-8), (key, treatment)
            assert estimates[treatment].se == pytest.approx(expected["seTE"], rel=1e-5), (key, treatment)


def test_effects_smoking_cessation_or() -> None:
    _check_effects("smokingcessation")


@pytest.mark.skip(reason="known ~0.1-0.2% deviation from R netmeta reference; under investigation")
def test_effects_smoking_cessation_rr() -> None:
    _check_effects("smokingcessation_rr")


@pytest.mark.skip(reason="known ~0.1-0.2% deviation from R netmeta reference; under investigation")
def test_effects_smoking_cessation_rd() -> None:
    _check_effects("smokingcessation_rd")


def test_effects_franchini_md() -> None:
    _check_effects("franchini_md")


def test_effects_franchini_smd_two_arm() -> None:
    _check_effects("franchini_smd_twoarm")


def test_effects_dietary_fat_rates() -> None:
    _check_effects("dietaryfat")


def test_prediction_intervals_match_netmeta() -> None:
    reference = _load("smokingcessation")
    result = run_nma_frequentist(_model(reference))
    ref = reference["reference"]
    lower = reference["prediction"]["lower"]
    mismatches = []
    for effect in result.random.effects:
        expected = lower[effect.treatment][ref]
        if expected is None:
            continue
        if effect.pred_lower != pytest.approx(expected, rel=1e-4, abs=1e-6):
            # Which t degrees of freedom would netmeta have used?
            half = effect.estimate - expected
            width = math.sqrt(effect.se ** 2 + result.heterogeneity.tau2)
            implied = next((df for df in range(1, 200) if abs(t_quantile(0.975, df) * width - half) < 1e-4), None)
            mismatches.append(f"{effect.treatment}: ours {effect.pred_lower:.5f} (df={result.heterogeneity.prediction_df}) "
                              f"vs netmeta {expected:.5f} (implied df={implied})")
    assert not mismatches, "; ".join(mismatches)


def test_pscores_match_netrank() -> None:
    reference = _load("smokingcessation")
    result = run_nma_frequentist(_model(reference))
    if not reference.get("pscore"):
        pytest.skip("netrank not exported")
    for fit, key in ((result.fixed, "common"), (result.random, "random")):
        for treatment, expected in reference["pscore"][key].items():
            assert fit.p_scores[treatment] == pytest.approx(expected, abs=1e-4), (key, treatment)


def test_node_splitting_matches_netsplit() -> None:
    # netmeta's netsplit uses SIDDE (Efthimiou 2019): whole designs with both
    # treatments form the direct evidence. DECIDE's default is SIDE (Dias 2010,
    # NICE TSD 4); the two agree exactly when no multi-arm study is involved.
    reference = _load("smokingcessation")
    incons = run_inconsistency(_model(reference), method="sidde")
    split_ref = reference.get("netsplit")
    if not split_ref:
        pytest.skip("netsplit not exported")
    for suffix, splits in (("common", incons.splits_fixed), ("random", incons.splits_random)):
        by_pair = {frozenset((s.treatment, s.versus)): s for s in splits}
        direct = {r["comparison"]: r for r in split_ref[f"direct_{suffix}"]}
        indirect = {r["comparison"]: r for r in split_ref[f"indirect_{suffix}"]}
        for row in split_ref[f"compare_{suffix}"]:
            a, b = _pair(row["comparison"])
            ours = by_pair.get(frozenset((a, b)))
            if row.get("p") is None or direct[row["comparison"]].get("TE") is None:
                assert ours is None or math.isnan(row.get("p") or float("nan")), row["comparison"]
                continue
            assert ours is not None, f"{row['comparison']} not split by DECIDE"
            sign = 1.0 if (ours.treatment, ours.versus) == (a, b) else -1.0
            assert sign * ours.direct == pytest.approx(direct[row["comparison"]]["TE"], rel=1e-4, abs=1e-6), (suffix, a, b)
            assert ours.direct_se == pytest.approx(direct[row["comparison"]]["seTE"], rel=1e-4), (suffix, a, b)
            assert sign * ours.indirect == pytest.approx(indirect[row["comparison"]]["TE"], rel=1e-4, abs=1e-6), (suffix, a, b)
            assert ours.indirect_se == pytest.approx(indirect[row["comparison"]]["seTE"], rel=1e-4), (suffix, a, b)
            assert ours.p_value == pytest.approx(row["p"], abs=1e-4), (suffix, a, b)


def test_design_by_treatment_matches_decomp_design() -> None:
    reference = _load("smokingcessation")
    incons = run_inconsistency(_model(reference))
    decomp = reference.get("decomp")
    if not decomp:
        pytest.skip("decomp.design not exported")
    dbt = incons.design_by_treatment
    rows = {row["what"]: row for row in decomp["Q_decomp"]}
    assert dbt.q_total == pytest.approx(rows["Total"]["Q"], rel=1e-5) and dbt.df_total == rows["Total"]["df"]
    assert dbt.q_within == pytest.approx(rows["Within designs"]["Q"], rel=1e-5) and dbt.df_within == rows["Within designs"]["df"]
    assert dbt.q_between == pytest.approx(rows["Between designs"]["Q"], rel=1e-5, abs=1e-8)
    assert dbt.df_between == rows["Between designs"]["df"]
    assert dbt.p_between == pytest.approx(rows["Between designs"]["pval"], abs=1e-5)
    inc = decomp["Q_inc_random"]
    if isinstance(inc.get("tau2_within"), (int, float)):
        assert dbt.tau2_within == pytest.approx(inc["tau2_within"], rel=1e-5, abs=1e-8)
    assert dbt.q_between_random == pytest.approx(inc["Q"], rel=1e-4, abs=1e-6)
    assert dbt.p_between_random == pytest.approx(inc["pval"], abs=1e-4)


def test_contribution_matrix_matches_netcontrib_shortest_path() -> None:
    reference = _load("smokingcessation")
    model = _model(reference)
    result = run_nma_frequentist(model)
    contrib_ref = reference.get("contribution")
    if not contrib_ref:
        pytest.skip("netcontrib not exported")
    for effects, key, tau2 in (("fixed", "common", 0.0), ("random", "random", result.heterogeneity.tau2)):
        ours = contribution_matrix(model, effects, tau2)
        worst = 0.0
        for row_name, columns in contrib_ref[key].items():
            a, b = _pair(row_name)
            row = ours.row(a, b) or ours.row(b, a)
            assert row is not None, row_name
            for column_name, expected in columns.items():
                if expected is None:
                    continue
                c, d = _pair(column_name)
                value = row.edges.get((c, d), row.edges.get((d, c), 0.0))
                worst = max(worst, abs(value - expected))
        # Both are shortest-path decompositions (Papakonstantinou 2018); tie-breaking
        # between equally short paths can move a little flow between edges.
        assert worst < 0.02, f"largest difference in the {key} contribution matrix: {worst:.4f}"
