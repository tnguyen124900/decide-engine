import math

import pytest

from heor_tree_engine.nma_freq import run_nma_frequentist, t_cdf, t_quantile
from heor_tree_engine.nma_inconsistency import rebase_block, run_inconsistency
from heor_tree_engine.nma_parser import resolve_nma


def _contrast_doc(rows, reference="A", measure="HR"):
    """rows: (study, treatment, versus, estimate, se)."""

    return {"spec": {"outcome": "contrast", "measure": measure, "reference": reference},
            "studies": [{"id": sid, "contrasts": [{"treatment": t, "versus": v, "estimate": e, "se": se}]}
                        for sid, t, v, e, se in rows]}


def test_triangle_node_split_matches_closed_form() -> None:
    # One study per edge of an A-B-C triangle: direct B-vs-A is its own
    # study, indirect is (C vs A) - (C vs B) with summed variances.
    y_ab, se_ab = -0.50, 0.20
    y_ac, se_ac = -0.10, 0.15
    y_bc, se_bc = 0.30, 0.25
    doc = _contrast_doc([("s1", "B", "A", y_ab, se_ab), ("s2", "C", "A", y_ac, se_ac), ("s3", "C", "B", y_bc, se_bc)])

    result = run_inconsistency(resolve_nma(doc))
    splits = {(s.treatment, s.versus): s for s in result.splits_fixed}

    ab = splits[("B", "A")]
    assert ab.direct == pytest.approx(y_ab, abs=1e-12)
    assert ab.direct_se == pytest.approx(se_ab, rel=1e-10)
    assert ab.indirect == pytest.approx(y_ac - y_bc, abs=1e-12)
    assert ab.indirect_se == pytest.approx(math.sqrt(se_ac ** 2 + se_bc ** 2), rel=1e-10)
    diff = y_ab - (y_ac - y_bc)
    diff_se = math.sqrt(se_ab ** 2 + se_ac ** 2 + se_bc ** 2)
    assert ab.difference == pytest.approx(diff, abs=1e-12)
    assert ab.difference_se == pytest.approx(diff_se, rel=1e-10)
    assert ab.z == pytest.approx(diff / diff_se, rel=1e-10)
    assert ab.natural_direct == pytest.approx(math.exp(y_ab))
    assert len(result.splits_fixed) == 3  # every edge of the loop is splittable

    # With one study per design, all of Q is between-design inconsistency and
    # equals z^2 of the (single independent) loop.
    dbt = result.design_by_treatment
    assert dbt.q_within == pytest.approx(0.0, abs=1e-12) and dbt.df_within == 0
    assert dbt.df_between == 1
    assert dbt.q_between == pytest.approx((diff / diff_se) ** 2, rel=1e-10)


def test_consistent_data_gives_zero_difference() -> None:
    d = {"A": 0.0, "B": -0.4, "C": -0.1, "D": 0.2}
    rows = [(f"s{i}", t, v, d[t] - d[v], 0.1 + 0.03 * i)
            for i, (t, v) in enumerate([("B", "A"), ("C", "A"), ("C", "B"), ("D", "C"), ("D", "A"), ("B", "A")])]
    result = run_inconsistency(resolve_nma(_contrast_doc(rows)))

    assert result.errors == []
    assert result.splits_fixed
    for split in result.splits_fixed:
        assert split.difference == pytest.approx(0.0, abs=1e-12)
        assert split.p_value == pytest.approx(1.0, abs=1e-9)
        assert split.direct == pytest.approx(split.network, abs=1e-12)
    assert result.design_by_treatment.q_between == pytest.approx(0.0, abs=1e-12)


def test_direct_estimate_equals_pairwise_meta_analysis_of_direct_studies() -> None:
    rows = [("s1", "B", "A", -0.5, 0.2), ("s2", "B", "A", -0.3, 0.25),
            ("s3", "C", "A", -0.1, 0.15), ("s4", "C", "B", 0.3, 0.25)]
    split = {(s.treatment, s.versus): s for s in run_inconsistency(resolve_nma(_contrast_doc(rows))).splits_fixed}[("B", "A")]

    w = [1 / 0.2 ** 2, 1 / 0.25 ** 2]
    pooled = (w[0] * -0.5 + w[1] * -0.3) / sum(w)
    assert split.direct == pytest.approx(pooled, rel=1e-10)
    assert split.direct_se == pytest.approx(math.sqrt(1 / sum(w)), rel=1e-10)
    assert split.n_direct_studies == 2


def test_star_network_has_nothing_to_split() -> None:
    rows = [("s1", "B", "A", -0.5, 0.2), ("s2", "C", "A", -0.1, 0.15), ("s3", "B", "A", -0.4, 0.3)]
    result = run_inconsistency(resolve_nma(_contrast_doc(rows)))

    assert result.splits_fixed == []
    assert sorted(result.not_splittable) == [("B", "A"), ("C", "A")]
    assert any("no closed loops" in warning for warning in result.warnings)
    assert result.design_by_treatment.df_between == 0
    assert result.design_by_treatment.p_between is None


def _multi_arm_network(order=("P", "A", "B")):
    arms3 = {"P": (20, 100), "A": (12, 101), "B": (15, 99)}
    return {
        "spec": {"outcome": "binary", "reference": "P"},
        "studies": [
            {"id": "S1", "arms": [{"treatment": "P", "events": 18, "n": 120}, {"treatment": "A", "events": 11, "n": 118}]},
            {"id": "S2", "arms": [{"treatment": "P", "events": 30, "n": 150}, {"treatment": "B", "events": 20, "n": 148}]},
            {"id": "S3", "arms": [{"treatment": "A", "events": 12, "n": 90}, {"treatment": "B", "events": 19, "n": 92}]},
            {"id": "S4", "arms": [{"treatment": t, "events": arms3[t][0], "n": arms3[t][1]} for t in order]},
            {"id": "S5", "arms": [{"treatment": "P", "events": 25, "n": 200}, {"treatment": "A", "events": 9, "n": 196}]},
        ],
    }


def test_multi_arm_split_does_not_depend_on_arm_order() -> None:
    first = run_inconsistency(resolve_nma(_multi_arm_network()))
    second = run_inconsistency(resolve_nma(_multi_arm_network(("B", "A", "P"))))

    assert first.errors == [] and second.errors == []
    for fit_a, fit_b in ((first.splits_fixed, second.splits_fixed), (first.splits_random, second.splits_random)):
        assert len(fit_a) == len(fit_b) > 0
        for a, b in zip(fit_a, fit_b):
            assert (a.treatment, a.versus) == (b.treatment, b.versus)
            assert a.direct == pytest.approx(b.direct, rel=1e-9, abs=1e-12)
            assert a.indirect == pytest.approx(b.indirect, rel=1e-9, abs=1e-12)
            assert a.p_value == pytest.approx(b.p_value, rel=1e-9, abs=1e-12)
    assert first.design_by_treatment.q_between == pytest.approx(second.design_by_treatment.q_between, rel=1e-9, abs=1e-12)


def test_q_decomposition_adds_up_and_matches_consistency_q() -> None:
    model = resolve_nma(_multi_arm_network())
    dbt = run_inconsistency(model).design_by_treatment
    het = run_nma_frequentist(model).heterogeneity

    assert dbt.q_total == pytest.approx(het.q, rel=1e-10)
    assert dbt.df_total == het.df
    assert dbt.q_within + dbt.q_between == pytest.approx(dbt.q_total, rel=1e-10)
    assert dbt.df_within + dbt.df_between == dbt.df_total
    # Designs: P-A (2 studies), P-B, A-B, P-A-B.
    designs = {design.design: design for design in dbt.designs}
    assert designs[("P", "A")].n_studies == 2 and designs[("P", "A")].df == 1
    assert sum(design.q for design in dbt.designs) == pytest.approx(dbt.q_within, rel=1e-10)


def test_rebase_block_keeps_contrasts_and_covariance_consistent() -> None:
    from heor_tree_engine.nma_freq import _StudyBlock

    block = _StudyBlock("s", "P", ["A", "B"], [-0.5, -0.2], [0.10 + 0.04, 0.12 + 0.04], 0.04)
    rebased = rebase_block(block, "A")

    assert rebased.baseline == "A" and rebased.treatments == ["P", "B"]
    assert rebased.estimates == pytest.approx([0.5, 0.3])
    assert rebased.variances == pytest.approx([0.04 + 0.10, 0.12 + 0.10])
    assert rebased.baseline_variance == pytest.approx(0.10)


def test_prediction_interval_uses_t_with_heterogeneity_df() -> None:
    rows = [(f"s{i}", "B", "A", e, se) for i, (e, se) in enumerate([(-0.9, 0.2), (-0.2, 0.15), (-0.5, 0.25), (0.1, 0.2), (-0.6, 0.3)])]
    result = run_nma_frequentist(resolve_nma(_contrast_doc(rows)))
    effect = result.random.effects[0]
    tau2 = result.heterogeneity.tau2

    assert tau2 > 0
    assert result.heterogeneity.prediction_df == result.heterogeneity.df  # contrasts - effects, as netmeta
    half = t_quantile(0.975, 4) * math.sqrt(effect.se ** 2 + tau2)  # 5 contrasts - 1 effect (netmeta's df.Q)
    assert effect.pred_lower == pytest.approx(effect.estimate - half, rel=1e-10)
    assert effect.natural_pred_upper == pytest.approx(math.exp(effect.estimate + half), rel=1e-10)
    assert result.fixed.effects[0].pred_lower is None


def test_t_distribution_helpers() -> None:
    assert t_quantile(0.975, 1) == pytest.approx(12.706204736, rel=1e-9)
    assert t_quantile(0.975, 5) == pytest.approx(2.570581836, rel=1e-9)
    assert t_quantile(0.975, 10) == pytest.approx(2.228138852, rel=1e-9)
    assert t_quantile(0.025, 10) == pytest.approx(-2.228138852, rel=1e-9)
    assert t_cdf(-2.0, 7) == pytest.approx(0.0428096643, rel=1e-8)


def test_sidde_equals_side_without_multi_arm_studies_and_back_calculates_indirect() -> None:
    from heor_tree_engine.nma_inconsistency import SPLIT_METHODS

    assert SPLIT_METHODS == ("side", "sidde")
    rows = [("s1", "B", "A", -0.5, 0.2), ("s2", "B", "A", -0.3, 0.2), ("s3", "C", "A", -0.2, 0.2),
            ("s4", "C", "B", 0.1, 0.25), ("s5", "C", "B", 0.4, 0.3)]
    model = resolve_nma(_contrast_doc(rows))
    side = run_inconsistency(model, method="side")
    sidde = run_inconsistency(model, method="sidde")
    assert sidde.method == "sidde" and len(sidde.splits_fixed) == len(side.splits_fixed) == 3
    for a, b in zip(side.splits_fixed, sidde.splits_fixed):
        assert (a.treatment, a.versus) == (b.treatment, b.versus)
        assert b.direct == pytest.approx(a.direct, rel=1e-9) and b.indirect == pytest.approx(a.indirect, rel=1e-9)
        assert b.p_value == pytest.approx(a.p_value, abs=1e-9)
        # back-calculation identity: network precision = direct + indirect precision
        assert 1 / b.network_se ** 2 == pytest.approx(1 / b.direct_se ** 2 + 1 / b.indirect_se ** 2, rel=1e-9)
    assert run_inconsistency(model, method="bogus").errors[0].code == "unknown_split_method"
