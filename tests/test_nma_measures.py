"""Arm-level RR / RD (binary) and SMD (continuous) against closed forms."""

import math

import pytest

from heor_tree_engine.nma_freq import run_nma_frequentist
from heor_tree_engine.nma_parser import resolve_nma
from heor_tree_engine.nma_regression import baseline_risk_values
from heor_tree_engine.nma_validation import validate_nma


def _binary(measure, studies, zero_correction=0.5):
    return resolve_nma({"spec": {"outcome": "binary", "measure": measure, "reference": "P",
                                 "zero_correction": zero_correction},
                        "treatments": [{"name": "P"}, {"name": "A"}],
                        "studies": studies})


def _arms(study_id, *rows):
    return {"id": study_id, "arms": [{"treatment": t, "events": e, "n": n} for t, e, n in rows]}


def test_single_study_rr_and_rd_match_the_textbook_formulas() -> None:
    study = [_arms("s1", ("P", 20, 100), ("A", 10, 100))]
    rr = run_nma_frequentist(_binary("RR", study)).fixed.effects[0]
    assert rr.natural == pytest.approx(0.5)
    assert rr.se == pytest.approx(math.sqrt(1 / 10 - 1 / 100 + 1 / 20 - 1 / 100))

    rd = run_nma_frequentist(_binary("RD", study)).fixed.effects[0]
    assert rd.natural == pytest.approx(-0.10)
    assert rd.se == pytest.approx(math.sqrt(0.1 * 0.9 / 100 + 0.2 * 0.8 / 100))
    assert rd.natural_lower < -0.10 < rd.natural_upper  # identity scale, no exp


def test_zero_event_rules_differ_by_measure() -> None:
    studies = [_arms("s1", ("P", 20, 100), ("A", 10, 100)), _arms("s2", ("P", 0, 50), ("A", 0, 50))]
    rr = run_nma_frequentist(_binary("RR", studies))
    assert rr.excluded_studies == ["s2"]  # no information on a ratio
    rd = run_nma_frequentist(_binary("RD", studies))
    assert rd.excluded_studies == []  # 0 vs 0 is informative for the risk difference
    no_correction = run_nma_frequentist(_binary("RD", studies, zero_correction=0.0))
    assert no_correction.excluded_studies == ["s2"] and no_correction.errors == []


def test_arm_level_measure_validation() -> None:
    study = [_arms("s1", ("P", 20, 100), ("A", 10, 100))]
    for measure in ("OR", "RR", "RD"):
        assert validate_nma(_binary(measure, study)) == []
    assert [e.code for e in validate_nma(_binary("MD", study))] == ["unsupported_measure"]


def _continuous(measure, studies):
    return resolve_nma({"spec": {"outcome": "continuous", "measure": measure, "reference": "P"},
                        "treatments": [{"name": "P"}, {"name": "A"}, {"name": "B"}],
                        "studies": [{"id": sid, "arms": [{"treatment": t, "mean": m, "sd": sd, "n": n}
                                                         for t, m, sd, n in arms]} for sid, arms in studies]})


def test_two_arm_smd_is_hedges_g() -> None:
    model = _continuous("SMD", [("s1", [("P", 10.0, 4.0, 30), ("A", 8.0, 5.0, 40)]),
                                ("s2", [("P", 10.0, 4.0, 30), ("B", 9.0, 4.0, 30)])])
    effect = run_nma_frequentist(model).fixed.effects[0]

    df = 30 + 40 - 2
    pooled = math.sqrt((29 * 16 + 39 * 25) / df)
    g = (1 - 3 / (4 * df - 1)) * (8.0 - 10.0) / pooled
    assert effect.estimate == pytest.approx(g)
    assert effect.se == pytest.approx(math.sqrt(1 / 30 + 1 / 40 + g * g / (2 * 70)))


def test_smd_network_with_a_three_arm_study_runs_and_is_scale_free() -> None:
    studies = [("s1", [("P", 10.0, 4.0, 30), ("A", 8.0, 5.0, 40)]),
               ("s2", [("P", 12.0, 3.0, 50), ("B", 11.0, 3.5, 50)]),
               ("s3", [("P", 9.0, 4.0, 40), ("A", 7.5, 4.0, 40), ("B", 8.0, 4.5, 42)])]
    result = run_nma_frequentist(_continuous("SMD", studies))
    assert result.errors == [] and len(result.random.effects) == 2

    # SMD does not change when one study is measured on another scale (x10).
    rescaled = [(sid, [(t, m * 10, sd * 10, n) for t, m, sd, n in arms]) if sid == "s2" else (sid, arms)
                for sid, arms in studies]
    other = run_nma_frequentist(_continuous("SMD", rescaled))
    assert [e.estimate for e in other.random.effects] == pytest.approx([e.estimate for e in result.random.effects])
    assert baseline_risk_values(_continuous("SMD", studies)) == {}


def test_baseline_risk_follows_the_measure() -> None:
    study = [_arms("s1", ("P", 20, 100), ("A", 10, 100))]
    assert baseline_risk_values(_binary("RD", study))["s1"] == pytest.approx(0.2)
    assert baseline_risk_values(_binary("RR", study))["s1"] == pytest.approx(math.log(0.2))
    assert baseline_risk_values(_binary("OR", study))["s1"] == pytest.approx(math.log(0.25))
