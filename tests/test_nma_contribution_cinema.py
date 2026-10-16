import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # app-level modules

import nma_editor_support as sup  # noqa: E402
from heor_tree_engine.nma_cinema import (CONFIDENCE, DOMAINS, cinema_report_rows,  # noqa: E402
                                         normalize_level, run_cinema)
from heor_tree_engine.nma_contribution import contribution_matrix  # noqa: E402
from heor_tree_engine.nma_freq import run_nma_frequentist  # noqa: E402
from heor_tree_engine.nma_inconsistency import run_inconsistency  # noqa: E402
from heor_tree_engine.nma_parser import resolve_nma  # noqa: E402


def _triangle(se_bc: float = 1.0) -> dict:
    return {"spec": {"outcome": "contrast", "measure": "MD", "reference": "A"},
            "treatments": [{"name": name} for name in "ABC"],
            "studies": [
                {"id": "s1", "contrasts": [{"treatment": "B", "versus": "A", "estimate": 1.0, "se": 1.0}]},
                {"id": "s2", "contrasts": [{"treatment": "C", "versus": "A", "estimate": 1.0, "se": 1.0}]},
                {"id": "s3", "contrasts": [{"treatment": "C", "versus": "B", "estimate": 0.0, "se": se_bc}]},
            ]}


def test_triangle_contributions_match_the_closed_form() -> None:
    # Equal variances, fixed effect: the direct study has hat weight 2/3 and the
    # two-step indirect path 1/3, split over its two edges (1/6 each).
    result = contribution_matrix(resolve_nma(_triangle()), "fixed")

    row = result.row("B", "A")
    assert row.edges[("B", "A")] == pytest.approx(2 / 3)
    assert row.edges[("C", "A")] == pytest.approx(1 / 6)
    assert row.edges[("C", "B")] == pytest.approx(1 / 6)
    assert row.studies == pytest.approx({"s1": 2 / 3, "s2": 1 / 6, "s3": 1 / 6})
    assert row.direct_share == pytest.approx(2 / 3)


def test_rows_sum_to_one_and_multi_arm_is_baseline_invariant() -> None:
    document = sup.default_document()
    result = contribution_matrix(resolve_nma(document), "random")

    assert result.errors == []
    assert len(result.rows) == 10  # 5 treatments -> 10 pairs
    for row in result.rows:
        assert sum(row.edges.values()) == pytest.approx(1.0)
        assert sum(row.studies.values()) == pytest.approx(1.0)
        assert row.unexplained < 1e-9
    # Drug D vs Drug A has no direct evidence.
    assert result.row("Drug D", "Drug A").direct_share == 0.0

    # A different reference makes Drug C the baseline of the 3-arm study
    # (instead of Drug A); contributions must not change.
    shuffled = sup.default_document()
    shuffled["spec"]["reference"] = "Drug C"
    other = contribution_matrix(resolve_nma(shuffled), "random")
    for row in result.rows:
        twin = other.row(row.treatment, row.versus)
        assert twin.edges == pytest.approx(row.edges)


def test_contribution_frame_is_percent() -> None:
    result = contribution_matrix(resolve_nma(_triangle()), "fixed")
    frame = sup.contribution_frame(result)

    assert list(frame.columns) == ["Network estimate", "B vs A", "C vs A", "C vs B"]
    assert frame.iloc[0, 1:].sum() == pytest.approx(100.0, abs=0.2)
    assert "outline:2px solid #ea580c" in sup.contribution_html_table(result)


def _cinema(document, **kwargs):
    model = resolve_nma(document)
    nma = run_nma_frequentist(model)
    return run_cinema(model, nma, inconsistency=run_inconsistency(model), **kwargs)


def test_cinema_rates_every_comparison_on_six_domains() -> None:
    document = sup.default_document()
    ids = [study["id"] for study in document["studies"]]
    characteristics = [{"study": study_id, "rob": "Low", "indirectness": "Low"} for study_id in ids]
    result = _cinema(document, characteristics=characteristics)

    assert result.errors == []
    assert len(result.rows) == 10
    for row in result.rows:
        assert set(row.domains) == set(DOMAINS)
        assert row.confidence in CONFIDENCE
        assert row.domains["within_study_bias"].level == "No concerns"
        assert row.domains["indirectness"].level == "No concerns"
    # Wide CIs (e.g. Drug D vs Drug C, 0.30-1.52) cross both equivalence
    # boundaries 0.8 and 1.25 -> major imprecision.
    assert result.rows[-1].domains["imprecision"].level == "Major concerns"
    report = cinema_report_rows(result)
    assert report[0]["Comparison"] == "Drug A vs Placebo" and "Confidence" in report[0]


def test_cinema_risk_of_bias_is_contribution_weighted_and_missing_counts_as_some() -> None:
    document = sup.default_document()
    ids = [study["id"] for study in document["studies"]]
    all_high = [{"study": study_id, "rob": "High"} for study_id in ids]
    assert all(row.domains["within_study_bias"].level == "Major concerns"
               for row in _cinema(document, characteristics=all_high).rows)

    none_entered = _cinema(document, characteristics=[])
    assert all(row.domains["within_study_bias"].level == "Some concerns" for row in none_entered.rows)
    assert none_entered.warnings  # tells the user to enter RoB

    majority = _cinema(document, characteristics=[{"study": "Study 01", "rob": "High"},
                                                  {"study": "Study 02", "rob": "High"},
                                                  {"study": "Study 03", "rob": "High"}], rob_rule="majority")
    assert majority.rows[0].domains["within_study_bias"].level == "Major concerns"  # A vs P rests on 01-03


def test_cinema_margin_and_overrides() -> None:
    document = sup.default_document()
    wide = _cinema(document, margin=4.0)  # every CI within 0.25-4 -> no imprecision concerns
    assert all(row.domains["imprecision"].level == "No concerns" for row in wide.rows)

    result = _cinema(document, overrides={"Drug A|Placebo": {"imprecision": "Major concerns",
                                                             "confidence": "Moderate", "note": "clinical input"}})
    row = result.rows[0]
    assert row.domains["imprecision"].overridden and row.domains["imprecision"].level == "Major concerns"
    assert row.auto_confidence == "Very low" and row.confidence == "Moderate"  # Some RoB (not entered) + Major imprecision
    assert row.confidence_overridden and row.note == "clinical input"
    html = sup.cinema_html_table(result, {}, {}, {})
    assert "Moderate *" in html and sup.CONFIDENCE_COLORS["Moderate"] in html


def test_cinema_incoherence_flags_the_conflicting_loop() -> None:
    document = sup.default_document()
    document["studies"][6]["arms"] = [{"treatment": "Drug A", "events": 40, "n": 90},
                                      {"treatment": "Drug B", "events": 5, "n": 92},
                                      {"treatment": "Drug C", "events": 17, "n": 91}]
    result = _cinema(document, effects="fixed")
    levels = {row.label: row.domains["incoherence"].level for row in result.rows}

    assert any(level != "No concerns" for level in levels.values())


def test_normalize_level() -> None:
    assert normalize_level("high risk") == "High"
    assert normalize_level("Some concerns") == "Some concerns"
    assert normalize_level("unclear") == "Some concerns"
    assert normalize_level("L") == "Low"
    assert normalize_level(None) == ""
