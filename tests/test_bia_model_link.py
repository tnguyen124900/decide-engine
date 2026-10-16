import copy
import pytest
from heor_tree_engine import bia, bia_model_link as ml


def _doc(basis="treatment_year"):
    return {
        "time_horizon_years": 3, "cost_basis": basis, "discount_rate": 0.0,
        "eligible_population": [100.0, 200.0, 0.0],
        "comparators": [
            {"id": "soc", "label": "SOC", "cost_acquisition": 1000.0},
            {"id": "new", "label": "New", "is_new": True, "cost_acquisition": 5000.0},
        ],
        "current_mix_share": {"soc": [1, 1, 1], "new": [0, 0, 0]},
        "new_mix_share": {"soc": [0.5, 0.5, 0.5], "new": [0.5, 0.5, 0.5]},
    }


def test_yearly_from_cycles():
    assert ml.yearly_from_cycles([1, 1, 1, 1, 1], 4) == [4.0, 1.0]
    assert ml.yearly_from_cycles([3, 2], 1) == [3.0, 2.0]


def test_cohort_mode_overlaps_cohorts():
    doc = _doc()
    ml.apply_link(doc["comparators"][1], "psm", "n", "New", [10.0, 6.0, 2.0])
    result, errors = bia.calculate_budget_impact(doc)
    assert not errors
    # year 1: cohort1 age0 -> 100*0.5*10 = 500 new; soc 100*.5*1000=50000 ; current soc 100*1000=100000
    assert result.years[0].new_mix_cost == pytest.approx(50000 + 500)
    # year 2: cohort1 age1 (6) + cohort2 age0 (10): new = 100*.5*6 + 200*.5*10 = 300+1000; soc flat 1000 each cohort
    assert result.years[1].new_mix_cost == pytest.approx(0.5 * 1000 * (100 + 200) + 1300)
    # year 3: no new patients; cohorts age 2 and 1: 100*.5*2 + 200*.5*6
    assert result.years[2].new_mix_cost == pytest.approx(0.5 * 1000 * 300 + 100 + 600)


def test_profile_ends_then_zero_and_annual_mode_ignores_profile():
    doc = _doc()
    ml.apply_link(doc["comparators"][1], "markov", "s", "S", [10.0])
    result, _ = bia.calculate_budget_impact(doc)
    assert result.years[2].new_mix_cost == pytest.approx(0.5 * 1000 * 300)  # linked comparator costs nothing after its profile
    annual = _doc("annual")
    ml.apply_link(annual["comparators"][1], "markov", "s", "S", [10.0])
    r2, _ = bia.calculate_budget_impact(annual)
    assert r2.years[0].new_mix_cost == pytest.approx(100 * 0.5 * 1000 + 100 * 0.5 * 5000)


def test_price_scenario_scales_profile_and_refresh():
    doc = _doc()
    ml.apply_link(doc["comparators"][1], "psm", "n", "New", [10.0, 6.0])
    from heor_tree_engine.bia_scenarios import scenario_document
    scaled = scenario_document(doc, price_multiplier=2.0)
    assert scaled["comparators"][1]["cost_by_treatment_year"] == [20.0, 12.0]
    assert doc["comparators"][1]["cost_by_treatment_year"] == [10.0, 6.0]
    problems = ml.refresh_links(doc, {"psm": lambda key: ([1.0, 1.0, 1.0], None)})
    assert not problems and doc["comparators"][1]["cost_by_treatment_year"] == [1.0, 1.0, 1.0]
    assert ml.refresh_links(doc, {})  # no provider -> message, profile untouched


def test_scenario_overrides_reach_profile():
    from heor_tree_engine.bia_scenarios import apply_overrides
    doc = _doc()
    ml.apply_link(doc["comparators"][1], "psm", "n", "New", [10.0, 6.0, 2.0])
    changed, _ = apply_overrides(doc, [{"param": "comparators/new/cost_by_treatment_year", "mode": "multiply", "value": 2, "years": "1"}])
    assert changed["comparators"][1]["cost_by_treatment_year"] == [20.0, 6.0, 2.0]
