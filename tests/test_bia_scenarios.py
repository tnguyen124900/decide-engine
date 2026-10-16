import pytest

from heor_tree_engine.bia import calculate_budget_impact
from heor_tree_engine.bia_scenarios import (
    apply_overrides, list_parameters, parse_years, preset_scenarios, resolve, run_scenarios, tornado,
)


def _doc():
    comps = [{"id": "new", "label": "New", "is_new": True, "cost_acquisition": 1000.0},
             {"id": "old", "label": "Old", "cost_acquisition": 400.0, "cost_administration": 50.0}]
    return {"time_horizon_years": 3, "discount_rate": 0.0, "comparators": comps, "eligible_population": [100.0, 100.0, 100.0],
            "current_mix_share": {"new": [0, 0, 0], "old": [1, 1, 1]},
            "new_mix_share": {"new": [0.1, 0.2, 0.3], "old": [0.9, 0.8, 0.7]}}


def test_parse_years_and_resolve():
    assert parse_years("", 5) == [0, 1, 2, 3, 4] and parse_years("2-4", 5) == [1, 2, 3] and parse_years("1, 5, 9", 5) == [0, 4]
    c, k = resolve(_doc(), "comparators/old/cost_acquisition")
    assert c[k] == 400.0
    assert resolve(_doc(), "comparators/zzz/cost_acquisition") == (None, None)


def test_set_multiply_add_scalar_and_series_with_years():
    doc = _doc()
    out, _ = apply_overrides(doc, [
        {"param": "comparators/old/cost_acquisition", "mode": "add", "value": 100},
        {"param": "comparators/new/cost_acquisition", "mode": "multiply", "value": 0.5},
        {"param": "new_mix_share/new", "mode": "set", "value": 0.5, "years": "3"},
        {"param": "eligible_population", "mode": "multiply", "value": 2, "years": "1-2"},
    ])
    assert out["comparators"][1]["cost_acquisition"] == 500.0 and out["comparators"][0]["cost_acquisition"] == 500.0
    assert out["new_mix_share"]["new"] == [0.1, 0.2, 0.5]
    assert out["eligible_population"] == [200.0, 200.0, 100.0]
    assert doc["eligible_population"] == [100.0] * 3  # original untouched


def test_scenarios_match_direct_calculation_and_shortcuts():
    doc = _doc()
    rows = {r["name"]: r for r in run_scenarios(doc, [
        {"name": "Cheaper", "overrides": [{"param": "@price", "mode": "multiply", "value": 0.5}]},
        {"name": "Same via path", "overrides": [{"param": "comparators/new/cost_acquisition", "mode": "multiply", "value": 0.5}]},
        {"name": "Bigger pop", "overrides": [{"param": "@population", "mode": "multiply", "value": 2}]},
        {"name": "Faster uptake", "overrides": [{"param": "@uptake", "mode": "multiply", "value": 2}]},
        {"name": "", "overrides": []},
    ])}
    base, _ = calculate_budget_impact(doc)
    assert rows["Base case"]["total"] == pytest.approx(base.total_budget_impact)
    assert rows["Cheaper"]["total"] == pytest.approx(rows["Same via path"]["total"])
    assert rows["Cheaper"]["total"] < rows["Base case"]["total"] < rows["Faster uptake"]["total"]
    assert rows["Bigger pop"]["total"] == pytest.approx(2 * rows["Base case"]["total"])
    assert len(rows) == 5  # base + 4 named (the unnamed one is skipped)


def test_tornado_sorted_by_swing_and_list_parameters():
    doc = _doc()
    params = [p for p, _ in list_parameters(doc)]
    assert "@price" in params and "comparators/new/cost_acquisition" in params and "new_mix_share/old" in params
    rows = tornado(doc, ["comparators/new/cost_acquisition", "comparators/old/cost_administration"], 20.0)
    assert rows[0]["param"] == "comparators/new/cost_acquisition" and rows[0]["swing"] >= rows[1]["swing"]
    assert rows[0]["low"] < rows[0]["base"] < rows[0]["high"]


def test_presets():
    names = [s["name"] for s in preset_scenarios()]
    assert len(names) == 3
