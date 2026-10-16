import pytest

from heor_tree_engine import find_threshold, run_grid, run_one_way, run_scenarios, evaluate_markov_scenario, run_tornado
from heor_tree_engine.sensitivity import _sample_range_by_interval


def _document() -> dict:
    return {
        "cycles": 1,
        "half_cycle_correction": False,
        "parameters": [{"name": "state_cost", "base": 100, "low": 50, "high": 200}],
        "variables": [{"name": "state_effect", "definition": 1, "low": 0.5, "high": 2}],
        "strategies": [
            {
                "id": "standard",
                "name": "Standard care",
                "states": [{"id": "well", "cost": "state_cost", "effect": "state_effect", "initial_prob": 1}],
                "transitions": [{"from_state": "well", "to_state": "well", "probability": 1}],
            },
            {
                "id": "new",
                "name": "New treatment",
                "states": [{"id": "well", "cost": 150, "effect": 1.2, "initial_prob": 1}],
                "transitions": [{"from_state": "well", "to_state": "well", "probability": 1}],
            },
        ],
    }


def test_scenario_runs_all_strategies_and_calculates_nmb() -> None:
    result = evaluate_markov_scenario(_document(), willingness_to_pay=1_000)

    assert result.errors == []
    assert result.reference_strategy_id == "standard"
    assert result.optimal_strategy_id == "new"
    standard, new = result.outcomes
    assert standard.nmb == 900.0
    assert new.nmb == 1050.0
    assert new.incremental_cost == 50.0
    assert new.incremental_effect == pytest.approx(0.2)
    assert new.icer == pytest.approx(250.0)


def test_scenario_overrides_parameter_and_variable_without_mutating_document() -> None:
    document = _document()

    result = evaluate_markov_scenario(
        document,
        overrides={"state_cost": 200, "state_effect": 0.5},
        willingness_to_pay=1_000,
    )

    assert result.errors == []
    assert result.outcomes[0].total_cost == 200.0
    assert result.outcomes[0].total_effect == 0.5
    assert document["parameters"][0]["base"] == 100
    assert document["variables"][0]["definition"] == 1


def test_scenario_reports_unknown_overrides_without_running() -> None:
    result = evaluate_markov_scenario(_document(), overrides={"unknown": 1})

    assert result.outcomes == []
    assert result.errors == ["Override 'unknown' does not match a parameter or variable"]


def test_tornado_requires_bounds_then_returns_ranked_low_high_results() -> None:
    document = _document()
    document["variables"][0].pop("high")

    entries, errors = run_tornado(document)

    assert entries == []
    assert errors == ["Enter low and high values for: state_effect"]

    document["variables"][0]["high"] = 2
    entries, errors = run_tornado(document, willingness_to_pay=1_000)

    assert errors == []
    assert [entry.input_name for entry in entries] == ["state_effect", "state_cost"]
    assert entries[0].low_outcome == 650.0
    assert entries[0].high_outcome == -850.0


def test_one_way_grid_threshold_and_scenarios_reuse_scenario_evaluator() -> None:
    document = _document()

    one_way, errors = run_one_way(document, "state_cost", points=3, willingness_to_pay=1_000)
    assert errors == []
    assert [row["value"] for row in one_way] == [50.0, 125.0, 200.0]

    grid, errors = run_grid(document, ["state_cost", "state_effect"], points=2, willingness_to_pay=1_000)
    assert errors == []
    assert len(grid) == 4

    threshold, errors = find_threshold(document, "state_effect", willingness_to_pay=1_000)
    assert errors == []
    assert threshold == pytest.approx(1.15, abs=0.01)

    scenarios, errors = run_scenarios(
        document,
        {"Lower cost": {"state_cost": 50}, "Higher cost": {"state_cost": 200}},
        willingness_to_pay=1_000,
    )
    assert errors == []
    assert [row["scenario"] for row in scenarios] == ["Lower cost", "Higher cost"]

def test_sample_range_by_interval_exact_multiple() -> None:
    assert _sample_range_by_interval(50, 200, 50) == [50.0, 100.0, 150.0, 200.0]


def test_sample_range_by_interval_non_exact_multiple_still_ends_at_high() -> None:
    # span=150, interval=60 -> 2 full steps (50, 110, 170), then a short final
    # step to land exactly on 200 instead of overshooting or stopping short.
    values = _sample_range_by_interval(50, 200, 60)
    assert values == pytest.approx([50.0, 110.0, 170.0, 200.0])


def test_sample_range_by_interval_larger_than_span_still_covers_both_ends() -> None:
    assert _sample_range_by_interval(50, 200, 300) == [50.0, 200.0]


def test_sample_range_by_interval_swaps_reversed_bounds() -> None:
    assert _sample_range_by_interval(200, 50, 50) == [50.0, 100.0, 150.0, 200.0]


def test_sample_range_by_interval_rejects_non_positive_interval() -> None:
    with pytest.raises(ValueError):
        _sample_range_by_interval(50, 200, 0)
    with pytest.raises(ValueError):
        _sample_range_by_interval(50, 200, -10)


def test_run_one_way_interval_overrides_points_and_ends_at_high() -> None:
    document = _document()

    # Regression: points-based default behaviour is unchanged when interval
    # is not given (already covered above, re-asserted here for locality).
    rows, errors = run_one_way(document, "state_cost", points=3, willingness_to_pay=1_000)
    assert errors == []
    assert [row["value"] for row in rows] == [50.0, 125.0, 200.0]

    # interval takes priority over points when both are given.
    rows, errors = run_one_way(
        document, "state_cost", points=3, interval=50, willingness_to_pay=1_000
    )
    assert errors == []
    assert [row["value"] for row in rows] == [50.0, 100.0, 150.0, 200.0]

    # interval=None (default) must behave exactly like before -- no interval kwarg at all.
    rows_default, errors = run_one_way(document, "state_cost", points=3, willingness_to_pay=1_000)
    assert errors == []
    assert rows_default == rows[:0] + [row for row in rows_default]  # sanity: still a list of rows
    assert [row["value"] for row in rows_default] == [50.0, 125.0, 200.0]


def test_run_grid_first_and_second_interval_independent() -> None:
    document = _document()

    grid, errors = run_grid(
        document,
        ["state_cost", "state_effect"],
        first_interval=50,
        second_interval=0.5,
        willingness_to_pay=1_000,
    )
    assert errors == []
    # state_cost: 50..200 step 50 -> 4 values; state_effect: 0.5..2 step 0.5 -> 4 values
    assert len({row["state_cost"] for row in grid}) == 4
    assert len({row["state_effect"] for row in grid}) == 4
    assert len(grid) == 16
    assert set(row["state_cost"] for row in grid) == {50.0, 100.0, 150.0, 200.0}
    assert sorted(set(row["state_effect"] for row in grid)) == pytest.approx([0.5, 1.0, 1.5, 2.0])

    # Regression: default points-based behaviour unchanged when no interval given.
    grid_default, errors = run_grid(document, ["state_cost", "state_effect"], points=2, willingness_to_pay=1_000)
    assert errors == []
    assert len(grid_default) == 4


def test_find_threshold_interval_passthrough() -> None:
    document = _document()

    # Regression: default points-based behaviour unchanged.
    threshold_default, errors = find_threshold(document, "state_effect", willingness_to_pay=1_000)
    assert errors == []
    assert threshold_default == pytest.approx(1.15, abs=0.01)

    # interval-based sweep should locate essentially the same threshold.
    threshold_interval, errors = find_threshold(
        document, "state_effect", interval=0.01, willingness_to_pay=1_000
    )
    assert errors == []
    assert threshold_interval == pytest.approx(1.15, abs=0.02)
