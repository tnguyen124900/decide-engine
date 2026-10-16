"""DES as a node type of the decision tree:
Decision -> [Chance ->] DES -> Time -> Chance -> Terminal."""

import pytest

from heor_tree_engine import Engine, resolve
from heor_tree_engine.des_engine import run_des_document
from heor_tree_engine.models import DESTreeNode


def _des_nodes(suffix, rate, n=4000):
    return [
        {"id": f"des_{suffix}", "type": "des", "time_horizon": 100, "n_patients": n, "random_seed": 3,
         "branches": [{"label": "Alive", "child_id": f"alive_{suffix}", "probability": 1}]},
        {"id": f"alive_{suffix}", "type": "time", "name": "Alive", "cost_rate": 100, "effect_rate": 1,
         "branches": [{"label": "Death", "child_id": f"dead_{suffix}",
                       "time_to_event": {"distribution": "exponential", "params": {"rate": rate}}}]},
        {"id": f"dead_{suffix}", "type": "terminal", "cost": 1000},
    ]


def _document():
    return {
        "root_id": "root",
        "nodes": [
            {"id": "root", "type": "decision", "branches": [
                {"label": "Standard", "child_id": "des_std", "cost_increment": 0, "effect_increment": 0},
                {"label": "New drug", "child_id": "resp", "cost_increment": 5000, "effect_increment": 0}]},
            {"id": "resp", "type": "chance", "branches": [
                {"label": "Responds", "child_id": "des_resp", "probability": 0.4},
                {"label": "No response", "child_id": "des_nonresp", "probability": "#"}]},
            *_des_nodes("std", 0.2),
            *_des_nodes("resp", 0.1),
            *_des_nodes("nonresp", 0.2),
        ],
    }


def test_resolve_turns_des_subtree_into_one_tree_node():
    root, _ = resolve(_document())
    std = root.branches[0].child
    assert isinstance(std, DESTreeNode)
    assert set(std.model.nodes) == {"des_std", "alive_std", "dead_std"}


def test_des_node_rolls_back_like_any_node_including_after_a_chance_node():
    document = _document()
    root, parameters = resolve(document)
    result = Engine().run(root, parameters, cost_discount_rate=0.03, effect_discount_rate=0.03)
    assert not result.errors
    direct = run_des_document({"root_id": "des_std", "nodes": document["nodes"]}, 0.03, 0.03)
    (std_direct,) = direct.values()
    assert result.strategies["Standard"].cost == pytest.approx(std_direct.total_cost)
    assert result.strategies["Standard"].effect == pytest.approx(std_direct.total_effect)
    resp = run_des_document({"root_id": "des_resp", "nodes": document["nodes"]}, 0.03, 0.03)
    (resp_direct,) = resp.values()
    expected_effect = 0.4 * resp_direct.total_effect + 0.6 * std_direct.total_effect  # same seed/rate -> identical
    assert result.strategies["New drug"].effect == pytest.approx(expected_effect)
    assert result.strategies["New drug"].cost == pytest.approx(
        5000 + 0.4 * resp_direct.total_cost + 0.6 * std_direct.total_cost)


def test_per_patient_variables_are_allowed_in_a_tree_document_with_des():
    document = _document()
    document["variables"] = [{"name": "age", "definition": "60 + _time"}]
    document["nodes"][3]["cost_rate"] = "age"  # alive_std
    root, parameters = resolve(document)
    result = Engine().run(root, parameters)
    assert not result.errors
    assert result.strategies["Standard"].cost > 60 * 1


def test_des_validation_errors_surface_through_engine_run():
    document = _document()
    document["nodes"][3]["branches"][0]["time_to_event"] = {"distribution": "weibull", "params": {"shape": 2}}
    root, parameters = resolve(document)
    result = Engine().run(root, parameters)
    assert any(error.code == "missing_tte_parameter" for error in result.errors)


def test_des_runtime_error_is_reported_not_raised_by_engine_run():
    document = _document()
    document["nodes"][3]["cost_rate"] = "no_such_name * 2"
    root, parameters = resolve(document)
    result = Engine().run(root, parameters)
    assert result.errors and "no_such_name" in result.errors[0].message


def test_extra_outcomes_roll_back_through_the_tree():
    document = _document()
    for node in document["nodes"]:
        if node["type"] == "time":
            node["extra_rates"] = {"ly": 1}
    root, parameters = resolve(document)
    result = Engine().run(root, parameters, outcome_kinds={"ly": "effect"})
    assert not result.errors
    # undiscounted life-years == effect (effect_rate is 1 in these models)
    assert result.strategies["Standard"].extra_totals["ly"] == pytest.approx(result.strategies["Standard"].effect)


def _calibration_document(rate_value=0.1):
    return {
        "root_id": "root",
        "parameters": [{"name": "death_rate", "base": rate_value, "low": 0.05, "high": 0.5}],
        "distributions": [{"name": "death", "type": "Exponential", "parameters": {"rate": "death_rate"}}],
        "nodes": [
            {"id": "root", "type": "decision", "branches": [
                {"label": "Only", "child_id": "des", "cost_increment": 0, "effect_increment": 0}]},
            {"id": "des", "type": "des", "time_horizon": 20, "n_patients": 3000, "random_seed": 2,
             "branches": [{"label": "Alive", "child_id": "alive", "probability": 1}]},
            {"id": "alive", "type": "time", "name": "Alive", "branches": [
                {"label": "Death", "child_id": "dead",
                 "time_to_event": {"distribution": "named", "params": {"name": "death"}}}]},
            {"id": "dead", "type": "terminal", "name": "Dead"},
        ],
    }


def test_des_state_share_reads_the_trace_by_state_name():
    import math
    from heor_tree_engine.engine import des_state_share, des_strategy_runs

    document = _calibration_document(0.2)
    root, parameters = resolve(document)
    distributions = {d["name"]: d for d in document["distributions"]}
    runs = des_strategy_runs(root, "Only", parameters, distributions=distributions)
    assert des_state_share(runs, "Dead", 5) == pytest.approx(1 - math.exp(-1.0), abs=0.02)
    assert des_state_share(runs, "Alive", 5) == pytest.approx(math.exp(-1.0), abs=0.02)


def test_calibration_fits_a_des_rate_to_an_observed_state_share():
    """Observed: 63.2% dead at year 5 -> the exponential death rate is 0.2."""
    import math
    from heor_tree_engine.calibration import (
        CalibrationParameter, run_tree_calibration_search, tree_calibration_targets_from_rows)

    targets, errors = tree_calibration_targets_from_rows(
        [{"metric": "des_state_share", "target_id": "Dead", "time": 5, "value": 1 - math.exp(-1.0), "weight": 1}])
    assert not errors
    result = run_tree_calibration_search(
        _calibration_document(0.1), "Only", [CalibrationParameter("death_rate", 0.05, 0.5)], targets,
        method="grid", grid_points_per_dim=46, refine_with_local=False,
    )
    assert not result.errors, result.errors
    assert result.calibrated_values["death_rate"] == pytest.approx(0.2, abs=0.02)
