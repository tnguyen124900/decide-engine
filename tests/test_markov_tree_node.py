import pytest

from heor_tree_engine import Engine, run_markov
from heor_tree_engine.markov_parser import resolve_markov
from heor_tree_engine.parser import resolve


def _markov_raw(cost: object = 100.0) -> dict:
    return {
        "id": "mk", "type": "markov", "label": "Markov", "cycles": 4, "cycles_per_year": 12,
        "half_cycle_correction": False, "branches": [],
        "states": [
            {"id": "well", "name": "Well", "cost": cost, "effect": "u_well", "initial_prob": 1.0},
            {"id": "dead", "name": "Dead", "cost": 0.0, "effect": 0.0, "initial_prob": 0.0, "is_absorbing": True},
        ],
        "transitions": [
            {"from_state": "well", "to_state": "dead", "probability": "p_die"},
            {"from_state": "well", "to_state": "well", "probability": "#"},
        ],
    }


def _doc(cost: object = 100.0, variables: list | None = None) -> dict:
    return {
        "root_id": "d",
        "parameters": [{"name": "p_die", "base": 0.2}, {"name": "u_well", "base": 0.8}],
        "variables": variables or [],
        "nodes": [
            {"id": "d", "type": "decision", "branches": [
                {"label": "Treat", "child_id": "c"}, {"label": "Nothing", "child_id": "t"}]},
            {"id": "c", "type": "chance", "branches": [
                {"label": "ok", "child_id": "mk", "probability": 0.6, "cost_increment": 50.0},
                {"label": "fail", "child_id": "t", "probability": "#"}]},
            _markov_raw(cost),
            {"id": "t", "type": "terminal", "cost": 0.0, "effect": 0.0, "year": 0.0, "branches": []},
        ],
    }


def _direct(doc: dict, discount_c: float = 0.0, discount_e: float = 0.0):
    root, params = resolve(doc)
    mk_raw = next(n for n in doc["nodes"] if n["id"] == "mk")
    model = resolve_markov(mk_raw)
    variables = {v["name"]: v["definition"] for v in doc["variables"]}
    base = {k: v for k, v in params.items() if k not in variables}
    return run_markov(model, base, discount_c, discount_e, variables=variables)


def test_markov_node_value_matches_direct_markov_run() -> None:
    doc = _doc()
    root, params = resolve(doc)
    result = Engine().run(root, params)
    assert result.errors == []
    ref = _direct(doc)
    assert result.strategies["Treat"].cost == pytest.approx(0.6 * (50.0 + ref.total_cost))
    assert result.strategies["Treat"].effect == pytest.approx(0.6 * ref.total_effect)
    assert result.strategies["Nothing"].cost == 0.0


def test_tree_annual_discount_becomes_per_cycle_rate() -> None:
    doc = _doc()
    root, params = resolve(doc)
    result = Engine().run(root, params, cost_discount_rate=0.03, effect_discount_rate=0.05)
    per_c = 1.03 ** (1 / 12) - 1
    per_e = 1.05 ** (1 / 12) - 1
    ref = _direct(doc, per_c, per_e)
    assert result.strategies["Treat"].cost == pytest.approx(0.6 * (50.0 + ref.total_cost))
    assert result.strategies["Treat"].effect == pytest.approx(0.6 * ref.total_effect)


def test_variable_using_cycle_is_evaluated_per_cycle_inside_markov_node() -> None:
    doc = _doc(cost="cDrug", variables=[{"name": "cDrug", "definition": "if(_cycle<2, 200, 10)"}])
    root, params = resolve(doc)
    result = Engine().run(root, params)
    assert result.errors == []
    ref = _direct(doc)
    assert result.strategies["Treat"].cost == pytest.approx(0.6 * (50.0 + ref.total_cost))
    assert ref.total_cost > 0


def test_changed_parameter_flows_into_markov_node() -> None:
    doc = _doc()
    doc["parameters"][0]["base"] = 0.5
    root, params = resolve(doc)
    result = Engine().run(root, params)
    ref = _direct(doc)
    assert result.strategies["Treat"].cost == pytest.approx(0.6 * (50.0 + ref.total_cost))


def test_bad_markov_node_reports_error_not_crash() -> None:
    doc = _doc()
    doc["nodes"][2]["transitions"][0]["probability"] = "missing_param"
    root, params = resolve(doc)
    result = Engine().run(root, params)
    assert result.errors


def test_table_helpers_round_trip_and_keep_hidden_state_fields() -> None:
    from heor_tree_engine.tree_markov_node import (
        rows_to_states, rows_to_transitions, states_to_rows, table_problems, transitions_to_rows,
    )
    raw = _markov_raw()
    raw["states"][0]["tenure_applied"] = True
    raw["states"][0]["tenure_count"] = 3
    rows = states_to_rows(raw["states"])
    rows[0]["cost"] = "cDrug+5"
    rows[1]["initial_prob"] = "0.0"
    states = rows_to_states(rows, raw["states"])
    assert states[0]["tenure_applied"] is True and states[0]["tenure_count"] == 3
    assert states[0]["cost"] == "cDrug+5"
    assert states[1]["initial_prob"] == 0.0
    transitions = rows_to_transitions(transitions_to_rows(raw["transitions"]), raw["transitions"])
    assert transitions[0]["probability"] == "p_die" and transitions[1]["probability"] == "#"
    assert table_problems(states, transitions) == []
    transitions.append({"from_state": "well", "to_state": "ghost", "probability": 0.1})
    assert table_problems(states, transitions)
    assert rows_to_states([{"id": "", "name": "x"}], []) == []


def test_transition_table_keeps_hidden_fields_and_import_merges_inputs() -> None:
    import json
    from heor_tree_engine.tree_markov_node import (
        import_markov_strategy, rows_to_transitions, strategies_of, transitions_to_rows,
    )
    existing = [{"from_state": "a", "to_state": "b", "probability": 0.1, "label": "AE", "extra_increment": {"ly": 1.0}},
                {"from_state": "a", "to_state": "b", "probability": 0.2, "label": "second"}]
    rows = transitions_to_rows(existing)
    rows[1]["probability"] = "0.3"
    out = rows_to_transitions(rows, existing)
    assert out[0]["label"] == "AE" and out[0]["extra_increment"] == {"ly": 1.0}
    assert out[1]["label"] == "second" and out[1]["probability"] == 0.3

    markov_doc = {
        "cycles": 20, "cycles_per_year": 12.0, "half_cycle_correction": False,
        "parameters": [{"name": "p_die", "base": 0.9}, {"name": "u_extra", "base": 0.7}],
        "variables": [{"name": "cDrug", "definition": "if(_cycle<2, 200, 10)"}],
        "distributions": [], "tables": [],
        "strategies": [{"id": "s1", "name": "SOC", "states": _markov_raw()["states"],
                        "transitions": _markov_raw()["transitions"]}],
    }
    assert strategies_of(markov_doc) == [("s1", "SOC")]
    doc = _doc()
    node = doc["nodes"][2]
    error, report = import_markov_strategy(node, doc, markov_doc, "s1")
    assert error is None
    assert node["cycles"] == 20 and node["cycles_per_year"] == 12.0 and node["half_cycle_correction"] is False
    assert "parameter u_extra" in report["added"] and "variable cDrug" in report["added"]
    assert report["conflicts"] == ["parameter p_die"]  # tree keeps its own p_die = 0.2
    assert next(p for p in doc["parameters"] if p["name"] == "p_die")["base"] == 0.2
    root, params = resolve(doc)
    assert Engine().run(root, params).errors == []
    assert import_markov_strategy(node, doc, markov_doc, "nope")[0]
    json.dumps(doc)
