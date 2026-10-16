import math

import pytest

from heor_tree_engine.engine import Engine
from heor_tree_engine.parser import resolve
from heor_tree_engine.tree_montecarlo import (
    run_group_mc_document,
    tracker_problems,
)


def _term(i, cost=0.0, effect=0.0):
    return {"id": i, "type": "terminal", "cost": cost, "effect": effect, "branches": []}


def _doc():
    """S: event1 (0.4) then event2 whose probability grows with the number of events so far (tracker n).
    Expected cost = 0.4*(0.4*60 + 0.6*10) + 0.6*(0.1*50 + 0.9*0) = 15."""
    return {
        "root_id": "d", "parameters": [], "variables": [], "distributions": [], "tables": [],
        "trackers": [{"name": "n", "initial": 0}],
        "nodes": [
            {"id": "d", "type": "decision", "branches": [{"label": "S", "child_id": "c1"}]},
            {"id": "c1", "type": "chance", "branches": [
                {"label": "event", "child_id": "m1", "probability": "0.4"},
                {"label": "none", "child_id": "m0", "probability": "#"}]},
            {"id": "m1", "type": "chance", "tracker_updates": [{"tracker": "n", "expr": "n + 1"}], "branches": [
                {"label": "event2", "child_id": "t11", "probability": "0.1 + 0.3*n"},
                {"label": "none2", "child_id": "t10", "probability": "#"}]},
            {"id": "m0", "type": "chance", "branches": [
                {"label": "event2", "child_id": "t01", "probability": "0.1 + 0.3*n"},
                {"label": "none2", "child_id": "t00", "probability": "#"}]},
            _term("t11", "50 + 10*n"), _term("t10", "10*n"), _term("t01", "50 + 10*n"), _term("t00", "10*n"),
        ],
    }


def test_tracker_changes_probabilities_and_payoffs_in_expectation():
    result = run_group_mc_document(_doc(), n=40000, seed=1)
    cost = result.outcomes["S"]["cost"]
    se = cost.std(ddof=1) / math.sqrt(len(cost))
    assert abs(cost.mean() - 15.0) < 4 * se
    n_final = result.tracker_final["S"]["n"]
    assert abs(n_final.mean() - 0.4) < 4 * n_final.std(ddof=1) / math.sqrt(len(n_final))
    assert set(n_final) <= {0.0, 1.0}


def test_matches_expected_value_engine_without_trackers():
    doc = _doc()
    doc["trackers"] = []
    doc["nodes"][2].pop("tracker_updates")
    for node in doc["nodes"]:
        for b in node.get("branches", []):
            if isinstance(b.get("probability"), str) and "n" in b["probability"]:
                b["probability"] = "0.25" if b["label"] == "event2" else "#"
    for t in doc["nodes"][4:]:
        t["cost"] = 20.0 if t["id"] in ("t11", "t01") else 5.0
    tree, params = resolve(doc)
    ev = Engine().run(tree, params).strategies["S"].cost
    result = run_group_mc_document(doc, n=40000, seed=3)
    cost = result.outcomes["S"]["cost"]
    assert abs(cost.mean() - ev) < 4 * cost.std(ddof=1) / math.sqrt(len(cost))


def test_deterministic_tracker_updated_on_decision_and_terminal():
    doc = {
        "root_id": "d", "parameters": [], "variables": [], "distributions": [], "tables": [],
        "trackers": [{"name": "age", "initial": 30}],
        "nodes": [
            {"id": "d", "type": "decision", "tracker_updates": [{"tracker": "age", "expr": "age + 1"}],
             "branches": [{"label": "A", "child_id": "ta"}, {"label": "B", "child_id": "tb"}]},
            {**_term("ta", "age"), "tracker_updates": [{"tracker": "age", "expr": "age + 10"}]},
            _term("tb", "age"),
        ],
    }
    result = run_group_mc_document(doc, n=5, seed=0)
    assert list(result.outcomes["A"]["cost"]) == [41.0] * 5   # terminal update runs before its payoff
    assert list(result.outcomes["B"]["cost"]) == [31.0] * 5
    inc = result.incremental("A", "B")
    assert inc["cost"]["mean"] == pytest.approx(10.0) and inc["cost"]["se"] == pytest.approx(0.0)


def test_reproducible_and_common_random_numbers():
    a = run_group_mc_document(_doc(), n=500, seed=5)
    b = run_group_mc_document(_doc(), n=500, seed=5)
    c = run_group_mc_document(_doc(), n=500, seed=6)
    assert (a.outcomes["S"]["cost"] == b.outcomes["S"]["cost"]).all()
    assert not (a.outcomes["S"]["cost"] == c.outcomes["S"]["cost"]).all()
    # two identical strategies see the same random numbers -> identical outcomes
    doc2 = _doc()
    clone_nodes = [dict(n, id=n["id"] + "x", branches=[dict(b, child_id=b["child_id"] + "x") for b in n.get("branches", [])])
                   for n in doc2["nodes"][1:]]
    doc2["nodes"] = doc2["nodes"][:1] + doc2["nodes"][1:] + clone_nodes
    doc2["nodes"][0]["branches"].append({"label": "S_copy", "child_id": "c1x"})
    r = run_group_mc_document(doc2, n=300, seed=9)
    assert (r.outcomes["S"]["cost"] == r.outcomes["S_copy"]["cost"]).all()


def test_clone_copies_keep_tracker_updates():
    doc = _doc()
    doc["nodes"][0]["branches"].append({"label": "S2", "child_id": "c1_clone"})
    doc["nodes"].append({"id": "c1_clone", "type": "chance", "clone_of": "c1", "branches": []})
    r = run_group_mc_document(doc, n=300, seed=4)
    assert (r.outcomes["S"]["cost"] == r.outcomes["S2"]["cost"]).all()
    assert (r.tracker_final["S"]["n"] == r.tracker_final["S2"]["n"]).all()


def test_problems_are_reported():
    doc = _doc()
    doc["trackers"] = [{"name": "n", "initial": 0}, {"name": "n", "initial": 1}, {"name": "1bad"}, {"name": "_x"}]
    doc["parameters"] = [{"name": "n", "base": 1}]
    doc["nodes"][2]["tracker_updates"] = [{"tracker": "ghost", "expr": "1"}]
    text = " | ".join(tracker_problems(doc))
    assert "Duplicate tracker" in text and "1bad" in text and "reserved" in text
    assert "same name as a parameter" in text and "unknown tracker 'ghost'" in text
    with pytest.raises(ValueError):
        run_group_mc_document(doc, n=10)


def test_probabilities_must_sum_to_one():
    doc = _doc()
    doc["nodes"][1]["branches"][1]["probability"] = "0.9"
    with pytest.raises(ValueError, match="sum to"):
        run_group_mc_document(doc, n=5)
