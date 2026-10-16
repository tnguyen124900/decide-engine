import pytest

from heor_tree_engine import Engine
from heor_tree_engine.parser import resolve
from heor_tree_engine.tree_clone import tree_clone_errors


def _terminal(node_id: str, cost: object, effect: object) -> dict:
    return {"id": node_id, "type": "terminal", "cost": cost, "effect": effect, "year": 0.0, "branches": []}


def _doc() -> dict:
    """Two strategies reuse the same 'event' structure; the second swaps the
    event probability and cost parameters through substitutions."""
    return {
        "root_id": "d",
        "parameters": [
            {"name": "p_a", "base": 0.2}, {"name": "p_b", "base": 0.5},
            {"name": "c_a", "base": 100.0}, {"name": "c_b", "base": 300.0},
        ],
        "nodes": [
            {"id": "d", "type": "decision", "branches": [
                {"label": "A", "child_id": "ev_a"}, {"label": "B", "child_id": "ev_b"}]},
            {"id": "ev_a", "type": "chance", "label": "event A", "branches": [
                {"label": "event", "child_id": "t_ev_a", "probability": "p_a", "cost_increment": "c_a"},
                {"label": "none", "child_id": "t_no_a", "probability": "#"}]},
            _terminal("t_ev_a", 0.0, 1.0),
            _terminal("t_no_a", 0.0, 2.0),
            {"id": "ev_b", "type": "chance", "label": "event B", "clone_of": "ev_a",
             "clone_substitutions": {"p_a": "p_b", "c_a": "c_b"}, "branches": []},
        ],
    }


def _expected(p: float, c: float) -> tuple[float, float]:
    return p * c, p * 1.0 + (1 - p) * 2.0


def _run(doc: dict):
    root, params = resolve(doc)
    result = Engine().run(root, params)
    assert result.errors == []
    return result.strategies


def test_clone_runs_as_copy_of_master_with_substitutions() -> None:
    rows = _run(_doc())
    cost_a, eff_a = _expected(0.2, 100.0)
    cost_b, eff_b = _expected(0.5, 300.0)
    assert rows["A"].cost == pytest.approx(cost_a)
    assert rows["A"].effect == pytest.approx(eff_a)
    assert rows["B"].cost == pytest.approx(cost_b)
    assert rows["B"].effect == pytest.approx(eff_b)


def test_editing_master_updates_clone() -> None:
    doc = _doc()
    doc["nodes"][2]["effect"] = 10.0  # master's event terminal
    rows = _run(doc)
    assert rows["B"].effect == pytest.approx(0.5 * 10.0 + 0.5 * 2.0)


def test_clone_errors() -> None:
    doc = _doc()
    doc["nodes"][4]["clone_of"] = "missing"
    assert tree_clone_errors(doc)
    with pytest.raises(ValueError):
        resolve(doc)
    doc = _doc()
    doc["nodes"][4]["clone_of"] = "ev_b"
    assert tree_clone_errors(doc)
    doc = _doc()
    doc["nodes"].append({"id": "ev_c", "type": "chance", "clone_of": "ev_b", "branches": []})
    assert tree_clone_errors(doc)


def test_document_without_clones_is_untouched() -> None:
    from heor_tree_engine.tree_clone import expand_tree_clones
    doc = _doc()
    doc["nodes"].pop()
    doc["nodes"][0]["branches"].pop()
    assert expand_tree_clones(doc) is doc


def test_des_clone_still_ignores_non_des_clone_nodes() -> None:
    from heor_tree_engine.des_clone import expand_des_clones
    doc = _doc()
    assert expand_des_clones(doc) is doc


def test_add_clone_and_make_independent_copy() -> None:
    from heor_tree_engine.tree_clone import add_tree_clone, make_independent_copy, parse_substitutions
    doc = _doc()
    doc["nodes"].pop()
    doc["nodes"][0]["branches"].pop()
    assert add_tree_clone(doc, "d", "ev_a", "ev_b", parse_substitutions("p_a -> p_b\nc_a = c_b")) is None
    assert add_tree_clone(doc, "d", "ev_a", "ev_b")  # duplicate id
    assert add_tree_clone(doc, "ev_a", "ev_a", "x")  # inside own master
    doc["nodes"][0]["branches"][-1]["label"] = "B"
    doc["nodes"][0]["branches"][0]["label"] = "A"
    before = _run(doc)["B"].cost
    assert make_independent_copy(doc, "ev_b") is None
    assert not any(n.get("clone_of") for n in doc["nodes"])
    assert _run(doc)["B"].cost == pytest.approx(before)
    assert make_independent_copy(doc, "ev_b")  # no longer a clone


def test_rename_master_repoints_clones() -> None:
    from heor_tree_engine.tree_clone import rename_master_references

    doc = _doc()
    for n in doc["nodes"]:
        if n["id"] == "ev_a":
            n["id"] = "ev_main"
    for n in doc["nodes"]:
        for b in n.get("branches", []):
            if b.get("child_id") == "ev_a":
                b["child_id"] = "ev_main"
    assert rename_master_references(doc, "ev_a", "ev_main") == 1
    assert tree_clone_errors(doc) == []
    assert rename_master_references(doc, "ev_a", "ev_a") == 0
    rows = _run(doc)
    assert rows["B"].cost == pytest.approx(_expected(0.5, 300.0)[0])


def test_delete_master_detaches_clone_and_keeps_result() -> None:
    from heor_tree_engine.tree_clone import detach_clones_of

    doc = _doc()
    before = _run(doc)
    detached = detach_clones_of(doc, {"ev_a", "t_ev_a", "t_no_a"})
    assert detached == ["ev_b"]
    assert not any(n.get("clone_of") for n in doc["nodes"])
    # delete strategy A's whole subtree the way the app does, then run: B must keep its numbers
    doomed = {"ev_a", "t_ev_a", "t_no_a"}
    doc["nodes"] = [n for n in doc["nodes"] if n["id"] not in doomed]
    for n in doc["nodes"]:
        n["branches"] = [b for b in n.get("branches", []) if b.get("child_id") not in doomed]
    assert tree_clone_errors(doc) == []
    after = _run(doc)
    assert after["B"].cost == pytest.approx(before["B"].cost)
    assert after["B"].effect == pytest.approx(before["B"].effect)


def test_delete_clone_together_with_master_is_not_detached() -> None:
    from heor_tree_engine.tree_clone import detach_clones_of

    doc = _doc()
    assert detach_clones_of(doc, {"ev_a", "ev_b"}) == []
