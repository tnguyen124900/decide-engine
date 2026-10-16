import copy
import json
from pathlib import Path

from heor_tree_engine.node_numbers import ensure_node_numbers, node_by_number, node_number_text

ROOT = Path(__file__).resolve().parent.parent


def _doc():
    return {
        "root_id": "root",
        "nodes": [
            {"id": "t2", "type": "terminal"},
            {"id": "root", "type": "decision", "branches": [{"label": "A", "child_id": "c1"}, {"label": "B", "child_id": "t2"}]},
            {"id": "c1", "type": "chance", "branches": [{"label": "x", "child_id": "t1"}, {"label": "y", "child_id": "t3"}]},
            {"id": "t1", "type": "terminal"},
            {"id": "t3", "type": "terminal"},
        ],
    }


def test_old_document_is_numbered_in_canvas_order() -> None:
    doc = _doc()
    assert ensure_node_numbers(doc) == 5
    by_id = {n["id"]: n["num"] for n in doc["nodes"]}
    assert by_id == {"root": 1, "c1": 2, "t1": 3, "t3": 4, "t2": 5}
    assert doc["next_node_num"] == 6


def test_idempotent_and_stable_under_rename() -> None:
    doc = _doc()
    ensure_node_numbers(doc)
    before = copy.deepcopy(doc)
    assert ensure_node_numbers(doc) == 0 and doc == before
    for n in doc["nodes"]:
        n["label"] = "renamed " + n["id"]
    ensure_node_numbers(doc)
    assert {n["id"]: n["num"] for n in doc["nodes"]} == {n["id"]: n["num"] for n in before["nodes"]}


def test_new_nodes_get_next_number_and_deleted_numbers_are_not_reused() -> None:
    doc = _doc()
    ensure_node_numbers(doc)
    doc["nodes"] = [n for n in doc["nodes"] if n["id"] != "t3"]  # delete number 4
    doc["nodes"][2]["branches"] = [b for b in doc["nodes"][2]["branches"] if b["child_id"] != "t3"]
    doc["nodes"].append({"id": "new1", "type": "terminal"})
    doc["nodes"].append({"id": "new2", "type": "terminal"})
    ensure_node_numbers(doc)
    nums = {n["id"]: n["num"] for n in doc["nodes"]}
    assert nums["new1"] == 6 and nums["new2"] == 7 and 4 not in nums.values()


def test_copied_node_with_duplicate_number_is_renumbered() -> None:
    doc = _doc()
    ensure_node_numbers(doc)
    clone = copy.deepcopy(next(n for n in doc["nodes"] if n["id"] == "t1"))
    clone["id"] = "t1_copy"
    doc["nodes"].append(clone)
    assert ensure_node_numbers(doc) == 1
    nums = {n["id"]: n["num"] for n in doc["nodes"]}
    assert nums["t1"] == 3 and nums["t1_copy"] == 6
    assert len(set(nums.values())) == len(nums)


def test_invalid_numbers_are_replaced_and_helpers() -> None:
    doc = _doc()
    doc["nodes"][0]["num"] = "x"
    doc["nodes"][1]["num"] = 0
    doc["nodes"][2]["num"] = True
    ensure_node_numbers(doc)
    assert all(isinstance(n["num"], int) and n["num"] >= 1 for n in doc["nodes"])
    assert len({n["num"] for n in doc["nodes"]}) == 5
    root = node_by_number(doc, doc["nodes"][1]["num"])
    assert root["id"] == "root" and node_number_text(root) == f"#{root['num']}"
    assert node_number_text({"id": "z"}) == "" and node_by_number(doc, 999) is None
    assert ensure_node_numbers({"nodes": []}) == 0




def test_numbers_do_not_affect_the_calculation() -> None:
    from heor_tree_engine import evaluate_decision, resolve

    document = json.loads((ROOT / "tests" / "data" / "x.json").read_text()) if False else {
        "root_id": "d",
        "parameters": [{"name": "p", "base": 0.3}],
        "nodes": [
            {"id": "d", "type": "decision", "branches": [{"label": "S", "child_id": "c"}]},
            {"id": "c", "type": "chance", "branches": [
                {"label": "a", "child_id": "t1", "probability": "p"}, {"label": "b", "child_id": "t2", "probability": "#"}]},
            {"id": "t1", "type": "terminal", "cost": 100, "effect": 1},
            {"id": "t2", "type": "terminal", "cost": 10, "effect": 2},
        ],
    }
    root, params = resolve(copy.deepcopy(document))
    plain = evaluate_decision(root, params)
    numbered = copy.deepcopy(document)
    ensure_node_numbers(numbered)
    assert all("num" in n for n in numbered["nodes"]) and numbered["next_node_num"] == 5
    root2, params2 = resolve(numbered)
    assert repr(evaluate_decision(root2, params2)) == repr(plain)


def test_state_numbers_stable_and_never_reused():
    from heor_tree_engine.node_numbers import ensure_state_numbers, state_number_label

    strategy = {"states": [{"id": "a"}, {"id": "b"}, {"id": "c"}]}
    assert state_number_label(strategy, strategy["states"][1]) == "#2"
    strategy["states"][1]["id"] = "renamed"
    strategy["states"].pop(0)
    strategy["states"].append({"id": "d"})
    ensure_state_numbers(strategy)
    assert [s["num"] for s in strategy["states"]] == [2, 3, 4]
