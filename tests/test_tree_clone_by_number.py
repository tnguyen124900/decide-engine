import copy

from heor_tree_engine.tree_clone import add_tree_clone, expand_tree_clones, sync_clone_masters, tree_clone_errors


def _doc():
    return {
        "root_id": "root",
        "nodes": [
            {"id": "root", "type": "decision", "label": "Root", "branches": [
                {"label": "A", "child_id": "ma"}, {"label": "B", "child_id": "x"}]},
            {"id": "ma", "type": "chance", "label": "Test", "branches": [
                {"label": "pos", "probability": "p_a", "child_id": "t1"}, {"label": "neg", "probability": "1-p_a", "child_id": "t2"}]},
            {"id": "t1", "type": "terminal", "label": "t1", "payoff": "0"},
            {"id": "t2", "type": "terminal", "label": "t2", "payoff": "0"},
            {"id": "x", "type": "chance", "label": "Test", "branches": []},
        ],
    }


def test_clone_follows_master_after_rename_by_any_route():
    doc = _doc()
    assert add_tree_clone(doc, "x", "ma", "cl") is None
    clone = next(n for n in doc["nodes"] if n["id"] == "cl")
    master = next(n for n in doc["nodes"] if n["id"] == "ma")
    assert clone["clone_of_num"] == master["num"]
    master["id"] = "renamed"  # the id changed without anyone fixing clone_of
    for n in doc["nodes"]:
        for b in n.get("branches", []):
            if b["child_id"] == "ma":
                b["child_id"] = "renamed"
    assert tree_clone_errors(doc)  # name-only lookup would be dangling
    assert sync_clone_masters(doc) == 1
    assert clone["clone_of"] == "renamed" and not tree_clone_errors(doc)
    out = expand_tree_clones(copy.deepcopy(doc))
    assert {"cl__t1" if False else "t1__cl", "t2__cl"} <= {n["id"] for n in out["nodes"]}


def test_old_file_without_number_gets_linked():
    doc = _doc()
    doc["nodes"].append({"id": "c2", "type": "chance", "clone_of": "ma", "branches": []})
    sync_clone_masters(doc)
    c2 = doc["nodes"][-1]
    assert c2["clone_of_num"] == doc["nodes"][1]["num"]
