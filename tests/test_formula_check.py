from heor_tree_engine.formula_check import check_document_formulas
from heor_tree_engine.bayes_revision import add_matrix_bayes_structure


def _doc():
    return {
        "root_id": "s",
        "parameters": [{"name": "p_death", "base": 0.1}, {"name": "cost_a", "base": 100.0}],
        "variables": [{"name": "v1", "definition": "p_death * 2"}],
        "distributions": [{"name": "w", "type": "weibull"}],
        "tables": [],
        "outcomes": [{"id": "cost", "type": "cost"}],
        "nodes": [
            {"id": "s", "type": "decision", "branches": [{"label": "A", "child_id": "c"}]},
            {"id": "c", "type": "chance", "branches": [
                {"label": "x", "child_id": "t1", "probability": "p_deth"},
                {"label": "y", "child_id": "t2", "probability": "#"}]},
            {"id": "t1", "type": "terminal", "cost": "cost_a + sqt(4)", "effect": "1 +", "branches": []},
            {"id": "t2", "type": "terminal", "cost": "DistToProb(w, 1, 2) + DistToProb(ww, 1, 2)", "effect": 0, "branches": []},
        ],
    }


def test_lists_every_problem_at_once():
    issues = check_document_formulas(_doc())
    text = "\n".join(i.message for i in issues)
    assert "Unknown name 'p_deth'" in text and "Did you mean 'p_death'" in text
    assert "Unknown function 'sqt'" in text and "sqrt" in text
    assert "Cannot parse" in text
    assert "Unknown distribution or table 'ww'" in text and "'w'" in text
    assert len(issues) == 4


def test_clean_documents_and_builtin_structures_have_no_issues():
    doc = _doc()
    doc["nodes"][1]["branches"][0]["probability"] = "p_death"
    doc["nodes"][2]["cost"] = "cost_a + sqrt(4) + _cycle"
    doc["nodes"][2]["effect"] = "if(p_death > 0.05, 1, 2)"
    doc["nodes"][3]["cost"] = "DistToProb(w, 1, 2)"
    assert check_document_formulas(doc) == []
    d2 = {"root_id": "strategies", "parameters": [], "nodes": [{"id": "strategies", "type": "decision", "branches": []}]}
    add_matrix_bayes_structure(d2, "strategies", "t", ["a", "b"], ["0.3", "#"], ["n", "p"], [["0.9", "#"], ["0.2", "#"]])
    assert check_document_formulas(d2) == []


def test_duplicates_and_cycles():
    doc = _doc()
    doc["nodes"][1]["branches"][0]["probability"] = "p_death"
    doc["nodes"][2]["cost"] = "1"
    doc["nodes"][2]["effect"] = "1"
    doc["nodes"][3]["cost"] = "1"
    doc["variables"] = [{"name": "a", "definition": "b + 1"}, {"name": "b", "definition": "a * 2"},
                        {"name": "p_death", "definition": "0.1"}]
    msgs = [i.message for i in check_document_formulas(doc)]
    assert any("Circular definition" in m for m in msgs)
    assert any("both a parameter and a variable" in m for m in msgs)


def test_trackers_are_known_names_and_updates_are_checked():
    doc = _doc()
    doc["nodes"][1]["branches"][0]["probability"] = "p_death"
    doc["nodes"][2]["cost"] = "1"
    doc["nodes"][2]["effect"] = "1"
    doc["nodes"][3]["cost"] = "n * 2"          # tracker used in a payoff
    doc["trackers"] = [{"name": "n", "initial": 0}]
    doc["nodes"][1]["tracker_updates"] = [{"tracker": "n", "expr": "n + p_death"}]
    assert check_document_formulas(doc) == []
    doc["nodes"][1]["tracker_updates"] = [{"tracker": "n", "expr": "nn + 1"}]
    msgs = [i.message for i in check_document_formulas(doc)]
    assert any("Unknown name 'nn'" in m and "'n'" in m for m in msgs)
