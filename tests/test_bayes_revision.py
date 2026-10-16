import pytest

from heor_tree_engine.bayes_revision import add_binary_bayes_structure, posterior_binary
from heor_tree_engine.engine import terminal_probabilities
from heor_tree_engine.parser import resolve


def _doc() -> dict:
    return {
        "root_id": "strategies",
        "parameters": [],
        "nodes": [{"id": "strategies", "type": "decision", "branches": []}],
    }


def test_posterior_binary_closed_form() -> None:
    out = posterior_binary(0.1, 0.9, 0.8)
    assert out["p_test_pos"] == pytest.approx(0.27)
    assert out["p_disease_given_pos"] == pytest.approx(1 / 3)
    assert out["p_disease_given_neg"] == pytest.approx(0.01 / 0.73)


def test_posterior_binary_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        posterior_binary(1.2, 0.9, 0.8)
    with pytest.raises(ValueError):
        posterior_binary(0.0, 0.9, 1.0)


def test_structure_runs_through_engine_and_matches_closed_form() -> None:
    doc = _doc()
    error, test_id = add_binary_bayes_structure(doc, "strategies", "scr", "0.1", "0.9", "0.8")
    assert error is None and test_id
    doc["nodes"][0]["branches"][0]["label"] = "Screen"
    root, params = resolve(doc)
    resolved = params  # resolve() already evaluates the document's variables
    probs = terminal_probabilities(root, "Screen", params, {}, {}, {})
    assert sum(probs.values()) == pytest.approx(1.0)
    # joint P(disease and T+) = Se * prevalence; P(disease and T-) = (1-Se) * prevalence
    assert probs["scr_pos_d"] == pytest.approx(0.9 * 0.1)
    assert probs["scr_neg_d"] == pytest.approx(0.1 * 0.1)
    assert probs["scr_pos_nd"] == pytest.approx(0.2 * 0.9)
    assert probs["scr_neg_nd"] == pytest.approx(0.8 * 0.9)
    assert resolved["scr_p_pos"].base == pytest.approx(0.27)


def test_posteriors_follow_parameter_changes() -> None:
    doc = _doc()
    add_binary_bayes_structure(doc, "strategies", "scr", "0.1", "0.9", "0.8")
    doc["nodes"][0]["branches"][0]["label"] = "Screen"
    for row in doc["parameters"]:
        if row["name"] == "scr_se":
            row["base"] = 0.5  # what a PSA draw / one-way SA step does
    root, params = resolve(doc)
    probs = terminal_probabilities(root, "Screen", params, {}, {}, {})
    assert probs["scr_pos_d"] == pytest.approx(0.5 * 0.1)
    assert sum(probs.values()) == pytest.approx(1.0)


def test_rejects_duplicate_names_and_bad_inputs() -> None:
    doc = _doc()
    assert add_binary_bayes_structure(doc, "strategies", "scr", "0.1", "0.9", "0.8")[0] is None
    assert add_binary_bayes_structure(doc, "strategies", "scr", "0.1", "0.9", "0.8")[0]
    assert add_binary_bayes_structure(doc, "strategies", "x", "1.5", "0.9", "0.8")[0]
    assert add_binary_bayes_structure(doc, "nope", "y", "0.1", "0.9", "0.8")[0]
    assert add_binary_bayes_structure(doc, "strategies", "1bad", "0.1", "0.9", "0.8")[0]


def test_chance_parent_gets_incomplete_probability() -> None:
    doc = _doc()
    doc["nodes"].append({"id": "c", "type": "chance", "branches": []})
    add_binary_bayes_structure(doc, "c", "t", "0.1", "0.9", "0.8")
    assert doc["nodes"][1]["branches"][0]["probability"] == "0.0"
