import pytest

from heor_tree_engine.bayes_revision import add_matrix_bayes_structure, flip_chance_tree, posterior_matrix
from heor_tree_engine.engine import terminal_probabilities
from heor_tree_engine.parser import resolve


def _doc():
    return {"root_id": "strategies", "parameters": [],
            "nodes": [{"id": "strategies", "type": "decision", "branches": []}]}


STATES = ["Healthy", "Mild", "Severe"]
RESULTS = ["Neg", "Equivocal", "Pos"]
PRIORS = ["0.7", "0.2", "#"]
LIK = [["0.9", "0.08", "#"], ["0.2", "0.3", "0.5"], ["0.05", "#", "0.75"]]


def _probs(doc):
    doc["nodes"][0]["branches"][0]["label"] = "S"
    root, params = resolve(doc)
    return terminal_probabilities(root, "S", params, {}, {}, {})


def test_matrix_matches_bayes_rule():
    doc = _doc()
    err, tid = add_matrix_bayes_structure(doc, "strategies", "t", STATES, PRIORS, RESULTS, LIK)
    assert err is None and tid
    pri = [0.7, 0.2, 0.1]
    lik = [[0.9, 0.08, 0.02], [0.2, 0.3, 0.5], [0.05, 0.2, 0.75]]
    ref = posterior_matrix(pri, lik)
    probs = _probs(doc)
    assert sum(probs.values()) == pytest.approx(1.0)
    for j in range(3):
        for i in range(3):
            # joint P(state i, result j) = prior_i * L_ij
            assert probs[f"t_r{j + 1}_s{i + 1}"] == pytest.approx(pri[i] * lik[i][j])
    assert sum(ref["p_result"]) == pytest.approx(1.0)


def test_matrix_follows_parameter_changes_and_validates():
    doc = _doc()
    add_matrix_bayes_structure(doc, "strategies", "t", STATES, PRIORS, RESULTS, LIK)
    for row in doc["parameters"]:
        if row["name"] == "t_prev1":
            row["base"] = 0.5
    probs = _probs(doc)
    assert probs["t_r1_s1"] == pytest.approx(0.5 * 0.9)
    assert sum(probs.values()) == pytest.approx(1.0)
    d2 = _doc()
    assert add_matrix_bayes_structure(d2, "strategies", "t", STATES, ["0.9", "0.3", "#"], RESULTS, LIK)[0]
    assert add_matrix_bayes_structure(d2, "strategies", "t", STATES[:1], ["1"], RESULTS, LIK[:1])[0]
    assert add_matrix_bayes_structure(d2, "strategies", "t", STATES, PRIORS, RESULTS, [["0.5", "0.6", "#"]] * 3)[0]


def _flip_doc():
    t = lambda i: {"id": i, "type": "terminal", "branches": []}
    return {"root_id": "strategies", "parameters": [{"name": "prev", "base": 0.1}, {"name": "se", "base": 0.9},
                                                     {"name": "sp", "base": 0.8}],
            "nodes": [
                {"id": "strategies", "type": "decision", "branches": [{"label": "S", "child_id": "d"}]},
                {"id": "d", "type": "chance", "branches": [
                    {"label": "D", "child_id": "td", "probability": "prev"},
                    {"label": "ND", "child_id": "tn", "probability": "#"}]},
                {"id": "td", "type": "chance", "branches": [
                    {"label": "T+", "child_id": "dp", "probability": "se"},
                    {"label": "T-", "child_id": "dn", "probability": "#"}]},
                {"id": "tn", "type": "chance", "branches": [
                    {"label": "T-", "child_id": "np", "probability": "sp"},   # order differs on purpose
                    {"label": "T+", "child_id": "nn", "probability": "#"}]},
                t("dp"), t("dn"), t("np"), t("nn")]}


def test_flip_preserves_joint_probabilities_and_gives_ppv():
    doc = _flip_doc()
    before = _probs(doc)
    err, nid = flip_chance_tree(doc, "d", "f")
    assert err is None and nid == "d"
    after = _probs(doc)
    for k in before:
        assert after[k] == pytest.approx(before[k])
    assert before["dp"] == pytest.approx(0.09) and before["nn"] == pytest.approx(0.18)
    # new structure: d -> result nodes -> original terminals
    root, params = resolve(doc)
    assert params["f_pD1_T1"].base == pytest.approx(0.09 / 0.27)   # PPV
    assert params["f_pT1"].base == pytest.approx(0.27)


def test_flip_follows_parameter_changes_and_double_flip_roundtrip():
    doc = _flip_doc()
    assert flip_chance_tree(doc, "d", "f")[0] is None
    for row in doc["parameters"]:
        if row["name"] == "se":
            row["base"] = 0.5
    p = _probs(doc)
    assert p["dp"] == pytest.approx(0.05) and sum(p.values()) == pytest.approx(1.0)
    # flip back
    assert flip_chance_tree(doc, "d", "g")[0] is None
    q = _probs(doc)
    for k in p:
        assert q[k] == pytest.approx(p[k])


def test_flip_rejects_bad_structure():
    doc = _flip_doc()
    assert flip_chance_tree(doc, "strategies", "f")[0]       # decision node
    assert flip_chance_tree(doc, "td", "f")[0]               # children are terminals
    doc["nodes"][3]["branches"][0]["label"] = "other"
    assert flip_chance_tree(doc, "d", "f")[0]                # label sets differ
    doc2 = _flip_doc()
    doc2["nodes"][2]["branches"][0]["cost"] = 5
    assert "payoff" in flip_chance_tree(doc2, "d", "f")[0]
