import pytest

from heor_tree_engine import DecisionNode, evaluate_decision, resolve


DOCUMENT = {
    "root_id": "strategies",
    "parameters": [],
    "nodes": [
        {
            "id": "strategies",
            "type": "decision",
            "branches": [
                {
                    "label": "standard_care",
                    "child_id": "standard_terminal",
                    "cost_increment": 20,
                },
                {"label": "new_treatment", "child_id": "outcomes"},
            ],
        },
        {
            "id": "outcomes",
            "type": "chance",
            "branches": [
                {"label": "good", "child_id": "good_terminal", "probability": 0.7},
                {
                    "label": "poor",
                    "child_id": "poor_terminal",
                    "probability": 0.3,
                    "cost_increment": 10,
                },
            ],
        },
        {"id": "standard_terminal", "type": "terminal", "cost": 100, "effect": 2},
        {"id": "good_terminal", "type": "terminal", "cost": 100, "effect": 2},
        {"id": "poor_terminal", "type": "terminal", "cost": 250, "effect": 1},
    ],
}


def test_flat_json_resolves_to_same_rollback_values() -> None:
    root, parameters = resolve(DOCUMENT)

    assert isinstance(root, DecisionNode)
    result = evaluate_decision(root, parameters)
    assert result.strategies["standard_care"].cost == 120.0
    assert result.strategies["new_treatment"].cost == 148.0
    assert result.strategies["new_treatment"].effect == 1.7


def test_parser_rejects_orphan_and_cycle() -> None:
    with pytest.raises(ValueError, match="Orphan"):
        resolve({"root_id": "root", "nodes": [
            {"id": "root", "type": "terminal"},
            {"id": "orphan", "type": "terminal"},
        ]})

    with pytest.raises(ValueError, match="Cycle"):
        resolve({"root_id": "a", "nodes": [
            {"id": "a", "type": "decision", "branches": [
                {"label": "to_b", "child_id": "b"},
            ]},
            {"id": "b", "type": "decision", "branches": [
                {"label": "to_a", "child_id": "a"},
            ]},
        ]})