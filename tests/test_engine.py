from heor_tree_engine import (
    Branch,
    ChanceNode,
    DecisionNode,
    TerminalNode,
    evaluate_decision,
    expected_cost,
    expected_utility,
)


def test_chance_rollback_includes_one_time_branch_cost() -> None:
    tree = ChanceNode(
        id="outcomes",
        branches=[
            Branch(
                label="good",
                child=TerminalNode(id="good_terminal", cost=100.0, effect=2.0),
                probability=0.7,
                cost_increment=10.0,
            ),
            Branch(
                label="poor",
                child=TerminalNode(id="poor_terminal", cost=250.0, effect=1.0),
                probability=0.3,
            ),
        ],
    )

    assert expected_cost(tree) == 152.0
    assert expected_utility(tree) == 1.7


def test_decision_rollback_returns_each_strategy() -> None:
    tree = DecisionNode(
        id="strategies",
        branches=[
            Branch(
                label="standard_care",
                child=TerminalNode(id="standard_terminal", cost=100.0, effect=2.0),
                cost_increment=20.0,
            ),
            Branch(
                label="new_treatment",
                child=TerminalNode(id="new_terminal", cost=180.0, effect=2.5),
            ),
        ],
    )

    result = evaluate_decision(tree)

    assert result.strategies["standard_care"].cost == 120.0
    assert result.strategies["standard_care"].effect == 2.0
    assert result.strategies["new_treatment"].cost == 180.0
    assert result.strategies["new_treatment"].effect == 2.5
    assert result.strategies["standard_care"].paths == [
        ["strategies", "standard_care", "standard_terminal"]
    ]


def test_decision_tree_provides_default_cycle_and_tenure_variables() -> None:
    tree = DecisionNode(
        id="strategies",
        branches=[
            Branch(
                label="standard_care",
                child=TerminalNode(id="terminal", cost="_cycle + _tenure", effect="_cycle"),
            )
        ],
    )

    result = evaluate_decision(tree)

    assert result.strategies["standard_care"].cost == 1.0
    assert result.strategies["standard_care"].effect == 0.0