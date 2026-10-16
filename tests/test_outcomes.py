"""Tests for extra Cost/Effect outcomes beyond the primary pair (the
Outcomes table feature): Branch.extra_outcomes / TerminalNode.extra_outcomes,
threaded through Engine.run() via outcome_kinds."""

from heor_tree_engine import (
    Branch,
    ChanceNode,
    DecisionNode,
    Engine,
    TerminalNode,
)


def _tree() -> DecisionNode:
    return DecisionNode(
        id="d1",
        branches=[
            Branch(
                label="A",
                child=ChanceNode(
                    id="c1",
                    branches=[
                        Branch(
                            label="good",
                            child=TerminalNode(
                                id="t1", cost=100.0, effect=1.0,
                                extra_outcomes={"cost_2": 20.0, "effect_2": 0.5},
                            ),
                            probability=0.6,
                            cost_increment=10.0,
                            effect_increment=0.1,
                            extra_outcomes={"cost_2": 5.0, "effect_2": 0.01},
                        ),
                        Branch(
                            label="bad",
                            child=TerminalNode(id="t2", cost=200.0, effect=0.5),
                            probability=0.4,
                        ),
                    ],
                ),
            ),
            Branch(label="B", child=TerminalNode(id="t3", cost=50.0, effect=0.8, extra_outcomes={"cost_2": 3.0})),
        ],
    )


def test_extra_outcomes_are_ignored_without_outcome_kinds() -> None:
    result = Engine().run(_tree())
    assert not result.errors
    a = result.strategies["A"]
    assert a.extra_totals == {}
    assert a.cost == 0.6 * (10 + 100) + 0.4 * 200
    assert abs(a.effect - (0.6 * (0.1 + 1.0) + 0.4 * 0.5)) < 1e-9


def test_extra_cost_and_effect_outcomes_accumulate_like_the_primary_pair() -> None:
    outcome_kinds = {"cost_2": "cost", "effect_2": "effect"}
    result = Engine().run(_tree(), outcome_kinds=outcome_kinds)
    assert not result.errors
    a = result.strategies["A"]
    b = result.strategies["B"]
    assert abs(a.extra_totals["cost_2"] - (0.6 * (5 + 20) + 0.4 * 0)) < 1e-9
    assert abs(a.extra_totals["effect_2"] - (0.6 * (0.01 + 0.5) + 0.4 * 0)) < 1e-9
    assert abs(b.extra_totals["cost_2"] - 3.0) < 1e-9
    assert b.extra_totals.get("effect_2", 0.0) == 0.0
    # the primary pair is unaffected by extra outcomes being present
    assert a.cost == 0.6 * (10 + 100) + 0.4 * 200


def test_extra_outcomes_discount_by_their_own_kind() -> None:
    tree = DecisionNode(
        id="d1",
        branches=[
            Branch(
                label="A",
                child=TerminalNode(
                    id="t1", cost=100.0, effect=1.0, year=2.0,
                    extra_outcomes={"cost_2": 50.0, "effect_2": 0.4},
                ),
            ),
        ],
    )
    outcome_kinds = {"cost_2": "cost", "effect_2": "effect"}
    result = Engine().run(
        tree, outcome_kinds=outcome_kinds, cost_discount_rate=0.03, effect_discount_rate=0.015
    )
    a = result.strategies["A"]
    assert abs(a.cost - 100 / (1.03**2)) < 1e-9
    assert abs(a.extra_totals["cost_2"] - 50 / (1.03**2)) < 1e-9
    assert abs(a.effect - 1.0 / (1.015**2)) < 1e-9
    assert abs(a.extra_totals["effect_2"] - 0.4 / (1.015**2)) < 1e-9
