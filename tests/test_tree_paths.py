"""Tests for per-path results (tree_paths.enumerate_paths): every path's
p / cost / effect must reconcile with Engine.run's expected values."""

import math

from heor_tree_engine import (
    Branch,
    ChanceNode,
    DecisionNode,
    Engine,
    Param,
    TerminalNode,
)
from heor_tree_engine.tree_paths import (
    enumerate_paths,
    paths_to_records,
    strategy_expected_values,
    trace_to_records,
)


def _tree(with_complement: bool = False, year: float = 0.0) -> DecisionNode:
    second_p = "#" if with_complement else 0.6
    return DecisionNode(
        id="d1",
        branches=[
            Branch(
                label="Drug",
                cost_increment=300.0,
                effect_increment=0.1,
                child=ChanceNode(
                    id="c1",
                    branches=[
                        Branch(
                            label="Response",
                            probability=0.4,
                            cost_increment=10.0,
                            effect_increment=0.5,
                            extra_outcomes={"cost_2": 3.0},
                            child=TerminalNode(
                                id="t1", cost=100.0, effect=2.0, year=year,
                                extra_outcomes={"cost_2": 7.0},
                            ),
                        ),
                        Branch(
                            label="No response",
                            probability=second_p,
                            cost_increment=20.0,
                            effect_increment=0.0,
                            child=TerminalNode(id="t2", cost=250.0, effect=1.0, year=year),
                        ),
                    ],
                ),
            ),
            Branch(
                label="Usual care",
                cost_increment=50.0,
                child=TerminalNode(id="t3", cost=10.0, effect=1.5, year=year),
            ),
        ],
    )


def _check(tree: DecisionNode, cost_rate: float, effect_rate: float, kinds: dict[str, str]) -> None:
    result = Engine().run(
        tree, {}, cost_discount_rate=cost_rate, effect_discount_rate=effect_rate, outcome_kinds=kinds
    )
    assert not result.errors, result.errors
    rows = enumerate_paths(
        tree, {}, cost_discount_rate=cost_rate, effect_discount_rate=effect_rate, outcome_kinds=kinds
    )
    costs = strategy_expected_values(rows, "cost")
    effects = strategy_expected_values(rows, "effect")
    for label, strategy in result.strategies.items():
        assert math.isclose(costs[label], strategy.cost, rel_tol=1e-12, abs_tol=1e-12), label
        assert math.isclose(effects[label], strategy.effect, rel_tol=1e-12, abs_tol=1e-12), label
        for outcome_id in kinds:
            extra = strategy_expected_values(rows, outcome_id)[label]
            assert math.isclose(extra, strategy.extra_totals[outcome_id], abs_tol=1e-12), (label, outcome_id)
        # Probabilities of a strategy's paths sum to 1.
        assert math.isclose(sum(r.p for r in rows if r.strategy == label), 1.0, abs_tol=1e-12)


def test_paths_match_engine_no_discount():
    _check(_tree(), 0.0, 0.0, {"cost_2": "cost"})


def test_paths_match_engine_with_complement():
    _check(_tree(with_complement=True), 0.0, 0.0, {})


def test_paths_match_engine_with_discounting_and_year():
    _check(_tree(year=5.0), 0.03, 0.015, {"cost_2": "cost"})


def test_path_values_and_labels():
    rows = enumerate_paths(_tree(), {}, outcome_kinds={"cost_2": "cost"})
    first = rows[0]
    assert first.labels == ["Drug", "Response"]
    assert first.path_label == "Drug → Response"
    assert first.p == 0.4
    # strategy branch 300 + chance branch 10 + terminal 100
    assert first.undiscounted["cost"] == 410.0
    assert math.isclose(first.undiscounted["effect"], 0.1 + 0.5 + 2.0)
    assert first.undiscounted["cost_2"] == 3.0 + 7.0
    # Strategy with no chance node: single path with p == 1.
    usual = [r for r in rows if r.strategy == "Usual care"]
    assert len(usual) == 1 and usual[0].p == 1.0 and usual[0].undiscounted["cost"] == 60.0


def test_discounting_only_touches_leaf():
    rows = enumerate_paths(_tree(year=2.0), {}, cost_discount_rate=0.10)
    first = rows[0]
    assert first.undiscounted["cost"] == 410.0
    assert math.isclose(first.discounted["cost"], 300.0 + 10.0 + 100.0 / 1.1 ** 2)
    leaf = first.steps[-1]
    assert leaf.node_type == "leaf"
    assert math.isclose(leaf.cum_discounted["cost"], first.discounted["cost"])


def test_trace_and_records_shape():
    kinds = {"cost_2": "cost"}
    rows = enumerate_paths(_tree(), {}, outcome_kinds=kinds)
    ids = ["cost", "effect", "cost_2"]
    symbols = {"cost": "$", "effect": "E", "cost_2": "$2"}
    paths = paths_to_records(rows, ids, symbols)
    assert len(paths) == 3
    assert {"strategy", "path_id", "path", "p", "$ (undiscounted)", "$ (discounted)",
            "E (discounted)", "$2 (undiscounted)", "p x $ (discounted)"} <= set(paths[0])
    trace = trace_to_records(rows, ids, symbols)
    # Path 1: strategy step, chance step, leaf row.
    first_path = [t for t in trace if t["path_id"] == 1]
    assert [t["node_type"] for t in first_path] == ["decision", "chance", "leaf"]
    assert first_path[0]["p_step"] == "" and first_path[1]["p_step"] == 0.4
    assert first_path[-1]["$ cumulative (discounted)"] == paths[0]["$ (discounted)"]


def test_parameter_probabilities_resolve():
    tree = _tree()
    tree.branches[0].child.branches[0].probability = "p_resp"
    tree.branches[0].child.branches[1].probability = "#"
    params = {"p_resp": Param(name="p_resp", base=0.25)}
    rows = enumerate_paths(tree, params)
    assert math.isclose(rows[0].p, 0.25) and math.isclose(rows[1].p, 0.75)
