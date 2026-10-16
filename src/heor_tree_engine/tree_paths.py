"""Per-path results for a decision tree (Tree Editor only).

A *path* runs from the root Decision node through one strategy branch and
every Chance branch below it down to a leaf. For each path this module
reports the joint probability, the summed Cost/Effect (and any extra
Cost/Effect outcomes) both undiscounted and discounted, and a step-by-step
trace of how those numbers were built up.

Nothing here re-implements rollback maths: every number comes from the
same private helpers ``engine.evaluate_decision`` uses (``_resolve``,
``_branch_cost``, ``_branch_effect``, ``_branch_extra_outcomes``,
``_evaluate_node`` for the leaf), so per-path values cannot drift from the
Run/ICER results. ``sum(p * cost_discounted)`` over a strategy's paths
equals that strategy's expected cost from ``Engine.run`` -- the tests
assert exactly that.

Discounting follows the engine: branch cost/effect increments are NOT
discounted; only a leaf's payoff is, by the leaf's own ``year``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .engine import (
    _branch_cost,
    _branch_effect,
    _branch_extra_outcomes,
    _engine_variables,
    _evaluate_node,
    _resolve,
)
from .expressions import resolve_variables
from .models import (
    ChanceNode,
    DecisionNode,
    Node,
    Param,
    TerminalNode,
    is_complement,
)

PATH_SEPARATOR = " → "


@dataclass
class PathStep:
    """One row of a path's trace: a branch taken, or the final leaf payoff."""

    step: int
    node_type: str  # type of the node the branch leaves ("decision"/"chance"), or "leaf"
    label: str  # branch label ("" for the leaf row)
    p_step: float | None  # None for a strategy branch (no probability) and the leaf row
    p_cumulative: float
    # outcome id -> value for this step; ids are "cost", "effect" and any extra outcome id.
    undiscounted: dict[str, float] = field(default_factory=dict)
    discounted: dict[str, float] = field(default_factory=dict)
    cum_undiscounted: dict[str, float] = field(default_factory=dict)
    cum_discounted: dict[str, float] = field(default_factory=dict)


@dataclass
class PathRow:
    strategy: str
    path_id: int  # 1-based, unique across the whole tree, in tree order
    labels: list[str]  # branch labels from the strategy branch down to the leaf
    leaf_id: str  # id of the leaf node the path ends at
    leaf_type: str  # "terminal", or e.g. "psm"/"des" when a subtree ends the path
    p: float  # joint probability of the chance branches along the path (within its strategy)
    undiscounted: dict[str, float]  # totals per outcome id
    discounted: dict[str, float]
    steps: list[PathStep] = field(default_factory=list)

    @property
    def path_label(self) -> str:
        return PATH_SEPARATOR.join(self.labels)


def _leaf_type(node: Node) -> str:
    if isinstance(node, TerminalNode):
        return "terminal"
    return type(node).__name__.replace("Node", "").replace("Tree", "").lower() or "node"


def _chance_probabilities(
    node: ChanceNode,
    parameters: Mapping[str, Param],
    variables: Mapping[str, Any],
    distributions: Mapping[str, Mapping[str, Any]] | None,
    tables: Mapping[str, Mapping[str, Any]] | None,
) -> list[float]:
    """Branch probabilities in branch order, resolving a "#"/complement
    branch exactly like ``engine._evaluate_node`` does."""

    resolved: dict[int, float] = {}
    complement_index: int | None = None
    known_sum = 0.0
    for index, branch in enumerate(node.branches):
        if branch.probability is None or branch.child is None:
            raise ValueError(f"Chance branch '{branch.label}' is incomplete")
        if is_complement(branch.probability):
            complement_index = index
            continue
        probability = _resolve(branch.probability, parameters, variables, distributions, tables)
        resolved[index] = probability
        known_sum += probability
    if complement_index is not None:
        resolved[complement_index] = 1.0 - known_sum
    return [resolved[index] for index in range(len(node.branches))]


def _add(target: dict[str, float], source: Mapping[str, float]) -> dict[str, float]:
    result = dict(target)
    for key, value in source.items():
        result[key] = result.get(key, 0.0) + value
    return result


def enumerate_paths(
    tree: DecisionNode,
    parameters: Mapping[str, Param] | None = None,
    variables: Mapping[str, Any] | None = None,
    distributions: Mapping[str, Mapping[str, Any]] | None = None,
    cost_discount_rate: float = 0.0,
    effect_discount_rate: float = 0.0,
    outcome_kinds: Mapping[str, str] | None = None,
    tables: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[PathRow]:
    """Every root-to-leaf path of ``tree`` with probability, undiscounted and
    discounted outcome totals, and the step-by-step trace.

    Arguments mirror ``Engine.run``. Raises ``ValueError`` for an
    incomplete tree, like the engine; call it after a successful run.
    A leaf that is not a plain Terminal (a PSM or DES subtree) ends its
    path with the subtree's expected payoff, computed by the engine.
    """

    outcome_kinds = outcome_kinds or {}
    engine_variables = _engine_variables(variables)
    resolved_parameters = resolve_variables(parameters or {}, engine_variables, distributions, tables)
    outcome_ids = ["cost", "effect", *outcome_kinds]
    rows: list[PathRow] = []

    def branch_values(branch: Any) -> dict[str, float]:
        values = {
            "cost": _branch_cost(branch, resolved_parameters, engine_variables, distributions, tables),
            "effect": _branch_effect(branch, resolved_parameters, engine_variables, distributions, tables),
        }
        if outcome_kinds:
            extras = _branch_extra_outcomes(
                branch, outcome_kinds, resolved_parameters, engine_variables, distributions, tables
            )
            for outcome_id in outcome_kinds:
                values[outcome_id] = extras.get(outcome_id, 0.0)
        return {outcome_id: values.get(outcome_id, 0.0) for outcome_id in outcome_ids}

    def leaf_values(node: Node, cost_rate: float, effect_rate: float) -> dict[str, float]:
        result = _evaluate_node(
            node, resolved_parameters, engine_variables, distributions,
            cost_rate, effect_rate, outcome_kinds, tables,
        )
        values = {"cost": result.cost, "effect": result.effect}
        for outcome_id in outcome_kinds:
            values[outcome_id] = result.extra_totals.get(outcome_id, 0.0)
        return {outcome_id: float(values.get(outcome_id, 0.0) or 0.0) for outcome_id in outcome_ids}

    def walk(
        node: Node,
        strategy: str,
        labels: list[str],
        steps: list[PathStep],
        p: float,
        cum: dict[str, float],
    ) -> None:
        if isinstance(node, ChanceNode):
            probabilities = _chance_probabilities(node, resolved_parameters, engine_variables, distributions, tables)
            for branch, probability in zip(node.branches, probabilities):
                values = branch_values(branch)
                new_p = p * probability
                new_cum = _add(cum, values)
                step = PathStep(
                    step=len(steps) + 1,
                    node_type="chance",
                    label=branch.label,
                    p_step=probability,
                    p_cumulative=new_p,
                    undiscounted=values,
                    discounted=dict(values),
                    cum_undiscounted=new_cum,
                    cum_discounted=dict(new_cum),
                )
                walk(branch.child, strategy, labels + [branch.label], steps + [step], new_p, new_cum)
            return
        if isinstance(node, DecisionNode):
            # Only the root is a decision in a Tree Editor model; a nested one
            # is still walked (each of its branches is a further strategy step).
            for branch in node.branches:
                values = branch_values(branch)
                new_cum = _add(cum, values)
                step = PathStep(
                    step=len(steps) + 1,
                    node_type="decision",
                    label=branch.label,
                    p_step=None,
                    p_cumulative=p,
                    undiscounted=values,
                    discounted=dict(values),
                    cum_undiscounted=new_cum,
                    cum_discounted=dict(new_cum),
                )
                walk(branch.child, strategy, labels + [branch.label], steps + [step], p, new_cum)
            return

        undiscounted_leaf = leaf_values(node, 0.0, 0.0)
        discounted_leaf = leaf_values(node, cost_discount_rate, effect_discount_rate)
        cum_undiscounted = _add(
            {k: v for k, v in _branch_totals(steps, "undiscounted").items()}, undiscounted_leaf
        )
        cum_discounted = _add(
            {k: v for k, v in _branch_totals(steps, "discounted").items()}, discounted_leaf
        )
        leaf_step = PathStep(
            step=len(steps) + 1,
            node_type="leaf",
            label="",
            p_step=None,
            p_cumulative=p,
            undiscounted=undiscounted_leaf,
            discounted=discounted_leaf,
            cum_undiscounted=cum_undiscounted,
            cum_discounted=cum_discounted,
        )
        rows.append(
            PathRow(
                strategy=strategy,
                path_id=len(rows) + 1,
                labels=list(labels),
                leaf_id=getattr(node, "id", ""),
                leaf_type=_leaf_type(node),
                p=p,
                undiscounted={oid: cum_undiscounted.get(oid, 0.0) for oid in outcome_ids},
                discounted={oid: cum_discounted.get(oid, 0.0) for oid in outcome_ids},
                steps=steps + [leaf_step],
            )
        )

    def _branch_totals(steps: list[PathStep], kind: str) -> dict[str, float]:
        total: dict[str, float] = {}
        for step in steps:
            total = _add(total, getattr(step, kind))
        return total

    for branch in tree.branches:
        if branch.child is None:
            raise ValueError(f"Decision branch '{branch.label}' is incomplete")
        values = branch_values(branch)
        step = PathStep(
            step=1,
            node_type="decision",
            label=branch.label,
            p_step=None,
            p_cumulative=1.0,
            undiscounted=values,
            discounted=dict(values),
            cum_undiscounted=dict(values),
            cum_discounted=dict(values),
        )
        walk(branch.child, branch.label, [branch.label], [step], 1.0, dict(values))
    return rows


def strategy_expected_values(rows: list[PathRow], outcome_id: str = "cost", discounted: bool = True) -> dict[str, float]:
    """sum(p * value) per strategy -- what must equal ``Engine.run``'s
    per-strategy expected cost/effect. Used by the tests and as a runtime
    self-check in the app."""

    totals: dict[str, float] = {}
    for row in rows:
        values = row.discounted if discounted else row.undiscounted
        totals[row.strategy] = totals.get(row.strategy, 0.0) + row.p * values.get(outcome_id, 0.0)
    return totals


def paths_to_records(rows: list[PathRow], outcome_ids: list[str], outcome_symbols: Mapping[str, str]) -> list[dict[str, Any]]:
    """Flat one-row-per-path records (Paths table / Paths CSV). Columns per
    outcome: ``<symbol> (undiscounted)`` / ``<symbol> (discounted)``, then
    p x discounted value."""

    records: list[dict[str, Any]] = []
    for row in rows:
        record: dict[str, Any] = {
            "strategy": row.strategy,
            "path_id": row.path_id,
            "path": row.path_label,
            "leaf": row.leaf_type,
            "p": row.p,
        }
        for outcome_id in outcome_ids:
            symbol = outcome_symbols.get(outcome_id, outcome_id)
            record[f"{symbol} (undiscounted)"] = row.undiscounted.get(outcome_id, 0.0)
            record[f"{symbol} (discounted)"] = row.discounted.get(outcome_id, 0.0)
        for outcome_id in outcome_ids:
            symbol = outcome_symbols.get(outcome_id, outcome_id)
            record[f"p x {symbol} (discounted)"] = row.p * row.discounted.get(outcome_id, 0.0)
        records.append(record)
    return records


def trace_to_records(rows: list[PathRow], outcome_ids: list[str], outcome_symbols: Mapping[str, str]) -> list[dict[str, Any]]:
    """Flat one-row-per-step records (Trace CSV)."""

    records: list[dict[str, Any]] = []
    for row in rows:
        for step in row.steps:
            record: dict[str, Any] = {
                "strategy": row.strategy,
                "path_id": row.path_id,
                "step": step.step,
                "node_type": step.node_type,
                "branch_label": step.label,
                "p_step": "" if step.p_step is None else step.p_step,
                "p_cumulative": step.p_cumulative,
            }
            for outcome_id in outcome_ids:
                symbol = outcome_symbols.get(outcome_id, outcome_id)
                record[f"{symbol} step (undiscounted)"] = step.undiscounted.get(outcome_id, 0.0)
                record[f"{symbol} step (discounted)"] = step.discounted.get(outcome_id, 0.0)
                record[f"{symbol} cumulative (undiscounted)"] = step.cum_undiscounted.get(outcome_id, 0.0)
                record[f"{symbol} cumulative (discounted)"] = step.cum_discounted.get(outcome_id, 0.0)
            records.append(record)
    return records
