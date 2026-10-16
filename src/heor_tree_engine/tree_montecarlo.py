"""Group Monte Carlo with tracker variables for the decision tree.

TreeAge's *trackers* are values that travel with one simulated individual
through the tree: they start at an initial value, change at nodes
(``tAge = tAge + 1``) and can be used in later probabilities and payoffs.
That only has a meaning when individuals are simulated one by one, so this
module is a separate individual-level engine next to the expected-value
engine (``engine.Engine``), which ignores trackers.

Document schema (all optional):

* ``document["trackers"]``: ``[{"name": "nEvents", "initial": 0}, ...]``
  (``initial`` may be a number or an expression of parameters / earlier trackers).
* ``node["tracker_updates"]``: ``[{"tracker": "nEvents", "expr": "nEvents + 1"}, ...]``
  on a decision, chance or terminal node, applied in order when the individual
  *enters* that node (a terminal's updates run before its payoff is evaluated).

How an individual is simulated: at a chance node one branch is drawn with the
node's probabilities (evaluated with the current trackers); branch cost/effect
increments are added along the way; the leaf payoff is discounted by the leaf's
``year`` exactly like the expected-value engine.  A leaf that is not a plain
terminal (a Markov / PSM / DES node) contributes its *expected value* -- it has
no individual-level simulation here.  Every strategy is run for the same
individuals with the same random stream (common random numbers), so differences
between strategies are paired.

Limits: trackers are not visible to document-level *variables* (those are
resolved once, before any individual exists) -- use them in node expressions.
A nested decision node inside a strategy is not supported.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .engine import (
    _branch_cost,
    _branch_effect,
    _branch_extra_outcomes,
    _engine_variables,
    _evaluate_node,
)
from .expressions import evaluate_expression, resolve_variables
from .models import ChanceNode, DecisionNode, Node, Param, TerminalNode
from .tree_paths import _chance_probabilities

_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass
class TrackerSpec:
    name: str
    initial: Any = 0.0


def trackers_from_document(document: Mapping[str, Any]) -> list[TrackerSpec]:
    return [
        TrackerSpec(str(row["name"]).strip(), row.get("initial", 0.0))
        for row in (document.get("trackers") or [])
        if isinstance(row, Mapping) and str(row.get("name", "")).strip()
    ]


def node_updates_from_document(document: Mapping[str, Any]) -> dict[str, list[tuple[str, str]]]:
    """{node id: [(tracker, expression), ...]} -- call on the clone-expanded document."""
    updates: dict[str, list[tuple[str, str]]] = {}
    for node in document.get("nodes", []) or []:
        rows = node.get("tracker_updates") or []
        pairs = [(str(r.get("tracker", "")).strip(), str(r.get("expr", "")).strip())
                 for r in rows if isinstance(r, Mapping)]
        pairs = [p for p in pairs if p[0]]
        if pairs:
            updates[str(node.get("id"))] = pairs
    return updates


def tracker_problems(document: Mapping[str, Any]) -> list[str]:
    """Human-readable problems with the tracker definitions / updates (empty = fine)."""
    problems: list[str] = []
    trackers = trackers_from_document(document)
    names = [t.name for t in trackers]
    taken = {str(r.get("name")) for r in (document.get("parameters") or []) if isinstance(r, Mapping)}
    taken |= {str(r.get("name")) for r in (document.get("variables") or []) if isinstance(r, Mapping)}
    seen: set[str] = set()
    for name in names:
        if not _IDENT.match(name):
            problems.append(f"Tracker name '{name}' must be letters, digits or underscores and not start with a digit.")
        if name in seen:
            problems.append(f"Duplicate tracker '{name}'.")
        seen.add(name)
        if name in taken:
            problems.append(f"Tracker '{name}' has the same name as a parameter or variable.")
        if name.startswith("_"):
            problems.append(f"Tracker '{name}': names starting with '_' are reserved.")
    for node_id, pairs in node_updates_from_document(document).items():
        for tracker, expr in pairs:
            if tracker not in seen:
                problems.append(f"Node '{node_id}' updates unknown tracker '{tracker}'.")
            if not expr:
                problems.append(f"Node '{node_id}': empty update expression for '{tracker}'.")
    return problems


@dataclass
class GroupMCResult:
    n: int
    seed: int
    outcome_ids: list[str]
    outcomes: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)  # strategy -> outcome -> (n,)
    tracker_final: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)  # strategy -> tracker -> (n,)
    ev_leaf_types: set[str] = field(default_factory=set)  # leaf kinds that used the expected value
    warnings: list[str] = field(default_factory=list)

    @property
    def strategies(self) -> list[str]:
        return list(self.outcomes)

    def summary(self, level: float = 0.95) -> list[dict[str, Any]]:
        z = _z(level)
        rows = []
        for strategy, per_outcome in self.outcomes.items():
            row: dict[str, Any] = {"strategy": strategy}
            for oid in self.outcome_ids:
                x = per_outcome[oid]
                mean = float(x.mean())
                sd = float(x.std(ddof=1)) if self.n > 1 else 0.0
                se = sd / math.sqrt(self.n)
                row[oid] = {"mean": mean, "sd": sd, "se": se, "lower": mean - z * se, "upper": mean + z * se}
            rows.append(row)
        return rows

    def incremental(self, strategy: str, baseline: str, level: float = 0.95) -> dict[str, Any]:
        """Paired differences strategy - baseline (same individuals, same random numbers)."""
        z = _z(level)
        out: dict[str, Any] = {"strategy": strategy, "baseline": baseline}
        for oid in self.outcome_ids:
            d = self.outcomes[strategy][oid] - self.outcomes[baseline][oid]
            mean = float(d.mean())
            se = float(d.std(ddof=1)) / math.sqrt(self.n) if self.n > 1 else 0.0
            out[oid] = {"mean": mean, "se": se, "lower": mean - z * se, "upper": mean + z * se}
        dc, de = out["cost"]["mean"], out["effect"]["mean"]
        out["icer"] = dc / de if de else None
        return out


def _z(level: float) -> float:
    from statistics import NormalDist

    return NormalDist().inv_cdf(0.5 + level / 2.0)


def run_group_mc(
    tree: DecisionNode,
    parameters: Mapping[str, Param],
    trackers: list[TrackerSpec] | None = None,
    node_updates: Mapping[str, list[tuple[str, str]]] | None = None,
    n: int = 1000,
    seed: int = 2026,
    cost_discount_rate: float = 0.0,
    effect_discount_rate: float = 0.0,
    outcome_kinds: Mapping[str, str] | None = None,
    distributions: Mapping[str, Mapping[str, Any]] | None = None,
    tables: Mapping[str, Mapping[str, Any]] | None = None,
    variables: Mapping[str, Any] | None = None,
) -> GroupMCResult:
    if n < 1:
        raise ValueError("n must be at least 1")
    trackers = trackers or []
    node_updates = node_updates or {}
    outcome_kinds = dict(outcome_kinds or {})
    outcome_ids = ["cost", "effect", *outcome_kinds]
    base_vars = _engine_variables(variables)
    resolved = resolve_variables(parameters, base_vars, distributions, tables)
    result = GroupMCResult(n=n, seed=seed, outcome_ids=outcome_ids)
    strategies = [b.label for b in tree.branches]
    for branch in tree.branches:
        if branch.child is None:
            raise ValueError(f"Decision branch '{branch.label}' is incomplete")
        result.outcomes[branch.label] = {oid: np.zeros(n) for oid in outcome_ids}
        result.tracker_final[branch.label] = {t.name: np.zeros(n) for t in trackers}

    def ev(expr: Any, state: Mapping[str, float]) -> float:
        if isinstance(expr, (int, float)):
            return float(expr)
        return evaluate_expression(str(expr), resolved, {**base_vars, **state}, distributions, tables)

    def apply_updates(node_id: str, state: dict[str, float]) -> None:
        for tracker, expr in node_updates.get(node_id, ()):
            if tracker not in state:
                raise ValueError(f"Node '{node_id}' updates unknown tracker '{tracker}'")
            state[tracker] = ev(expr, state)

    def branch_values(branch, vars_now) -> dict[str, float]:
        values = {
            "cost": _branch_cost(branch, resolved, vars_now, distributions, tables),
            "effect": _branch_effect(branch, resolved, vars_now, distributions, tables),
        }
        if outcome_kinds:
            extras = _branch_extra_outcomes(branch, outcome_kinds, resolved, vars_now, distributions, tables)
            for oid in outcome_kinds:
                values[oid] = extras.get(oid, 0.0)
        return values

    def simulate(branch, rng, i: int, strategy: str) -> None:
        state: dict[str, float] = {}
        for t in trackers:
            state[t.name] = ev(t.initial, state)
        apply_updates(tree.id, state)
        totals = {oid: 0.0 for oid in outcome_ids}

        def add(values: Mapping[str, float]) -> None:
            for oid in outcome_ids:
                totals[oid] += values.get(oid, 0.0)

        add(branch_values(branch, {**base_vars, **state}))
        node: Node = branch.child
        while True:
            apply_updates(node.id, state)
            vars_now = {**base_vars, **state}
            if isinstance(node, ChanceNode):
                probs = _chance_probabilities(node, resolved, vars_now, distributions, tables)
                if any(p < -1e-12 for p in probs):
                    raise ValueError(f"Chance node '{node.id}' has a negative probability")
                total = sum(probs)
                if abs(total - 1.0) > 1e-6:
                    raise ValueError(f"Probabilities at chance node '{node.id}' sum to {total:.6g}, not 1")
                u = rng.random()
                acc, chosen = 0.0, len(probs) - 1
                for k, p in enumerate(probs):
                    acc += p
                    if u < acc:
                        chosen = k
                        break
                nxt = node.branches[chosen]
                add(branch_values(nxt, vars_now))
                node = nxt.child
                continue
            if isinstance(node, DecisionNode):
                raise ValueError(f"Nested decision node '{node.id}' is not supported in group Monte Carlo")
            if not isinstance(node, TerminalNode):
                result.ev_leaf_types.add(type(node).__name__.replace("Node", "").lower())
            leaf = _evaluate_node(node, resolved, vars_now, distributions,
                                  cost_discount_rate, effect_discount_rate, outcome_kinds, tables)
            totals["cost"] += leaf.cost
            totals["effect"] += leaf.effect
            for oid in outcome_kinds:
                totals[oid] += leaf.extra_totals.get(oid, 0.0)
            break
        for oid in outcome_ids:
            result.outcomes[strategy][oid][i] = totals[oid]
        for name, value in state.items():
            result.tracker_final[strategy][name][i] = value

    for i in range(n):
        for branch in tree.branches:
            # same stream for every strategy -> common random numbers
            simulate(branch, np.random.default_rng([seed, i]), i, branch.label)
    if result.ev_leaf_types:
        result.warnings.append(
            "Leaf nodes of type " + ", ".join(sorted(result.ev_leaf_types))
            + " use their expected value (no individual-level simulation inside them).")
    return result


def run_group_mc_document(
    document: Mapping[str, Any],
    n: int = 1000,
    seed: int = 2026,
    cost_discount_rate: float | None = None,
    effect_discount_rate: float | None = None,
    outcome_kinds: Mapping[str, str] | None = None,
) -> GroupMCResult:
    """Run from a Tree document: expands clones, reads trackers / updates, tables, distributions."""
    from .expressions import build_tables_mapping
    from .parser import resolve
    from .tree_clone import expand_tree_clones

    problems = tracker_problems(document)
    if problems:
        raise ValueError("; ".join(problems))
    expanded = expand_tree_clones(document)
    tree, parameters = resolve(document)
    return run_group_mc(
        tree, parameters, trackers_from_document(document), node_updates_from_document(expanded),
        n=n, seed=seed,
        cost_discount_rate=float(document.get("discount_cost_value", 0.0)) if cost_discount_rate is None else cost_discount_rate,
        effect_discount_rate=float(document.get("discount_effect_value", 0.0)) if effect_discount_rate is None else effect_discount_rate,
        outcome_kinds=outcome_kinds,
        distributions={str(r["name"]): r for r in document.get("distributions", []) if r.get("name")},
        tables=build_tables_mapping(document.get("tables")),
    )
