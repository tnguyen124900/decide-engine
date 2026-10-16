"""Per-cycle trace of a Partitioned Survival (PSM) node -- the single place that replays the engine's own PSM
arithmetic and hands the cycle-by-cycle numbers to the Excel export, the technical report, the validation
self-checks and the dashboard.

`engine._evaluate_node` only returns each PSM node's final (cost, effect); the per-cycle steps are never exposed.
Instead of re-deriving the survival / reward formulas here, this module imports the engine's own private helpers
(`_curve_survival`, `_branch_cost`, `_resolve`, ...) and replays the same loop, per-cycle `_cycle` injection
included, so the trace cannot drift from what Run / ICER compute.  `psm_validation_checks` then verifies the trace
against both the engine's own total and an independent re-implementation.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .engine import (
    _branch_cost,
    _branch_effect,
    _curve_survival,
    _engine_variables,
    _resolve,
)
from .expressions import resolve_variables
from .models import PSM_STATE_ROLES, Param, PSMNode, TerminalNode


@dataclass
class PSMTrace:
    node_id: str
    cycles: int
    cycles_per_year: float
    half_cycle_correction: bool
    cost_discount_rate: float
    effect_discount_rate: float
    # one dict per cycle 0..cycles (the last one is the end-of-horizon point used only by the half-cycle correction)
    rows: list[dict[str, float]] = field(default_factory=list)
    total_cost: float = 0.0  # discounted, half-cycle correction included -- equals the engine's value for the node
    total_effect: float = 0.0
    total_cost_undiscounted: float = 0.0
    total_effect_undiscounted: float = 0.0
    rates_cycle_dependent: bool = False  # a role's cost/effect rate changes with _cycle
    # the resolved numeric parameters of each curve, by name ({"pfs": {"shape": 1.5, ...}, "os": {...}})
    curve_params: dict[str, dict[str, float]] = field(default_factory=dict)


def find_psm_nodes(tree: Any) -> list[PSMNode]:
    """Every PSM node of a tree, in depth-first (canvas) order."""

    found: list[PSMNode] = []

    def walk(node: Any) -> None:
        if isinstance(node, PSMNode):
            found.append(node)
            return  # a PSM node's children are its three terminal states
        for branch in getattr(node, "branches", None) or []:
            if branch.child is not None:
                walk(branch.child)

    walk(tree)
    return found


def psm_branches_by_role(node: PSMNode) -> dict[str, Any]:
    by_role = {b.role: b for b in node.branches if b.role in PSM_STATE_ROLES}
    if set(by_role) != set(PSM_STATE_ROLES):
        raise ValueError(f"PSM node '{node.id}' needs one branch for each of: {', '.join(PSM_STATE_ROLES)}")
    for branch in by_role.values():
        if not isinstance(branch.child, TerminalNode):
            raise ValueError(f"PSM node '{node.id}' branch '{branch.label}' must lead to a Terminal node")
    return by_role


def compute_psm_trace(
    node: PSMNode,
    parameters: Mapping[str, Param],
    variables: Mapping[str, Any] | None = None,
    distributions: Mapping[str, Mapping[str, Any]] | None = None,
    tables: Mapping[str, Mapping[str, Any]] | None = None,
    cost_discount_rate: float = 0.0,
    effect_discount_rate: float = 0.0,
) -> PSMTrace:
    """Replay `engine._evaluate_node`'s PSMNode branch cycle by cycle.  `parameters` are the model's raw
    parameters (the same object `parser.resolve` returns); variables/distributions/tables as in `Engine.run`."""

    engine_vars = _engine_variables(variables)
    resolved = resolve_variables(parameters or {}, engine_vars, distributions, tables)
    by_role = psm_branches_by_role(node)
    cpy = node.cycles_per_year if node.cycles_per_year > 0 else 1.0
    cycles = int(node.cycles)

    def cycle_rates(cycle: int) -> tuple[dict[str, float], dict[str, float]]:
        cycle_params = {
            **resolved,
            "_cycle": Param(name="_cycle", base=float(cycle)),
            "_tenure": Param(name="_tenure", base=1.0),
        }
        cycle_vars = {**(engine_vars or {}), "_cycle": cycle, "_tenure": 1}
        cost, effect = {}, {}
        for role, branch in by_role.items():
            child = branch.child
            cost[role] = _branch_cost(branch, cycle_params, cycle_vars, distributions, tables) + _resolve(
                child.cost, cycle_params, cycle_vars, distributions, tables
            )
            effect[role] = _branch_effect(branch, cycle_params, cycle_vars, distributions, tables) + _resolve(
                child.effect, cycle_params, cycle_vars, distributions, tables
            )
        return cost, effect

    from .des_models import DES_TTE_PARAMS

    def resolved_curve_params(curve: Any) -> dict[str, float]:
        names = DES_TTE_PARAMS.get(curve.distribution) or ()
        if curve.distribution == "generalized_gamma" and "mu" in curve.params:
            names = ("mu", "sigma", "Q")
        return {n: float(_resolve(curve.params[n], resolved, engine_vars, distributions, tables)) for n in names}

    trace = PSMTrace(
        node_id=node.id, cycles=cycles, cycles_per_year=cpy, half_cycle_correction=bool(node.half_cycle_correction),
        cost_discount_rate=cost_discount_rate, effect_discount_rate=effect_discount_rate,
    )
    for cycle in range(cycles + 1):
        t_years = cycle / cpy
        s_pfs = _curve_survival(node.pfs_curve, t_years, resolved, engine_vars, distributions, tables)
        s_os = _curve_survival(node.os_curve, t_years, resolved, engine_vars, distributions, tables)
        occ = {"pre_progression": s_pfs, "post_progression": max(0.0, s_os - s_pfs), "dead": 1.0 - s_os}
        cost_rate, effect_rate = cycle_rates(cycle)
        raw_cost = sum(occ[r] * cost_rate[r] for r in PSM_STATE_ROLES)
        raw_effect = sum(occ[r] * effect_rate[r] for r in PSM_STATE_ROLES)
        row: dict[str, float] = {
            "cycle": cycle, "time_years": t_years, "s_pfs": s_pfs, "s_os": s_os,
            "pre_progression": occ["pre_progression"], "post_progression": occ["post_progression"], "dead": occ["dead"],
            "cost_undiscounted": raw_cost, "effect_undiscounted": raw_effect,
            "cost_discounted": raw_cost / (1.0 + cost_discount_rate) ** t_years,
            "effect_discounted": raw_effect / (1.0 + effect_discount_rate) ** t_years,
        }
        for role in PSM_STATE_ROLES:
            row[f"rate_cost_{role}"] = cost_rate[role]
            row[f"rate_effect_{role}"] = effect_rate[role]
        trace.rows.append(row)

    body = trace.rows[:cycles]
    last = trace.rows[cycles]
    first = trace.rows[0]

    def total(key: str) -> float:
        value = sum(r[key] for r in body)
        if trace.half_cycle_correction and cycles > 0:
            value += 0.5 * (last[key] - first[key])
        return value

    trace.total_cost = total("cost_discounted")
    trace.total_effect = total("effect_discounted")
    trace.total_cost_undiscounted = total("cost_undiscounted")
    trace.total_effect_undiscounted = total("effect_undiscounted")
    trace.curve_params = {"pfs": resolved_curve_params(node.pfs_curve), "os": resolved_curve_params(node.os_curve)}
    first_rates = {k: v for k, v in first.items() if k.startswith("rate_")}
    trace.rates_cycle_dependent = any(
        any(abs(row[k] - v) > 1e-12 for k, v in first_rates.items()) for row in trace.rows[1:]
    )
    return trace
