"""Validation self-checks for partitioned-survival (PSM) nodes (AdViSHE internal validity).

`run_psm_self_checks` is run when a PSM report is built, on the exact model being reported.  Each check summarises
the worst case across every PSM node in the tree.  Error-severity checks compare the engine with something
independent -- a closed form, a different implementation of S(t) (`psm_km_fit.survival`, vectorised numpy), or a
direct sum over the cycles -- so they catch engine bugs on the model at hand, not only on the test suite's models.
Warning-severity checks (crossing survival curves, unsupported extra outcomes) describe the model and never block.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .engine import _engine_variables, _evaluate_node
from .expressions import resolve_variables
from .models import PSM_CURVE_TIME_UNITS, PSM_STATE_ROLES, Param, PSMNode, TerminalNode
from .psm_km_fit import survival as independent_survival
from .psm_trace import PSMTrace, compute_psm_trace, find_psm_nodes
from .selfcheck import SelfCheckResult, values_close


def _independent_curve(node_curve: Any, params: Mapping[str, float], years: np.ndarray) -> np.ndarray:
    """S(t) from psm_km_fit's own implementation (not the engine's), in the curve's time unit."""

    t = years * PSM_CURVE_TIME_UNITS.get(node_curve.time_unit, 1.0)
    return independent_survival(node_curve.distribution, dict(params), t)


def _independent_total(
    node: PSMNode, trace: PSMTrace, cost_rate: np.ndarray | float, effect_rate: np.ndarray | float,
    disc_cost: float, disc_effect: float,
) -> tuple[float, float]:
    """Totals by a direct vectorised sum (own S(t), occupancy, discounting and half-cycle correction).
    `cost_rate`/`effect_rate` are either (3,) role vectors or (cycles+1, 3) per-cycle arrays [pre, post, dead]."""

    n = trace.cycles
    cpy = trace.cycles_per_year
    years = np.arange(n + 1, dtype=float) / cpy
    s_pfs = _independent_curve(node.pfs_curve, trace.curve_params["pfs"], years)
    s_os = _independent_curve(node.os_curve, trace.curve_params["os"], years)
    occ = np.column_stack([s_pfs, np.maximum(0.0, s_os - s_pfs), 1.0 - s_os])
    out = []
    for rate, d in ((cost_rate, disc_cost), (effect_rate, disc_effect)):
        rate = np.broadcast_to(np.asarray(rate, dtype=float), occ.shape)
        per = (occ * rate).sum(axis=1) / (1.0 + d) ** years
        total = float(per[:n].sum())
        if trace.half_cycle_correction and n > 0:
            total += 0.5 * float(per[n] - per[0])
        out.append(total)
    return out[0], out[1]


def _zeroed_copy(node: PSMNode, *, dead_only: bool = False, constant: float | None = None) -> PSMNode:
    """A copy of the node whose per-cycle rewards are replaced: all zero (default), dead-state-only cost/effect
    of 1, or `constant` in every state (cost and effect)."""

    branches = []
    for branch in node.branches:
        if constant is not None:
            value = float(constant)
        elif dead_only:
            value = 1.0 if branch.role == "dead" else 0.0
        else:
            value = 0.0
        child = dataclasses.replace(branch.child, cost=value, effect=value, extra_outcomes={}) \
            if isinstance(branch.child, TerminalNode) else branch.child
        branches.append(dataclasses.replace(branch, cost_increment=0.0, effect_increment=0.0, child=child,
                                            extra_outcomes={}))
    return dataclasses.replace(node, branches=branches)


def run_psm_self_checks(
    tree: Any,
    parameters: Mapping[str, Param],
    paths: Sequence[Any],
    *,
    variables: Mapping[str, Any] | None = None,
    distributions: Mapping[str, Mapping[str, Any]] | None = None,
    tables: Mapping[str, Mapping[str, Any]] | None = None,
    cost_discount_rate: float = 0.0,
    effect_discount_rate: float = 0.0,
    extra_outcome_ids: Sequence[str] = (),
) -> list[SelfCheckResult]:
    nodes = find_psm_nodes(tree)
    if not nodes:
        return [SelfCheckResult("Partitioned survival checks", True, "This model has no PSM node.", applicable=False)]

    engine_vars = _engine_variables(variables)
    resolved = resolve_variables(parameters or {}, engine_vars, distributions, tables)
    traces = {n.id: compute_psm_trace(n, parameters, variables, distributions, tables, cost_discount_rate,
                                      effect_discount_rate) for n in nodes}

    def engine_value(n: PSMNode, discounted: bool = True) -> tuple[float, float]:
        for row in paths:
            if row.leaf_id == n.id and row.leaf_type == "psm":
                step = row.steps[-1]
                src = step.discounted if discounted else step.undiscounted
                return float(src.get("cost", 0.0)), float(src.get("effect", 0.0))
        r = _evaluate_node(n, resolved, engine_vars, distributions, cost_discount_rate, effect_discount_rate, {}, tables)
        return r.cost, r.effect

    def run_engine(n: PSMNode, cr: float, er: float) -> tuple[float, float]:
        r = _evaluate_node(n, resolved, engine_vars, distributions, cr, er, {}, tables)
        return r.cost, r.effect

    checks: list[SelfCheckResult] = []
    label = lambda n: f"'{n.id}'"  # noqa: E731

    # 1. curves cross?  (model-quality warning)
    worst_cross = {n.id: max(r["s_pfs"] - r["s_os"] for r in traces[n.id].rows) for n in nodes}
    crossing = [nid for nid, v in worst_cross.items() if v > 1e-9]
    checks.append(SelfCheckResult(
        "PFS curve does not exceed OS curve",
        not crossing,
        ("Largest PFS - OS over all cycles: " + ", ".join(f"{nid}: {max(v, 0.0):.2e}" for nid, v in worst_cross.items())
         if not crossing else
         "PFS exceeds OS in node(s) " + ", ".join(crossing) + " (worst " + ", ".join(f"{k}: {worst_cross[k]:.3g}" for k in crossing)
         + "): the engine floors post-progression at 0, so these curves distort the occupancy; review the fitted curves"),
        severity="warning",
    ))

    # 2. S(0)=1 and non-increasing
    problems, worst_s0, worst_inc = [], 0.0, 0.0
    for n in nodes:
        for key in ("s_pfs", "s_os"):
            vals = [r[key] for r in traces[n.id].rows]
            worst_s0 = max(worst_s0, abs(vals[0] - 1.0))
            worst_inc = max(worst_inc, max((b - a) for a, b in zip(vals, vals[1:])) if len(vals) > 1 else 0.0)
            if abs(vals[0] - 1.0) > 1e-9 or any(b - a > 1e-9 for a, b in zip(vals, vals[1:])) or min(vals) < -1e-12 or max(vals) > 1 + 1e-12:
                problems.append(f"{label(n)} {key}")
    checks.append(SelfCheckResult(
        "Survival curves start at 1, stay within [0, 1] and never increase",
        not problems,
        f"|S(0) - 1| <= {worst_s0:.1e}; largest increase between cycles {max(worst_inc, 0.0):.1e}"
        if not problems else "Violated by: " + ", ".join(problems),
    ))

    # 3. occupancy sums to 1
    worst_sum = max(abs(r["pre_progression"] + r["post_progression"] + r["dead"] - 1.0)
                    for n in nodes for r in traces[n.id].rows)
    ok_sum = worst_sum <= 1e-9
    checks.append(SelfCheckResult(
        "State occupancy sums to 1 in every cycle",
        ok_sum,
        f"Largest |pre + post + dead - 1| over all cycles and nodes: {worst_sum:.1e}"
        + ("" if ok_sum or not crossing else " (caused by the crossing curves flagged above)"),
        severity="error" if not crossing else "warning",
    ))

    # 4. engine value == cycle-by-cycle replay
    diffs = []
    for n in nodes:
        ec, ee = engine_value(n)
        diffs.append((n.id, abs(traces[n.id].total_cost - ec), abs(traces[n.id].total_effect - ee),
                      values_close(traces[n.id].total_cost, ec) and values_close(traces[n.id].total_effect, ee)))
    checks.append(SelfCheckResult(
        "Cycle-by-cycle replay reproduces the engine's node value",
        all(d[3] for d in diffs),
        "Largest |difference| in discounted cost / effect: "
        f"{max(d[1] for d in diffs):.1e} / {max(d[2] for d in diffs):.1e}",
    ))

    # 5. independent re-computation of the totals
    cross = []
    for n in nodes:
        t = traces[n.id]
        cost_rate = np.array([[r[f"rate_cost_{k}"] for k in PSM_STATE_ROLES] for r in t.rows])
        effect_rate = np.array([[r[f"rate_effect_{k}"] for k in PSM_STATE_ROLES] for r in t.rows])
        ic, ie = _independent_total(n, t, cost_rate, effect_rate, cost_discount_rate, effect_discount_rate)
        ec, ee = engine_value(n)
        cross.append((n.id, abs(ic - ec), abs(ie - ee), values_close(ic, ec) and values_close(ie, ee)))
    checks.append(SelfCheckResult(
        "Independent recomputation (own survival functions and aggregation) reconciles with the engine",
        all(c[3] for c in cross),
        "Largest |difference| in discounted cost / effect: "
        f"{max(c[1] for c in cross):.1e} / {max(c[2] for c in cross):.1e}",
    ))

    # 6. zero rewards -> zero
    zeros = []
    for n in nodes:
        zc, ze = run_engine(_zeroed_copy(n), cost_discount_rate, effect_discount_rate)
        zeros.append((n.id, abs(zc), abs(ze)))
    checks.append(SelfCheckResult(
        "Extreme value: all costs and effects set to 0 give a total of 0",
        all(values_close(z[1], 0.0) and values_close(z[2], 0.0) for z in zeros),
        f"Largest |total| with zero rewards: {max(max(z[1], z[2]) for z in zeros):.1e}",
    ))

    # 7. reward of 1 only in the dead state, no discounting -> sum of (1 - OS)
    dead = []
    for n in nodes:
        t = traces[n.id]
        vc, ve = run_engine(_zeroed_copy(n, dead_only=True), 0.0, 0.0)
        ic, ie = _independent_total(n, t, np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 1.0]), 0.0, 0.0)
        # direct closed form: sum over cycles of (1 - S_OS), half-cycle correction included
        years = np.arange(t.cycles + 1, dtype=float) / t.cycles_per_year
        dead_curve = 1.0 - _independent_curve(n.os_curve, t.curve_params["os"], years)
        closed = float(dead_curve[: t.cycles].sum())
        if t.half_cycle_correction and t.cycles > 0:
            closed += 0.5 * float(dead_curve[t.cycles] - dead_curve[0])
        dead.append((n.id, abs(vc - closed), abs(ve - closed), abs(ic - closed)))
    checks.append(SelfCheckResult(
        "Extreme value: a reward of 1 per cycle in the Dead state only equals the sum of (1 - OS)",
        all(values_close(d[1] + 0.0, 0.0, abs_=1e-9, rel=0) and values_close(d[2], 0.0, abs_=1e-9, rel=0)
            and values_close(d[3], 0.0, abs_=1e-9, rel=0) for d in dead),
        f"Largest |engine - closed form|: {max(max(d[1], d[2], d[3]) for d in dead):.1e}",
    ))

    # 8. a reward of 1 in every state -> sum of discount factors (curve-independent)
    flat_ok, flat_detail, flat_applicable = [], [], True
    if crossing:
        flat_applicable = False
    else:
        for n in nodes:
            t = traces[n.id]
            vc, ve = run_engine(_zeroed_copy(n, constant=1.0), cost_discount_rate, effect_discount_rate)
            years = np.arange(t.cycles + 1, dtype=float) / t.cycles_per_year
            res = []
            for d in (cost_discount_rate, effect_discount_rate):
                disc = (1.0 + d) ** (-years)
                closed = float(disc[: t.cycles].sum())
                if t.half_cycle_correction and t.cycles > 0:
                    closed += 0.5 * float(disc[t.cycles] - disc[0])
                res.append(closed)
            flat_ok.append(values_close(vc, res[0]) and values_close(ve, res[1]))
            flat_detail.append(max(abs(vc - res[0]), abs(ve - res[1])))
    checks.append(SelfCheckResult(
        "Extreme value: a reward of 1 per cycle in every state equals the sum of discount factors",
        all(flat_ok) if flat_applicable else True,
        (f"Largest |engine - closed form|: {max(flat_detail):.1e}" if flat_applicable
         else "Not run: the survival curves cross (see above)"),
        applicable=flat_applicable,
    ))

    # 9. extra outcomes
    if extra_outcome_ids:
        checks.append(SelfCheckResult(
            "Extra outcomes",
            False,
            "This model defines additional Cost/Effect outcomes (" + ", ".join(extra_outcome_ids) + "); a PSM node "
            "currently contributes 0 to them, so their totals exclude the PSM payoffs",
            severity="warning",
        ))
    return checks
