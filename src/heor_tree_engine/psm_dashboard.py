"""Data behind the PSM dashboard cards and the State Report: PFS/OS curves and state occupancy of every PSM node,
labelled by the strategy it belongs to, plus the exact per-state split of the node's totals.

Everything is derived from `psm_trace.compute_psm_trace` (the single replay of the engine's PSM loop), so the
charts, the report and Run/ICER cannot disagree.
"""

from __future__ import annotations

from typing import Any

from .expressions import build_tables_mapping
from .models import PSM_STATE_ROLES
from .parser import resolve
from .psm_trace import PSMTrace, compute_psm_trace, find_psm_nodes

STATE_LABELS = {"pre_progression": "Pre-progression (PFS)", "post_progression": "Post-progression", "dead": "Dead"}


def strategy_by_psm_node(document: dict[str, Any]) -> dict[str, list[str]]:
    """{psm node id: [labels of the root-decision strategies that reach it]} (document node dicts)."""

    nodes = {str(n.get("id")): n for n in document.get("nodes", [])}
    root = nodes.get(str(document.get("root_id")))
    out: dict[str, list[str]] = {}
    if not root:
        return out

    def walk(node_id: str, label: str, seen: set[str]) -> None:
        node = nodes.get(node_id)
        if node is None or node_id in seen:
            return
        if node.get("type") == "psm":
            out.setdefault(node_id, [])
            if label not in out[node_id]:
                out[node_id].append(label)
            return
        for b in node.get("branches", []) or []:
            walk(str(b.get("child_id")), label, seen | {node_id})

    for b in root.get("branches", []) or []:
        walk(str(b.get("child_id")), str(b.get("label") or b.get("child_id")), {str(root.get("id"))})
    return out


def build_series(document: dict[str, Any]) -> list[dict[str, Any]]:
    """One entry per well-formed PSM node: {node_id, label, trace}. `label` is the strategy name (with the node id
    appended when a strategy reaches several PSM nodes, e.g. under a chance node). Nodes that cannot be replayed
    yet (incomplete / invalid) are skipped."""

    try:
        tree, parameters = resolve(document)
    except ValueError:
        return []
    distributions = {str(r["name"]): r for r in document.get("distributions", []) if r.get("name")}
    tables = build_tables_mapping(document.get("tables"))
    cost_rate = float(document.get("discount_cost_value", 0.0) or 0.0)
    effect_rate = float(document.get("discount_effect_value", 0.0) or 0.0)
    strategies = strategy_by_psm_node(document)
    per_strategy: dict[str, int] = {}
    for labels in strategies.values():
        for lab in labels:
            per_strategy[lab] = per_strategy.get(lab, 0) + 1
    series: list[dict[str, Any]] = []
    for node in find_psm_nodes(tree):
        try:
            trace = compute_psm_trace(node, parameters, None, distributions, tables, cost_rate, effect_rate)
        except Exception:  # noqa: BLE001 - a half-edited node simply has no chart yet
            continue
        labels = strategies.get(node.id) or [node.id]
        base = " / ".join(labels)
        multi = any(per_strategy.get(lab, 0) > 1 for lab in labels)
        series.append({"node_id": node.id, "label": f"{base} ({node.id})" if multi else base, "trace": trace})
    return series


def survival_overlay_records(series: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Long-format rows (Strategy, Curve, Years, Survival) for the PFS/OS overlay chart."""

    out: list[dict[str, Any]] = []
    for s in series:
        for r in s["trace"].rows:
            out.append({"Strategy": s["label"], "Curve": "PFS", "Years": r["time_years"], "Survival": r["s_pfs"]})
            out.append({"Strategy": s["label"], "Curve": "OS", "Years": r["time_years"], "Survival": r["s_os"]})
    return out


def occupancy_records(series: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Long-format rows (Strategy, State, Years, Proportion) for the stacked state-occupancy chart."""

    out: list[dict[str, Any]] = []
    for s in series:
        for r in s["trace"].rows:
            for role in PSM_STATE_ROLES:
                out.append({"Strategy": s["label"], "State": STATE_LABELS[role], "Years": r["time_years"],
                            "Proportion": r[role]})
    return out


def cycle_weights(trace: PSMTrace) -> list[float]:
    """Weight of each row 0..cycles in the node's totals: 1 for a body cycle; with the half-cycle correction the
    first and last points count half; the end-of-horizon point counts 0 without it."""

    n = trace.cycles
    w = [1.0] * (n + 1)
    w[n] = 0.0
    if trace.half_cycle_correction and n > 0:
        w[0] = 0.5
        w[n] = 0.5
    if n == 0:
        w[0] = 1.0
    return w


def state_breakdown(trace: PSMTrace) -> list[dict[str, Any]]:
    """Per-state split of the node's totals (area under the occupancy curve in cycles, cost and effect, each
    undiscounted and discounted), using the same weights as the engine, so the rows add up to the node totals."""

    w = cycle_weights(trace)
    out = []
    for role in PSM_STATE_ROLES:
        area = cost_u = cost_d = eff_u = eff_d = 0.0
        for wi, r in zip(w, trace.rows):
            t = r["time_years"]
            occ = r[role]
            c = occ * r[f"rate_cost_{role}"]
            e = occ * r[f"rate_effect_{role}"]
            area += wi * occ
            cost_u += wi * c
            eff_u += wi * e
            cost_d += wi * c / (1.0 + trace.cost_discount_rate) ** t
            eff_d += wi * e / (1.0 + trace.effect_discount_rate) ** t
        out.append({"role": role, "area": area, "cost_undiscounted": cost_u, "cost_discounted": cost_d,
                    "effect_undiscounted": eff_u, "effect_discounted": eff_d})
    return out
