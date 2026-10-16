"""Technical Documentation Report for the Partitioned Survival (PSM) editor: a Word (.docx) file organised around
the AdViSHE / CHEERS checklist plus a companion Excel (.xlsx) with the same tables and the per-cycle traces.

Mirrors ``tree_report.py`` (same ``Docx`` / ``write_xlsx`` plumbing, same results/ICER convention) and adds the
PSM-specific content: the PFS / OS curve table with the source of every curve (missing sources highlighted),
curve-fitting evidence for curves fitted inside DECIDE (AIC / BIC of every candidate distribution and the
maximum gap to the Kaplan-Meier curve), figures (survival curves with KM overlay, stacked state occupancy),
state-occupancy and time-in-state tables, and the automated validation self-checks.

Streamlit-free.  Every figure comes from ``psm_trace.compute_psm_trace`` (a replay of the engine's own PSM
arithmetic, verified against the engine by the self-checks) or from ``Engine.run`` totals handed in by the caller.
A failing error-severity self-check raises ``SelfCheckFailure`` and no report is returned.
"""

from __future__ import annotations

import datetime
import io
import re
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from heor_tree_engine import cma_cba
from heor_tree_engine.models import PSM_STATE_ROLES
from heor_tree_engine.psm_excel_export import ROLE_LABELS
from heor_tree_engine.psm_trace import PSMTrace, compute_psm_trace, find_psm_nodes
from heor_tree_engine.psm_validation_checks import run_psm_self_checks
from heor_tree_engine.selfcheck import raise_on_failure
from heor_tree_engine.tree_excel_export import _config_hash
from report_docx import Docx, selfchecks_frame, write_selfchecks_section, write_xlsx
import tree_report
from tree_report import TreeReportBundle, TreeReportInput, _parameters_table, _results_table

SHEET_ORDER = [
    ("parameters", "Parameters"), ("structure", "PSM nodes"), ("curves", "Survival curves"),
    ("fits", "Curve fitting"), ("occupancy", "State occupancy"), ("time", "Time in state"),
    ("trace", "Cycle trace"), ("results", "Results"), ("cma_cba", "CMA-CBA"), ("selfchecks", "Validation self-checks"),
]


@dataclass
class PSMReportInput(TreeReportInput):
    """TreeReportInput plus what is needed to replay and check the PSM nodes."""

    tree: Any = None  # root DecisionNode (parser.resolve)
    variables: dict[str, Any] | None = None
    distributions: dict[str, Any] | None = None
    tables: dict[str, Any] | None = None


def _node_docs(document: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(n.get("id")): n for n in document.get("nodes", [])}


def _strategies_by_node(inp: PSMReportInput) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for row in inp.paths:
        if row.leaf_type == "psm":
            out.setdefault(row.leaf_id, [])
            if row.strategy not in out[row.leaf_id]:
                out[row.leaf_id].append(row.strategy)
    return out


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _curve_params_text(curve: Any, resolved: dict[str, float]) -> str:
    parts = []
    for name, value in resolved.items():
        raw = curve.params.get(name)
        if isinstance(raw, str):
            parts.append(f"{name} = {_fmt(value)} ({raw.strip()})")
        else:
            parts.append(f"{name} = {_fmt(value)}")
    return "; ".join(parts)


def _build_tables(inp: PSMReportInput, nodes: list[Any], traces: dict[str, PSMTrace], checks: list[Any]):
    docs = _node_docs(inp.document)
    strategies = _strategies_by_node(inp)
    params_df, missing_params = _parameters_table(inp)

    structure_rows, curve_rows, fit_rows, occ_rows, time_rows, trace_rows = [], [], [], [], [], []
    missing_curve_rows: set[int] = set()
    for node in nodes:
        t = traces[node.id]
        label = ", ".join(strategies.get(node.id, [])) or node.id
        by_role = {b.role: b for b in node.branches}
        rewards = {}
        for role in PSM_STATE_ROLES:
            b = by_role[role]
            rewards[role] = (f"cost {_fmt(b.cost_increment if b.cost_increment is not None else 0.0)}"
                             f" + {_fmt(b.child.cost)}; effect {_fmt(b.effect_increment if b.effect_increment is not None else 0.0)}"
                             f" + {_fmt(b.child.effect)}")
        structure_rows.append({
            "PSM node": node.id, "Strategy": label, "Cycles": node.cycles, "Cycles per year": node.cycles_per_year,
            "Horizon (years)": round(node.cycles / (node.cycles_per_year or 1.0), 4),
            "Half-cycle correction": "Yes" if node.half_cycle_correction else "No",
            "Pre-progression reward / cycle": rewards["pre_progression"],
            "Post-progression reward / cycle": rewards["post_progression"],
            "Dead reward / cycle": rewards["dead"],
        })
        for key, curve, name in (("pfs", node.pfs_curve, "PFS"), ("os", node.os_curve, "OS")):
            raw_doc = docs.get(node.id, {}).get(f"{key}_curve", {}) or {}
            info = raw_doc.get("fit_info")
            source = info["summary"] if info else ""
            if not source:
                missing_curve_rows.add(len(curve_rows))
            curve_rows.append({
                "PSM node": node.id, "Curve": name, "Distribution": curve.distribution,
                "Parameters": _curve_params_text(curve, t.curve_params[key]),
                "Time unit": curve.time_unit,
                "Source": source or "Not documented in DECIDE (enter published/fitted source)",
            })
            if info:
                for cand in info.get("candidates", []):
                    fit_rows.append({
                        "PSM node": node.id, "Curve": name, "Candidate": cand["distribution"],
                        "AIC": round(cand["aic"], 2), "BIC": round(cand["bic"], 2),
                        "Max |S_fit - S_KM|": round(cand["max_gap"], 4),
                        "Selected": "Yes" if cand["distribution"] == curve.distribution else "",
                    })
        n = t.cycles
        for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
            row = t.rows[min(n, int(round(frac * n)))]
            occ_rows.append({
                "PSM node": node.id, "Cycle": row["cycle"], "Time (years)": round(row["time_years"], 4),
                "PFS": round(row["s_pfs"], 4), "OS": round(row["s_os"], 4),
                "Pre-progression": round(row["pre_progression"], 4),
                "Post-progression": round(row["post_progression"], 4), "Dead": round(row["dead"], 4),
            })
        trow = {"PSM node": node.id}
        total_years = 0.0
        for role in PSM_STATE_ROLES:
            val = sum(r[role] for r in t.rows[:n])
            if t.half_cycle_correction and n > 0:
                val += 0.5 * (t.rows[n][role] - t.rows[0][role])
            years = val / t.cycles_per_year
            trow[f"Expected time {ROLE_LABELS[role]} (years)"] = round(years, 4)
            total_years += years
        trow["Total (years)"] = round(total_years, 4)
        time_rows.append(trow)
        for r in t.rows:
            trace_rows.append({
                "PSM node": node.id, "Cycle": r["cycle"], "Time (years)": r["time_years"], "S_PFS": r["s_pfs"],
                "S_OS": r["s_os"], "Pre-progression": r["pre_progression"], "Post-progression": r["post_progression"],
                "Dead": r["dead"], "Cost (discounted)": r["cost_discounted"], "Effect (discounted)": r["effect_discounted"],
            })
    tables = {
        "parameters": params_df, "structure": pd.DataFrame(structure_rows), "curves": pd.DataFrame(curve_rows),
        "fits": pd.DataFrame(fit_rows), "occupancy": pd.DataFrame(occ_rows), "time": pd.DataFrame(time_rows),
        "trace": pd.DataFrame(trace_rows), "results": _results_table(inp), "selfchecks": selfchecks_frame(checks),
    }
    return tables, missing_params, missing_curve_rows


def _figures(inp: PSMReportInput, nodes: list[Any], traces: dict[str, PSMTrace]) -> list[tuple[str, bytes]]:
    """Survival curves (with the KM curve when a curve was fitted in DECIDE) and stacked occupancy, per PSM node.
    Best effort: returns [] when matplotlib is unavailable -- the tables carry the same numbers."""

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001
        return []
    docs = _node_docs(inp.document)
    out: list[tuple[str, bytes]] = []
    for node in nodes:
        t = traces[node.id]
        years = [r["time_years"] for r in t.rows]
        fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=150)
        ax.plot(years, [r["s_pfs"] for r in t.rows], color="#2563EB", lw=2, label="PFS (fitted curve)")
        ax.plot(years, [r["s_os"] for r in t.rows], color="#DC2626", lw=2, label="OS (fitted curve)")
        for key, color, nm in (("pfs", "#1E3A8A", "PFS"), ("os", "#7F1D1D", "OS")):
            info = (docs.get(node.id, {}).get(f"{key}_curve") or {}).get("fit_info")
            if info and info.get("km"):
                unit = PSM_UNITS.get(info.get("time_unit", "year"), 1.0)
                ax.step([p[0] / unit for p in info["km"]], [p[1] for p in info["km"]], where="post",
                        color=color, lw=1.2, ls="--", label=f"{nm} Kaplan-Meier (data)")
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("Time (years)")
        ax.set_ylabel("Survival")
        ax.set_title(f"Survival curves - node {node.id}", fontsize=10)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, loc="upper right")
        buf = io.BytesIO()
        fig.tight_layout()
        fig.savefig(buf, format="png")
        plt.close(fig)
        out.append((f"Survival curves, node {node.id}: PFS and OS as used by the model"
                    + (" (dashed: Kaplan-Meier data the curves were fitted to)" if any(
                        (docs.get(node.id, {}).get(f"{k}_curve") or {}).get("fit_info") for k in ("pfs", "os")) else "")
                    + ".", buf.getvalue()))

        fig, ax = plt.subplots(figsize=(7.2, 3.4), dpi=150)
        ax.stackplot(years, [r["pre_progression"] for r in t.rows], [r["post_progression"] for r in t.rows],
                     [r["dead"] for r in t.rows], colors=["#2563EB", "#F59E0B", "#9CA3AF"],
                     labels=["Pre-progression", "Post-progression", "Dead"])
        ax.set_ylim(0, 1)
        ax.set_xlim(0, years[-1] if years[-1] > 0 else 1)
        ax.set_xlabel("Time (years)")
        ax.set_ylabel("Proportion of cohort")
        ax.set_title(f"State occupancy - node {node.id}", fontsize=10)
        ax.legend(fontsize=7, loc="center right")
        buf = io.BytesIO()
        fig.tight_layout()
        fig.savefig(buf, format="png")
        plt.close(fig)
        out.append((f"State occupancy over time, node {node.id}.", buf.getvalue()))
    if len(nodes) > 1:  # all strategies on one chart: PFS solid, OS dashed
        strat = _strategies_by_node(inp)
        fig, ax = plt.subplots(figsize=(7.2, 3.8), dpi=150)
        palette = ["#2563EB", "#DC2626", "#059669", "#D97706", "#7C3AED", "#0891B2"]
        for i, node in enumerate(nodes):
            t = traces[node.id]
            years = [r["time_years"] for r in t.rows]
            label = " / ".join(strat.get(node.id) or [node.id])
            color = palette[i % len(palette)]
            ax.plot(years, [r["s_pfs"] for r in t.rows], color=color, lw=2, label=f"{label} - PFS")
            ax.plot(years, [r["s_os"] for r in t.rows], color=color, lw=2, ls="--", label=f"{label} - OS")
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("Time (years)")
        ax.set_ylabel("Survival")
        ax.set_title("PFS (solid) and OS (dashed) by strategy", fontsize=10)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, loc="upper right")
        buf = io.BytesIO()
        fig.tight_layout()
        fig.savefig(buf, format="png")
        plt.close(fig)
        out.append(("PFS (solid) and OS (dashed) of every strategy on one chart.", buf.getvalue()))
    return out


PSM_UNITS = {"year": 1.0, "month": 12.0, "week": 365.25 / 7.0, "day": 365.25}


def psm_methods_paragraphs(inp: PSMReportInput, nodes: list[Any]) -> list[str]:
    n_strat = len(inp.strategy_order)
    paragraphs = [
        f"This report describes a partitioned survival model (PSM) with {n_strat} strateg"
        f"{'y' if n_strat == 1 else 'ies'} ({', '.join(inp.strategy_order)}) built from {len(nodes)} PSM node"
        f"{'' if len(nodes) == 1 else 's'}. In each PSM node the proportion of the cohort in the pre-progression, "
        "post-progression and dead states at every cycle is derived from two survival curves: pre-progression = "
        "PFS(t), post-progression = OS(t) - PFS(t) (floored at 0), dead = 1 - OS(t).",
        "Costs and effects are accumulated per cycle as the occupancy of each state multiplied by that state's "
        "per-cycle cost or effect rate, discounted to the start of the model "
        f"({inp.cost_discount_rate * 100:.2f}% per year for costs, {inp.effect_discount_rate * 100:.2f}% per year "
        "for effects, using elapsed time in years), and summed over the cycles. A half-cycle correction, where "
        "switched on for a node, adds half of the difference between the end-of-horizon and first-cycle values.",
    ]
    baseline = inp.baseline_strategy if inp.baseline_strategy in inp.strategy_totals else inp.strategy_order[0]
    objective = "maximizing" if inp.maximize_effect else "minimizing"
    paragraphs.append(
        f"Cost-effectiveness is evaluated against a single baseline strategy ({baseline}), {objective} the effect "
        f"outcome, at a willingness-to-pay threshold of {inp.cost_symbol} {inp.willingness_to_pay:,.0f} per unit of "
        "effect (incremental values relative to the baseline, not an efficient frontier with extended dominance)."
    )
    return paragraphs


def _write_docx(inp: PSMReportInput, nodes, traces, tables, missing_params, missing_curves, checks,
                methods, figures, config_hash) -> bytes:
    doc = Docx()
    doc.paragraph(inp.model_name or "DECIDE Partitioned Survival Model -- Technical Documentation Report", size=28)
    doc.caption(
        f"Exported {datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} "
        f"-- DECIDE engine version: {inp.engine_version or 'unknown'} -- Model configuration hash: {config_hash}"
    )
    doc.paragraph(
        "This document was produced by the free edition of DECIDE, which computes every result below with "
        "deterministic Python code. No AI/LLM component was used to calculate the cost, effect or ICER figures "
        "in this report.", italic=True)
    doc.page_break()

    doc.heading("1. Model structure", level=1)
    for paragraph in methods:
        doc.paragraph(paragraph)
    doc.table(tables["structure"])
    if inp.diagram_png:
        doc.image(inp.diagram_png)
        doc.caption("Model structure as shown on the canvas at export time.")

    doc.heading("2. Parameters", level=1)
    if missing_params:
        doc.paragraph(
            f"{len(missing_params)} of {len(tables['parameters'])} parameter(s) have no Source citation on file "
            "(highlighted). Add one in the Parameters table before relying on this report for a submission.",
            italic=True)
    doc.table(tables["parameters"], highlight_rows=missing_params)

    doc.heading("3. Survival curves", level=1)
    doc.paragraph(
        "The PFS and OS curve of every PSM node, the time unit its parameters are expressed in, and where the "
        "parameters come from. Curves fitted to Kaplan-Meier or patient-level data inside DECIDE record their "
        "source automatically; any other curve has no source on file and is highlighted.")
    if missing_curves:
        doc.paragraph(f"{len(missing_curves)} of {len(tables['curves'])} curve(s) have no documented source.", italic=True)
    doc.table(tables["curves"], highlight_rows=missing_curves)
    if len(tables["fits"]):
        doc.heading("3.1 Curve fitting", level=2)
        doc.paragraph(
            "Candidate distributions fitted by maximum likelihood (right-censored) to the data supplied in DECIDE. "
            "AIC and BIC compare the candidates (lower is better); 'Max |S_fit - S_KM|' is the largest absolute "
            "difference between the fitted curve and the Kaplan-Meier estimate at the observed event times. "
            "Statistical fit is only one input to the choice of a survival model (clinical plausibility and "
            "extrapolation behaviour matter too).")
        doc.table(tables["fits"], highlight_rows={i for i, r in tables["fits"].iterrows() if r["Selected"] == "Yes"},
                  highlight_fill="DCFCE7")
    for caption, png in figures:
        doc.image(png)
        doc.caption(caption)

    doc.heading("4. State occupancy and time in state", level=1)
    doc.paragraph("Cohort distribution across the three states at selected points of each node's time horizon, and the "
                  "expected time spent in each state over the horizon (undiscounted, in years). The full "
                  "cycle-by-cycle trace is in the companion Excel file.")
    doc.table(tables["occupancy"])
    doc.table(tables["time"])

    doc.heading("5. Results", level=1)
    doc.paragraph(
        "Expected cost and effect per strategy and the incremental cost-effectiveness ratio (ICER) against the "
        "baseline strategy. These figures reconcile with the transparency Excel export and the app's own Analysis "
        "view, since all are built from the same Engine.run() results.")
    doc.table(tables["results"])
    cma_cba.write_report_section(doc, tables)

    write_selfchecks_section(doc, checks, "6. Validation self-checks")

    doc.heading("7. Statement on AI use", level=1)
    doc.paragraph(
        "No AI/LLM component was used to compute the results in this report or in the underlying DECIDE model. "
        "All costs, effects and ICERs above were produced by deterministic Python code.")
    return doc.to_bytes()


def build_psm_report(inp: PSMReportInput) -> TreeReportBundle:
    if inp.tree is None:
        raise ValueError("PSMReportInput.tree is required")
    nodes = [n for n in find_psm_nodes(inp.tree)]
    if not nodes:
        raise ValueError("This model has no PSM node")
    checks = run_psm_self_checks(
        inp.tree, inp.parameters, inp.paths, variables=inp.variables, distributions=inp.distributions,
        tables=inp.tables, cost_discount_rate=inp.cost_discount_rate, effect_discount_rate=inp.effect_discount_rate,
        extra_outcome_ids=list(inp.extra_outcome_labels),
    )
    raise_on_failure(checks)  # an engine-consistency failure: no report
    traces = {
        n.id: compute_psm_trace(n, inp.parameters, inp.variables, inp.distributions, inp.tables,
                                inp.cost_discount_rate, inp.effect_discount_rate)
        for n in nodes
    }
    tables, missing_params, missing_curves = _build_tables(inp, nodes, traces, checks)
    _sec = cma_cba.report_section(
        inp.analysis_settings, inp.strategy_order,
        [inp.strategy_totals[n]["cost"] for n in inp.strategy_order],
        [inp.strategy_totals[n]["effect"] for n in inp.strategy_order],
        inp.maximize_effect, tree_report._effective_baseline(inp),
        inp.cost_symbol, inp.cost_decimals, inp.effect_decimals,
    )
    if _sec is not None:
        tables["cma_cba"] = _sec
    methods = psm_methods_paragraphs(inp, nodes)
    config_hash = _config_hash(inp.parameters, inp.paths)
    figures = _figures(inp, nodes, traces)
    docx_bytes = _write_docx(inp, nodes, traces, tables, missing_params, missing_curves, checks, methods, figures,
                             config_hash)
    title = inp.model_name or "DECIDE Partitioned Survival Model -- Technical Documentation Report"
    xlsx_bytes = write_xlsx(tables, title, methods, SHEET_ORDER)
    stem = "decide-psm-technical-report"
    if inp.model_name:
        slug = re.sub(r"[^a-z0-9]+", "-", inp.model_name.lower()).strip("-")
        if slug:
            stem += "-" + slug
    return TreeReportBundle(docx_bytes=docx_bytes, xlsx_bytes=xlsx_bytes, filename_stem=stem)
