"""Technical Documentation Report for the Decision Tree editor: a Word
(.docx) file structured around the AdViSHE/CHEERS checklist, plus a
companion Excel (.xlsx) of the same tables for anyone who wants the raw
numbers -- following ``nma_report.py``'s own dual-file pattern (see
``build_report`` there).

Streamlit-free so it can be unit-tested, like ``nma_report.py``. Word is
written with the same raw-OOXML builder NMA uses (``report_docx.Docx``);
the .xlsx uses ``report_docx.write_xlsx``. The structure-diagram image is
best-effort -- see ``build_tree_report``'s ``diagram_png`` argument -- the
report is complete and readable without one.

This module intentionally recomputes nothing the app doesn't already have
in hand: every number here comes from ``strategy_totals`` (built from
``Engine.run()``'s own ``result.strategies``, the same values the app's
Analysis section and ``tree_excel_export.py``'s Strategy Summary sheet
show), so this report, the on-screen Analysis view, and the transparency
Excel file always agree for the same model. The one exception is the ICER
column itself, which is *computed* here from those totals using the exact
same baseline-relative convention as ``app.py``'s
``_render_tree_analysis_section`` and ``tree_excel_export.py``'s
``_write_icer_sheet`` (one chosen baseline, not a sorted efficient
frontier with extended dominance) -- duplicated in three places because
two of them must produce Excel formulas or live UI widgets rather than a
plain Python value, not because the convention itself is in question.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import pandas as pd

from heor_tree_engine import cma_cba
from heor_tree_engine.models import Param
from heor_tree_engine.tree_excel_export import _config_hash
from heor_tree_engine.tree_paths import PathRow
from report_docx import Docx, write_xlsx

SHEET_ORDER = [("parameters", "Parameters"), ("results", "Results"), ("cma_cba", "CMA-CBA")]


@dataclass
class TreeReportInput:
    """Everything build_tree_report() needs, in the shapes the app
    already has lying around after a Run -- nothing here is recomputed
    from the model, only formatted."""

    model_name: str
    document: dict[str, Any]
    parameters: Mapping[str, Param]
    paths: Sequence[PathRow]
    strategy_order: list[str]
    # {"Drug A": {"cost": 123.4, "effect": 1.65, "<extra_id>": ...}, ...}
    strategy_totals: dict[str, dict[str, float]]
    cost_symbol: str
    effect_symbol: str
    cost_decimals: int
    effect_decimals: int
    # {outcome_id: display_label} for every Cost/Effect outcome beyond the
    # primary pair -- empty for a model with no extra outcomes.
    extra_outcome_labels: dict[str, str]
    baseline_strategy: str | None
    maximize_effect: bool
    willingness_to_pay: float
    cost_discount_rate: float
    effect_discount_rate: float
    engine_version: str = ""
    # Raw PNG bytes of the canvas structure diagram, if the user clicked
    # the camera button this session (see handle_tree_canvas_event's
    # "diagram_png" branch in app.py) -- None is a normal, fully
    # supported state, not an error.
    diagram_png: bytes | None = None
    # Analysis settings (type CMA/CBA adds a section to the report)
    analysis_settings: dict[str, Any] | None = None


@dataclass
class TreeReportBundle:
    docx_bytes: bytes
    xlsx_bytes: bytes
    filename_stem: str


def _effective_baseline(inp: TreeReportInput) -> str:
    if inp.baseline_strategy in inp.strategy_totals:
        return inp.baseline_strategy
    return inp.strategy_order[0]


def _icer_rows(inp: TreeReportInput) -> list[dict[str, Any]]:
    """One row per strategy, baseline-relative -- see this module's
    docstring for why this duplicates (rather than imports) the same
    convention used elsewhere."""

    baseline = _effective_baseline(inp)
    base_cost = inp.strategy_totals[baseline]["cost"]
    base_effect = inp.strategy_totals[baseline]["effect"]
    direction = 1.0 if inp.maximize_effect else -1.0
    rows: list[dict[str, Any]] = []
    for strategy in inp.strategy_order:
        totals = inp.strategy_totals[strategy]
        cost = float(totals.get("cost", 0.0))
        effect = float(totals.get("effect", 0.0))
        inc_cost = cost - base_cost
        inc_effect = effect - base_effect
        if strategy == baseline:
            icer: float | None = None
            status = "-"
        elif inc_effect == 0 or (direction * inc_effect) <= 0:
            icer = None
            status = "Dominated"
        else:
            icer = inc_cost / inc_effect
            status = "Within WTP" if icer <= inp.willingness_to_pay else "Above WTP"
        rows.append({
            "strategy": strategy,
            "cost": cost,
            "effect": effect,
            "incremental_cost": inc_cost,
            "incremental_effect": inc_effect,
            "icer": icer,
            "status": status,
        })
    return rows


def _node_counts(document: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for node in document.get("nodes", []):
        kind = str(node.get("type", "unknown"))
        counts[kind] = counts.get(kind, 0) + 1
    return counts


def tree_methods_paragraphs(inp: TreeReportInput) -> list[str]:
    """Auto-generated Methods paragraphs describing the model's structure
    and analytic settings -- follows the same idea as ``nma_report.py``'s
    ``methods_paragraphs()`` (plain sentences built from the model's own
    data, not a template with blanks the author forgot to fill in)."""

    counts = _node_counts(inp.document)
    n_strategies = counts.get("decision", 0) and len(inp.strategy_order) or len(inp.strategy_order)
    n_chance = counts.get("chance", 0)
    n_terminal = counts.get("terminal", 0)
    n_psm = counts.get("psm", 0)
    n_des = counts.get("des", 0)
    n_paths = len(inp.paths)

    structure_bits = [f"{n_chance} chance node{'s' if n_chance != 1 else ''}",
                       f"{n_terminal} terminal node{'s' if n_terminal != 1 else ''}"]
    if n_psm:
        structure_bits.append(f"{n_psm} partitioned-survival node{'s' if n_psm != 1 else ''}")
    if n_des:
        structure_bits.append(f"{n_des} discrete-event-simulation node{'s' if n_des != 1 else ''}")

    paragraphs = [
        f"This report describes a Decision Tree model with {n_strategies} "
        f"strateg{'y' if n_strategies == 1 else 'ies'} ({', '.join(inp.strategy_order)}), "
        f"{', '.join(structure_bits)}, giving {n_paths} root-to-terminal path{'s' if n_paths != 1 else ''} "
        "in total.",
        f"Costs are discounted at {inp.cost_discount_rate * 100:.2f}% per year and effects at "
        f"{inp.effect_discount_rate * 100:.2f}% per year, applied to each terminal node's own payoff "
        "using its individual time horizon (year field); intermediate branch increments are not "
        "separately discounted.",
    ]
    baseline = _effective_baseline(inp)
    objective = "maximizing" if inp.maximize_effect else "minimizing"
    paragraphs.append(
        f"Cost-effectiveness is evaluated against a single baseline strategy ({baseline}), "
        f"{objective} the effect outcome, at a willingness-to-pay threshold of "
        f"{inp.cost_symbol} {inp.willingness_to_pay:,.0f} per unit of effect. Every other strategy's "
        "incremental cost and effect are computed relative to this baseline (not a sorted efficient "
        "frontier with extended dominance)."
    )
    if inp.extra_outcome_labels:
        labels = ", ".join(inp.extra_outcome_labels.values())
        paragraphs.append(f"In addition to cost and effect, this model tracks: {labels}.")
    return paragraphs


def _parameters_table(inp: TreeReportInput) -> tuple[pd.DataFrame, set[int]]:
    names = sorted(inp.parameters)
    rows = []
    missing_source_rows: set[int] = set()
    for i, name in enumerate(names):
        p = inp.parameters[name]
        if not p.source:
            missing_source_rows.add(i)
        rows.append({
            "Name": name,
            "Base": p.base,
            "Low": p.low if p.low is not None else "",
            "High": p.high if p.high is not None else "",
            "Distribution": p.dist or "",
            "Dist args": ", ".join(f"{k}={v}" for k, v in (p.dist_args or {}).items()),
            "Source": p.source or "",
        })
    return pd.DataFrame(rows), missing_source_rows


def _results_table(inp: TreeReportInput) -> pd.DataFrame:
    icer_rows = _icer_rows(inp)
    extra_ids = list(inp.extra_outcome_labels)
    rows = []
    for icer_row, strategy in zip(icer_rows, inp.strategy_order):
        totals = inp.strategy_totals[strategy]
        row = {
            "Strategy": strategy,
            f"Expected cost ({inp.cost_symbol})": round(icer_row["cost"], inp.cost_decimals),
            f"Expected effect ({inp.effect_symbol})": round(icer_row["effect"], inp.effect_decimals),
        }
        for outcome_id in extra_ids:
            row[inp.extra_outcome_labels[outcome_id]] = round(float(totals.get(outcome_id, 0.0)), inp.effect_decimals)
        row.update({
            "Incremental cost": round(icer_row["incremental_cost"], inp.cost_decimals),
            "Incremental effect": round(icer_row["incremental_effect"], inp.effect_decimals),
            "ICER": "-" if icer_row["icer"] is None else round(icer_row["icer"], inp.cost_decimals),
            "Status": icer_row["status"],
        })
        rows.append(row)
    return pd.DataFrame(rows)


def _write_tree_docx(inp: TreeReportInput, tables: dict[str, pd.DataFrame],
                      missing_source_rows: set[int], methods: list[str], config_hash: str) -> bytes:
    doc = Docx()
    title = inp.model_name or "DECIDE Decision Tree -- Technical Documentation Report"
    doc.paragraph(title, size=28)
    doc.caption(
        f"Exported {datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} "
        f"-- DECIDE engine version: {inp.engine_version or 'unknown'} -- "
        f"Model configuration hash: {config_hash}"
    )
    doc.paragraph(
        "This document was produced by the free edition of DECIDE, which computes every result below "
        "with deterministic Python code. No AI/LLM component was used to calculate the cost, effect or "
        "ICER figures in this report.",
        italic=True,
    )
    doc.page_break()

    doc.heading("1. Model structure", level=1)
    for paragraph in methods:
        doc.paragraph(paragraph)
    if inp.diagram_png:
        doc.image(inp.diagram_png)
        doc.caption("Decision tree structure, as shown on the canvas at export time.")
    else:
        doc.paragraph(
            "No structure diagram was captured this session. Click the camera button on the Tree "
            "canvas, then export this report again, to include one -- the rest of this report is "
            "complete without it.",
            italic=True,
        )

    doc.heading("2. Parameters", level=1)
    if missing_source_rows:
        doc.paragraph(
            f"{len(missing_source_rows)} of {len(tables['parameters'])} parameter(s) below have no "
            "Source citation on file (highlighted). Add one in the Tree Editor's Parameters table "
            "before relying on this report for a submission.",
            italic=True,
        )
    doc.table(tables["parameters"], highlight_rows=missing_source_rows)

    doc.heading("3. Results", level=1)
    doc.paragraph(
        "Expected cost and effect per strategy, and the incremental cost-effectiveness ratio (ICER) "
        "against the baseline strategy. These figures reconcile exactly with the Strategy Summary and "
        "ICER sheets of this model's transparency Excel export, and with the app's own Analysis view, "
        "since all three are built from the same Engine.run() results."
    )
    doc.table(tables["results"])
    cma_cba.write_report_section(doc, tables)

    doc.heading("4. Validation self-checks", level=1)
    doc.paragraph(
        "Automated validation self-checks (e.g. extreme-value and internal-consistency tests per "
        "AdViSHE) have not yet been run for this model. This section is a placeholder for that future "
        "feature.",
        italic=True,
    )

    doc.heading("5. Statement on AI use", level=1)
    doc.paragraph(
        "No AI/LLM component was used to compute the results in this report or in the underlying "
        "DECIDE model. All costs, effects and ICERs above were produced by deterministic Python code."
    )
    return doc.to_bytes()


def build_tree_report(inp: TreeReportInput) -> TreeReportBundle:
    parameters_df, missing_source_rows = _parameters_table(inp)
    tables = {"parameters": parameters_df, "results": _results_table(inp)}
    _sec = cma_cba.report_section(
        inp.analysis_settings, inp.strategy_order,
        [inp.strategy_totals[n]["cost"] for n in inp.strategy_order],
        [inp.strategy_totals[n]["effect"] for n in inp.strategy_order],
        inp.maximize_effect, _effective_baseline(inp),
        inp.cost_symbol, inp.cost_decimals, inp.effect_decimals,
    )
    if _sec is not None:
        tables["cma_cba"] = _sec
    methods = tree_methods_paragraphs(inp)
    config_hash = _config_hash(inp.parameters, inp.paths)

    docx_bytes = _write_tree_docx(inp, tables, missing_source_rows, methods, config_hash)
    title = inp.model_name or "DECIDE Decision Tree -- Technical Documentation Report"
    xlsx_bytes = write_xlsx(tables, title, methods, SHEET_ORDER)

    stem = "decide-tree-technical-report"
    if inp.model_name:
        import re

        slug = re.sub(r"[^a-z0-9]+", "-", inp.model_name.lower()).strip("-")
        if slug:
            stem += "-" + slug
    return TreeReportBundle(docx_bytes=docx_bytes, xlsx_bytes=xlsx_bytes, filename_stem=stem)
