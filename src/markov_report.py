"""Technical Documentation Report for the Markov editor: a Word (.docx)
file structured around the AdViSHE/CHEERS checklist, plus a companion
Excel (.xlsx) of the same tables.

Deliberately mirrors ``tree_report.py`` line for line wherever the same
convention applies (same ``Docx``/``write_xlsx`` plumbing from
``report_docx.py``, same five-section AdViSHE/CHEERS layout, same
baseline-relative ICER computation) -- see that module's own docstring
for why the ICER convention is duplicated here rather than imported: two
of the three places it lives must produce Excel formulas or live UI
widgets, not a plain Python value, so all three keep their own copy
rather than share one that can't serve all three shapes.

Streamlit-free, unit-testable, like ``tree_report.py``. This module
recomputes nothing the app doesn't already have: every number in
``strategy_totals`` must come from that strategy's own ``run_markov()``
result (``result.total_cost``, ``result.total_effect``,
``result.extra_totals``) -- the exact same numbers that strategy's own
Markov transparency Excel export (``markov_excel_export.py``) self-checks
itself against before it will even return bytes -- so this report and
that Excel file always agree for a given strategy's own numbers. The one
value computed here is the ICER itself, from those totals, using the
same baseline-relative convention as ``tree_report.py`` and ``app.py``'s
own Analysis views (one chosen baseline strategy, not a sorted efficient
frontier with extended dominance).
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from typing import Any, Mapping

import pandas as pd

from heor_tree_engine import cma_cba
from heor_tree_engine.markov_excel_export import _config_hash
from heor_tree_engine.markov_models import MarkovModel
from heor_tree_engine.models import Param
from report_docx import Docx, write_xlsx

SHEET_ORDER = [("parameters", "Parameters"), ("results", "Results"), ("cma_cba", "CMA-CBA")]


@dataclass
class MarkovReportInput:
    """Everything build_markov_report() needs, in the shapes the app
    already has lying around after running every strategy -- nothing
    here is recomputed from the model, only formatted.

    ``strategy_totals`` is built by the caller by running ``run_markov()``
    once per strategy (mirroring ``_markov_merged_document(document,
    strategy)`` -> ``resolve_markov()`` -> ``run_markov()``, the same
    pattern ``_markov_preview_totals_cached()`` already uses) and reading
    off each result's own totals -- e.g. ``{"Drug A": {"cost":
    result.total_cost, "effect": result.total_effect, **result.extra_totals}}``.
    """

    model_name: str
    model: MarkovModel
    parameters: Mapping[str, Param]
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
    # the camera button this session (see handle_markov_canvas_event's
    # "diagram_png" branch in app.py, the same bridge tree_report.py
    # uses) -- None is a normal, fully supported state, not an error.
    diagram_png: bytes | None = None
    # Which strategy ``model`` above is the resolved structure OF -- unlike
    # Tree (where every strategy is a branch of one shared tree, so "the
    # model structure" is unambiguous), DECIDE's Markov editor lets each
    # strategy define its own independent states/transitions, so ``model``
    # can only ever be one specific strategy's own structure (by
    # convention, the currently active one -- see app.py's
    # _build_markov_report_for_document). Must be one of strategy_order's
    # entries. Defaults to strategy_order[0] when left blank.
    structure_strategy_label: str = ""
    # True when the caller found that this model's strategies don't all
    # share the same set of state ids -- when set,
    # markov_methods_paragraphs() adds an explicit caveat so the report
    # never silently implies every strategy shares the one structure
    # described (named via structure_strategy_label above).
    structure_varies_by_strategy: bool = False
    # Analysis settings (type CMA/CBA adds a section to the report)
    analysis_settings: dict[str, Any] | None = None


@dataclass
class MarkovReportBundle:
    docx_bytes: bytes
    xlsx_bytes: bytes
    filename_stem: str


def _effective_baseline(inp: MarkovReportInput) -> str:
    if inp.baseline_strategy in inp.strategy_totals:
        return inp.baseline_strategy
    return inp.strategy_order[0]


def _icer_rows(inp: MarkovReportInput) -> list[dict[str, Any]]:
    """One row per strategy, baseline-relative -- see this module's
    docstring for why this duplicates (rather than imports) the same
    convention used in tree_report.py and app.py."""

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


def markov_methods_paragraphs(inp: MarkovReportInput) -> list[str]:
    """Auto-generated Methods paragraphs describing the model's structure
    and analytic settings -- same idea as ``tree_methods_paragraphs()``
    (plain sentences built from the model's own data)."""

    model = inp.model
    n_strategies = len(inp.strategy_order)
    n_states = len(model.states)
    n_transitions = len(model.transitions)
    absorbing = [s.name for s in model.states if s.is_absorbing]
    tenure_states = [(s.name, s.tenure_count) for s in model.states if s.tenure_applied]

    half_cycle_bit = (
        "with half-cycle correction applied" if model.half_cycle_correction
        else "without half-cycle correction"
    )
    structure_label = inp.structure_strategy_label or inp.strategy_order[0]
    structure_scope = f"{structure_label}'s own" if inp.structure_varies_by_strategy else "the"
    paragraphs = [
        f"This report describes a cohort Markov model compared across {n_strategies} "
        f"strateg{'y' if n_strategies == 1 else 'ies'} ({', '.join(inp.strategy_order)}). "
        f"Structurally, {structure_scope} model has {n_states} health state{'s' if n_states != 1 else ''} "
        f"({', '.join(s.name for s in model.states)}) and {n_transitions} "
        f"transition{'s' if n_transitions != 1 else ''}, run over {model.cycles} "
        f"cycle{'s' if model.cycles != 1 else ''}, {half_cycle_bit}.",
    ]
    if inp.structure_varies_by_strategy:
        paragraphs.append(
            "Not every strategy compared in this report shares this same set of health states -- "
            "see each strategy's own Markov Editor canvas for its exact structure. Only Cost/Effect/"
            "ICER results (section 3 below) are compared across all strategies; the structure "
            f"description above describes {structure_label} specifically."
        )
    if absorbing:
        verb = "is an absorbing state" if len(absorbing) == 1 else "are absorbing states"
        paragraphs.append(
            f"{', '.join(absorbing)} {verb} (no further transitions occur once a cohort member "
            "enters)."
        )
    if tenure_states:
        bits = ", ".join(
            f"{name} ({count} tenure band{'s' if count != 1 else ''}, with separate "
            "initial/recurring/terminal cost)"
            for name, count in tenure_states
        )
        verb_form = "track" if len(tenure_states) != 1 else "tracks"
        paragraphs.append(
            f"The following state{'s' if len(tenure_states) != 1 else ''} {verb_form} time-since-entry "
            f"(tenure) rather than a single flat cost per cycle: {bits}."
        )
    paragraphs.append(
        f"Costs are discounted at {inp.cost_discount_rate * 100:.2f}% per cycle and effects at "
        f"{inp.effect_discount_rate * 100:.2f}% per cycle."
    )
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


def _parameters_table(inp: MarkovReportInput) -> tuple[pd.DataFrame, set[int]]:
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


def _results_table(inp: MarkovReportInput) -> pd.DataFrame:
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
            row[inp.extra_outcome_labels[outcome_id]] = round(
                float(totals.get(outcome_id, 0.0)), inp.effect_decimals
            )
        row.update({
            "Incremental cost": round(icer_row["incremental_cost"], inp.cost_decimals),
            "Incremental effect": round(icer_row["incremental_effect"], inp.effect_decimals),
            "ICER": "-" if icer_row["icer"] is None else round(icer_row["icer"], inp.cost_decimals),
            "Status": icer_row["status"],
        })
        rows.append(row)
    return pd.DataFrame(rows)


def _write_markov_docx(inp: MarkovReportInput, tables: dict[str, pd.DataFrame],
                        missing_source_rows: set[int], methods: list[str], config_hash: str) -> bytes:
    doc = Docx()
    title = inp.model_name or "DECIDE Markov Model -- Technical Documentation Report"
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
        doc.caption("Markov state-transition diagram, as shown on the canvas at export time.")
    else:
        doc.paragraph(
            "No structure diagram was captured this session. Click the camera button on the Markov "
            "canvas, then export this report again, to include one -- the rest of this report is "
            "complete without it.",
            italic=True,
        )

    doc.heading("2. Parameters", level=1)
    if missing_source_rows:
        doc.paragraph(
            f"{len(missing_source_rows)} of {len(tables['parameters'])} parameter(s) below have no "
            "Source citation on file (highlighted). Add one in the Markov Editor's Parameters table "
            "before relying on this report for a submission.",
            italic=True,
        )
    doc.table(tables["parameters"], highlight_rows=missing_source_rows)

    doc.heading("3. Results", level=1)
    doc.paragraph(
        "Expected cost and effect per strategy, and the incremental cost-effectiveness ratio (ICER) "
        "against the baseline strategy. Each strategy's cost/effect figures reconcile exactly with "
        "the Results sheet of that strategy's own Markov transparency Excel export, since both are "
        "built from the same run_markov() result."
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


def build_markov_report(inp: MarkovReportInput) -> MarkovReportBundle:
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
    methods = markov_methods_paragraphs(inp)
    config_hash = _config_hash(inp.parameters, inp.model)

    docx_bytes = _write_markov_docx(inp, tables, missing_source_rows, methods, config_hash)
    title = inp.model_name or "DECIDE Markov Model -- Technical Documentation Report"
    xlsx_bytes = write_xlsx(tables, title, methods, SHEET_ORDER)

    stem = "decide-markov-technical-report"
    if inp.model_name:
        import re

        slug = re.sub(r"[^a-z0-9]+", "-", inp.model_name.lower()).strip("-")
        if slug:
            stem += "-" + slug
    return MarkovReportBundle(docx_bytes=docx_bytes, xlsx_bytes=xlsx_bytes, filename_stem=stem)
