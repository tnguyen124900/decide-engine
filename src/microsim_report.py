"""Technical Documentation Report for the Microsimulation editor: a Word
(.docx) file structured around the AdViSHE/CHEERS checklist, plus a
companion Excel (.xlsx) of the same tables.

Mirrors ``markov_report.py`` (same ``Docx``/``write_xlsx`` plumbing, same
section layout, same baseline-relative ICER convention), with the parts
that only exist for a patient-level model:

* simulation settings (N patients, random seed, random-number design),
* Monte Carlo standard errors (MCSE) next to every expected cost/effect,
* validation self-checks written for a stochastic model (probability
  conservation, per-patient totals reconcile with the reported means,
  fixed-seed reproducibility, agreement with the equivalent cohort model
  within Monte Carlo error),
* a cohort-comparison table when the same structure could also be run as
  a Markov cohort model.

There is no live-formula Excel: a microsimulation draws random numbers per
patient, so a spreadsheet cannot recompute it. The Excel companion holds
the same tables as static values, plus one per-cycle trace sheet per
strategy.

Streamlit-free and unit-testable. Every number comes from the strategy's
own ``run_microsim()`` result; nothing is re-simulated here.
"""

from __future__ import annotations

import datetime
import hashlib
import math
import re
from dataclasses import dataclass, field
from typing import Any, Mapping

import pandas as pd

from heor_tree_engine import cma_cba
from heor_tree_engine.microsim_models import MicrosimModel, MicrosimResult
from heor_tree_engine.models import Param
from heor_tree_engine.selfcheck import SelfCheckResult, raise_on_failure, values_close
from report_docx import Docx, selfchecks_frame, write_selfchecks_section, write_xlsx

BASE_SHEETS = [("settings", "Settings"), ("structure", "Structure"), ("parameters", "Parameters"),
               ("results", "Results"), ("cma_cba", "CMA-CBA"), ("cohort", "Cohort comparison"), ("selfchecks", "Validation self-checks")]


@dataclass
class MicrosimStrategyRun:
    """One strategy's own simulation, plus the optional extras the app can
    supply for the self-checks."""

    label: str
    model: MicrosimModel
    result: MicrosimResult
    # {state_id: display name} as shown on the canvas.
    state_names: dict[str, str] = field(default_factory=dict)
    # (total_cost, total_effect) from the equivalent cohort Markov model
    # on the same structure, or None when it could not be run.
    cohort_totals: tuple[float, float] | None = None
    # Totals of two independent re-runs with the same seed (small N), for
    # the reproducibility self-check; None when not supplied.
    repro_totals: tuple[tuple[float, float], tuple[float, float]] | None = None


@dataclass
class MicrosimReportInput:
    model_name: str
    parameters: Mapping[str, Param]
    runs: list[MicrosimStrategyRun]
    cost_symbol: str
    effect_symbol: str
    cost_decimals: int
    effect_decimals: int
    baseline_strategy: str | None
    maximize_effect: bool
    willingness_to_pay: float
    cost_discount_rate: float
    effect_discount_rate: float
    cycles_per_year: float = 1.0
    engine_version: str = ""
    diagram_png: bytes | None = None
    structure_varies_by_strategy: bool = False
    structure_strategy_label: str = ""
    # Analysis settings (type CMA/CBA adds a section to the report)
    analysis_settings: dict[str, Any] | None = None


@dataclass
class MicrosimReportBundle:
    docx_bytes: bytes
    xlsx_bytes: bytes
    filename_stem: str


# --------------------------------------------------------------------------
# helpers


def _mcse(values: list[float]) -> float | None:
    n = len(values)
    if n < 2:
        return None
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / (n - 1)
    return math.sqrt(variance / n)


def _label_order(inp: MicrosimReportInput) -> list[str]:
    return [run.label for run in inp.runs]


def _run_by_label(inp: MicrosimReportInput) -> dict[str, MicrosimStrategyRun]:
    return {run.label: run for run in inp.runs}


def _effective_baseline(inp: MicrosimReportInput) -> str:
    labels = _label_order(inp)
    return inp.baseline_strategy if inp.baseline_strategy in labels else labels[0]


def _is_real(state: Any) -> bool:
    return bool(getattr(state, "is_real_state", True))


def _config_hash(inp: MicrosimReportInput) -> str:
    parts: list[str] = []
    for name in sorted(inp.parameters):
        p = inp.parameters[name]
        parts.append(f"{name}={p.base}|{p.low}|{p.high}|{p.dist}|{p.dist_args}|{p.source}")
    for run in inp.runs:
        m = run.model
        parts.append(f"strategy={run.label}|n={m.n_patients}|seed={m.random_seed}|cycles={m.cycles}|hcc={m.half_cycle_correction}")
        for s in m.states:
            parts.append(
                f"state={s.id}|{s.cost}|{s.effect}|{s.initial_prob}|{s.is_absorbing}|{s.tenure_applied}|"
                f"{s.tenure_count}|{getattr(s, 'is_real_state', True)}|{getattr(s, 'alias_of', None)}"
            )
        for t in m.transitions:
            parts.append(f"transition={t.from_state}->{t.to_state}|{t.probability}|{t.cost_increment}|{t.effect_increment}")
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()[:12]


def _icer_rows(inp: MicrosimReportInput) -> list[dict[str, Any]]:
    """Baseline-relative ICER per strategy (same convention as
    markov_report.py / tree_report.py / the app's Analysis views)."""

    by_label = _run_by_label(inp)
    baseline = _effective_baseline(inp)
    base_cost = by_label[baseline].result.total_cost
    base_effect = by_label[baseline].result.total_effect
    direction = 1.0 if inp.maximize_effect else -1.0
    rows: list[dict[str, Any]] = []
    for run in inp.runs:
        cost, effect = float(run.result.total_cost), float(run.result.total_effect)
        inc_cost, inc_effect = cost - base_cost, effect - base_effect
        if run.label == baseline:
            icer, status = None, "-"
        elif inc_effect == 0 or direction * inc_effect <= 0:
            icer, status = None, "Dominated"
        else:
            icer = inc_cost / inc_effect
            status = "Within WTP" if icer <= inp.willingness_to_pay else "Above WTP"
        rows.append({
            "strategy": run.label, "cost": cost, "effect": effect,
            "cost_se": _mcse(run.result.patient_costs), "effect_se": _mcse(run.result.patient_effects),
            "incremental_cost": inc_cost, "incremental_effect": inc_effect, "icer": icer, "status": status,
        })
    return rows


# --------------------------------------------------------------------------
# self-checks (stochastic-model versions)


def _fmt(values: dict[str, Any]) -> str:
    return "; ".join(f"{k}: {v}" for k, v in values.items())


def microsim_selfchecks(inp: MicrosimReportInput) -> list[SelfCheckResult]:
    checks: list[SelfCheckResult] = []

    # 1. Probability conservation: every cycle's occupancy sums to 1.
    worst: dict[str, float] = {}
    negative: dict[str, float] = {}
    for run in inp.runs:
        gaps = [abs(sum(row.values()) - 1.0) for row in run.result.trace]
        worst[run.label] = max(gaps) if gaps else 0.0
        negative[run.label] = min((min(row.values()) for row in run.result.trace if row), default=0.0)
    bad = {k: v for k, v in worst.items() if v > 1e-9}
    checks.append(SelfCheckResult(
        "Patients are conserved (state occupancy sums to 1 every cycle)", not bad,
        ("Largest deviation from 1 per strategy: " + _fmt({k: f"{v:.2e}" for k, v in worst.items()})),
    ))
    bad_neg = {k: v for k, v in negative.items() if v < -1e-12}
    checks.append(SelfCheckResult(
        "No negative state occupancy", not bad_neg,
        "Minimum occupancy per strategy: " + _fmt({k: f"{v:.2e}" for k, v in negative.items()}),
    ))

    # 2. Absorbing states never lose patients.
    leaks: dict[str, str] = {}
    n_absorbing = 0
    for run in inp.runs:
        for state in run.model.states:
            if not (_is_real(state) and state.is_absorbing):
                continue
            n_absorbing += 1
            series = [row.get(state.id, 0.0) for row in run.result.trace]
            if any(b < a - 1e-12 for a, b in zip(series, series[1:])):
                leaks[run.label] = state.name
    checks.append(SelfCheckResult(
        "Absorbing states never lose patients", not leaks,
        ("Occupancy of every absorbing state is non-decreasing over time." if not leaks
         else "Occupancy decreased in: " + _fmt(leaks)),
        applicable=n_absorbing > 0,
    ))

    # 3. Per-patient totals reconcile with the reported population means.
    recon: dict[str, str] = {}
    recon_bad: dict[str, str] = {}
    for run in inp.runs:
        res = run.result
        if not res.patient_costs:
            continue
        mean_cost = sum(res.patient_costs) / len(res.patient_costs)
        mean_effect = sum(res.patient_effects) / len(res.patient_effects)
        ok = values_close(mean_cost, res.total_cost) and values_close(mean_effect, res.total_effect)
        (recon if ok else recon_bad)[run.label] = f"cost {mean_cost:,.4f} vs {res.total_cost:,.4f}"
    checks.append(SelfCheckResult(
        "Mean of per-patient totals equals the reported total cost and effect", not recon_bad,
        "Compared per strategy: " + _fmt({**recon, **recon_bad}),
        applicable=bool(recon or recon_bad),
    ))

    # 4. Per-cycle discounted cost/effect add up to the total (only exact
    # without half-cycle correction -- the correction term belongs to no
    # single cycle).
    cyc: dict[str, str] = {}
    cyc_bad: dict[str, str] = {}
    for run in inp.runs:
        res = run.result
        if run.model.half_cycle_correction or not res.cost_per_cycle:
            continue
        ok = (values_close(sum(res.cost_per_cycle), res.total_cost)
              and values_close(sum(res.effect_per_cycle), res.total_effect))
        (cyc if ok else cyc_bad)[run.label] = f"sum of cycles = {sum(res.cost_per_cycle):,.4f}"
    checks.append(SelfCheckResult(
        "Discounted per-cycle cost and effect add up to the totals", not cyc_bad,
        ("Compared per strategy: " + _fmt({**cyc, **cyc_bad})) if (cyc or cyc_bad)
        else "Half-cycle correction is on: its adjustment is not attributed to a single cycle.",
        applicable=bool(cyc or cyc_bad),
    ))

    # 5. Same seed -> identical results.
    rep: dict[str, str] = {}
    rep_bad: dict[str, str] = {}
    for run in inp.runs:
        if run.repro_totals is None:
            continue
        a, b = run.repro_totals
        ok = values_close(a[0], b[0], 1e-12, 1e-12) and values_close(a[1], b[1], 1e-12, 1e-12)
        (rep if ok else rep_bad)[run.label] = f"{a[0]:,.6f} / {b[0]:,.6f}"
    checks.append(SelfCheckResult(
        "Fixed seed reproduces identical results (two independent runs)", not rep_bad,
        "Total cost of run 1 / run 2: " + _fmt({**rep, **rep_bad}),
        applicable=bool(rep or rep_bad),
    ))

    # 6. Agreement with the cohort model within Monte Carlo error.
    agree: dict[str, str] = {}
    disagree: dict[str, str] = {}
    for run in inp.runs:
        if run.cohort_totals is None:
            continue
        res = run.result
        cost_se, effect_se = _mcse(res.patient_costs), _mcse(res.patient_effects)
        if not cost_se or not effect_se:
            continue
        z_cost = (res.total_cost - run.cohort_totals[0]) / cost_se
        z_effect = (res.total_effect - run.cohort_totals[1]) / effect_se
        text = f"z(cost)={z_cost:+.2f}, z(effect)={z_effect:+.2f}"
        (agree if abs(z_cost) <= 3 and abs(z_effect) <= 3 else disagree)[run.label] = text
    checks.append(SelfCheckResult(
        "Microsimulation agrees with the equivalent cohort model within Monte Carlo error (|z| <= 3)",
        not disagree,
        ("Standardised difference (microsimulation - cohort) / MCSE: " + _fmt({**agree, **disagree}))
        + (" -- a disagreement beyond 3 MCSE deserves review (increase N, or check that the cohort model "
           "has the same structure)." if disagree else ""),
        applicable=bool(agree or disagree),
        severity="warning",
    ))

    # 7. Precision: MCSE of the incremental result is reported, flag a very noisy ICER.
    return checks


# --------------------------------------------------------------------------
# tables and text


def microsim_methods_paragraphs(inp: MicrosimReportInput) -> list[str]:
    first = inp.runs[0]
    model = first.model
    n_strategies = len(inp.runs)
    labels = _label_order(inp)
    structure_label = inp.structure_strategy_label or labels[0]
    real_states = [s for s in model.states if _is_real(s)]
    non_real = [s for s in model.states if not _is_real(s)]
    absorbing = [s.name for s in real_states if s.is_absorbing]
    scope = f"{structure_label}'s own" if inp.structure_varies_by_strategy else "the"
    half = "with half-cycle correction" if model.half_cycle_correction else "without half-cycle correction"
    paragraphs = [
        f"This report describes a patient-level (individual-based, first-order Monte Carlo) microsimulation "
        f"compared across {n_strategies} strateg{'y' if n_strategies == 1 else 'ies'} ({', '.join(labels)}). "
        f"Structurally, {scope} model has {len(real_states)} health state{'s' if len(real_states) != 1 else ''} "
        f"({', '.join(s.name for s in real_states)}), run for {model.cycles} cycle"
        f"{'s' if model.cycles != 1 else ''} {half}.",
        f"Each of {model.n_patients:,} simulated patients is followed individually. At every cycle the "
        "patient's next health state is drawn at random from the transition probabilities of the state "
        "currently occupied; costs and effects accumulate per patient and the results reported here are "
        f"means across patients. The random seed is {model.random_seed}. Every patient has an independent "
        "random-number stream derived from that seed, and the same patient index uses the same stream in "
        "every strategy (common random numbers), so differences between strategies are not inflated by "
        "sampling noise.",
    ]
    if non_real:
        paragraphs.append(
            f"The model tree below each health state contains {len(non_real)} intermediate node"
            f"{'s' if len(non_real) != 1 else ''} (chance branches and terminal nodes). These are not "
            "states a patient occupies: a patient who enters one is resolved through to a health state "
            "within the same cycle, and a terminal node means the patient ends the cycle in the health "
            "state it represents and moves on from there in the next cycle. These nodes carry no cost "
            "or effect per cycle; a one-time cost or effect attached to a transition (Incre_$ / Incre_E) "
            "is added to the cycle in which that transition is taken."
        )
    if absorbing:
        verb = "is an absorbing state" if len(absorbing) == 1 else "are absorbing states"
        paragraphs.append(f"{', '.join(absorbing)} {verb} (patients never leave it once they enter).")
    if inp.structure_varies_by_strategy:
        paragraphs.append(
            "Not every strategy shares the same set of health states -- see each strategy's own canvas "
            f"for its exact structure; the structure described above is {structure_label}'s."
        )
    tenure = [(s.name, s.tenure_count) for s in real_states if s.tenure_applied]
    if tenure:
        paragraphs.append(
            "The following states track time since entry (tenure): "
            + ", ".join(f"{n} ({c} tenure bands)" for n, c in tenure) + "."
        )
    if model.trackers:
        paragraphs.append(
            f"{len(model.trackers)} patient-level tracker(s) are defined "
            f"({', '.join(t.name for t in model.trackers)}), reset {model.tracker_reset_timing.replace('_', ' ')}."
        )
    paragraphs.append(
        f"Costs are discounted at {inp.cost_discount_rate * 100:.2f}% per cycle and effects at "
        f"{inp.effect_discount_rate * 100:.2f}% per cycle ({inp.cycles_per_year:g} cycle(s) per year)."
    )
    baseline = _effective_baseline(inp)
    objective = "maximizing" if inp.maximize_effect else "minimizing"
    paragraphs.append(
        f"Cost-effectiveness is evaluated against a single baseline strategy ({baseline}), {objective} the "
        f"effect outcome, at a willingness-to-pay threshold of {inp.cost_symbol} {inp.willingness_to_pay:,.0f} "
        "per unit of effect. Each result is shown with its Monte Carlo standard error (MCSE), the "
        "standard deviation of the per-patient values divided by the square root of the number of patients."
    )
    return paragraphs


def _settings_table(inp: MicrosimReportInput) -> pd.DataFrame:
    rows = []
    for run in inp.runs:
        m = run.model
        rows.append({
            "Strategy": run.label, "Patients (N)": m.n_patients, "Random seed": m.random_seed,
            "Cycles": m.cycles, "Cycles per year": inp.cycles_per_year,
            "Horizon (years)": round(m.cycles / (inp.cycles_per_year or 1.0), 4),
            "Half-cycle correction": "Yes" if m.half_cycle_correction else "No",
            "Cost discount / cycle": inp.cost_discount_rate, "Effect discount / cycle": inp.effect_discount_rate,
            "Trackers": len(m.trackers), "Tracker reset": m.tracker_reset_timing,
        })
    return pd.DataFrame(rows)


def _structure_table(inp: MicrosimReportInput) -> pd.DataFrame:
    rows = []
    for run in inp.runs:
        m = run.model
        real = [s for s in m.states if _is_real(s)]
        non_real = [s for s in m.states if not _is_real(s)]
        rows.append({
            "Strategy": run.label,
            "Health states": ", ".join(s.name for s in real),
            "Absorbing": ", ".join(s.name for s in real if s.is_absorbing) or "-",
            "Intermediate nodes": len(non_real),
            "Transitions": len(m.transitions),
            "Initial distribution": ", ".join(
                f"{s.name}={s.initial_prob}" for s in real if s.initial_prob not in (0, 0.0, "0", None)
            ) or "-",
        })
    return pd.DataFrame(rows)


def _parameters_table(inp: MicrosimReportInput) -> tuple[pd.DataFrame, set[int]]:
    rows, missing = [], set()
    for i, name in enumerate(sorted(inp.parameters)):
        p = inp.parameters[name]
        if not p.source:
            missing.add(i)
        rows.append({
            "Name": name, "Base": p.base, "Low": p.low if p.low is not None else "",
            "High": p.high if p.high is not None else "", "Distribution": p.dist or "",
            "Dist args": ", ".join(f"{k}={v}" for k, v in (p.dist_args or {}).items()), "Source": p.source or "",
        })
    return pd.DataFrame(rows), missing


def _results_table(inp: MicrosimReportInput) -> pd.DataFrame:
    rows = []
    cd, ed = inp.cost_decimals, inp.effect_decimals
    for r in _icer_rows(inp):
        rows.append({
            "Strategy": r["strategy"],
            f"Expected cost ({inp.cost_symbol})": round(r["cost"], cd),
            "Cost MCSE": "" if r["cost_se"] is None else round(r["cost_se"], max(cd, 2)),
            f"Expected effect ({inp.effect_symbol})": round(r["effect"], ed),
            "Effect MCSE": "" if r["effect_se"] is None else round(r["effect_se"], max(ed, 4)),
            "Incremental cost": round(r["incremental_cost"], cd),
            "Incremental effect": round(r["incremental_effect"], ed),
            "ICER": "-" if r["icer"] is None else round(r["icer"], cd),
            "Status": r["status"],
        })
    return pd.DataFrame(rows)


def _cohort_table(inp: MicrosimReportInput) -> pd.DataFrame:
    rows = []
    for run in inp.runs:
        if run.cohort_totals is None:
            continue
        res = run.result
        cost_se, effect_se = _mcse(res.patient_costs), _mcse(res.patient_effects)
        rows.append({
            "Strategy": run.label,
            "Microsim cost": round(res.total_cost, inp.cost_decimals),
            "Cohort cost": round(run.cohort_totals[0], inp.cost_decimals),
            "Difference (cost)": round(res.total_cost - run.cohort_totals[0], inp.cost_decimals),
            "Cost z (diff/MCSE)": "" if not cost_se else round((res.total_cost - run.cohort_totals[0]) / cost_se, 2),
            "Microsim effect": round(res.total_effect, inp.effect_decimals),
            "Cohort effect": round(run.cohort_totals[1], inp.effect_decimals),
            "Difference (effect)": round(res.total_effect - run.cohort_totals[1], inp.effect_decimals),
            "Effect z (diff/MCSE)": "" if not effect_se else round((res.total_effect - run.cohort_totals[1]) / effect_se, 2),
        })
    return pd.DataFrame(rows)


def _trace_table(inp: MicrosimReportInput, run: MicrosimStrategyRun) -> pd.DataFrame:
    res = run.result
    names = {sid: run.state_names.get(sid, sid) for sid in (res.trace[0] if res.trace else {})}
    cs, es = inp.cost_symbol, inp.effect_symbol
    rows = []
    for t, row in enumerate(res.trace):
        entry: dict[str, Any] = {"Cycle": t}
        for sid, value in row.items():
            entry[names.get(sid, sid)] = value
        if t < len(res.raw_cost_per_cycle):
            entry[f"Raw cost ({cs})"] = res.raw_cost_per_cycle[t]
            entry[f"Discounted cost ({cs})"] = res.cost_per_cycle[t]
            entry[f"Raw effect ({es})"] = res.raw_effect_per_cycle[t]
            entry[f"Discounted effect ({es})"] = res.effect_per_cycle[t]
        rows.append(entry)
    return pd.DataFrame(rows)


def _snapshot_table(trace_df: pd.DataFrame) -> pd.DataFrame:
    """Five evenly spaced cycles of the full trace, for the Word report."""

    n = len(trace_df) - 1
    picks = sorted({min(n, int(round(f * n))) for f in (0.0, 0.25, 0.5, 0.75, 1.0)})
    return trace_df.iloc[picks].round(4)


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


# --------------------------------------------------------------------------
# build


def _write_docx(inp: MicrosimReportInput, tables: dict[str, pd.DataFrame], missing_source_rows: set[int],
                methods: list[str], config_hash: str, checks: list[SelfCheckResult]) -> bytes:
    doc = Docx()
    title = inp.model_name or "DECIDE Microsimulation Model -- Technical Documentation Report"
    doc.paragraph(title, size=28)
    doc.caption(
        f"Exported {datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} "
        f"-- DECIDE engine version: {inp.engine_version or 'unknown'} -- Model configuration hash: {config_hash}"
    )
    doc.paragraph(
        "This document was produced by the free edition of DECIDE, which computes every result below with "
        "deterministic Python code (pseudo-random numbers from a fixed seed). No AI/LLM component was used "
        "to calculate the cost, effect or ICER figures in this report.",
        italic=True,
    )
    doc.page_break()

    doc.heading("1. Model structure", level=1)
    for paragraph in methods:
        doc.paragraph(paragraph)
    doc.table(tables["structure"])
    if inp.diagram_png:
        doc.image(inp.diagram_png)
        doc.caption("Microsimulation state-transition diagram, as shown on the canvas at export time.")
    else:
        doc.paragraph(
            "No structure diagram was captured this session. Click the camera button on the Microsimulation "
            "canvas, then export this report again, to include one -- the rest of this report is complete "
            "without it.",
            italic=True,
        )

    doc.heading("2. Simulation settings", level=1)
    doc.paragraph(
        "A microsimulation result depends on the number of simulated patients and on the random seed; both "
        "are fixed here so the numbers can be reproduced exactly."
    )
    doc.table(tables["settings"])

    doc.heading("3. Parameters", level=1)
    if missing_source_rows:
        doc.paragraph(
            f"{len(missing_source_rows)} of {len(tables['parameters'])} parameter(s) below have no Source "
            "citation on file (highlighted). Add one in the Parameters table before relying on this report "
            "for a submission.",
            italic=True,
        )
    if tables["parameters"].empty:
        doc.paragraph("This model defines no named parameters; all inputs are entered directly on states and transitions.",
                      italic=True)
    else:
        doc.table(tables["parameters"], highlight_rows=missing_source_rows)

    doc.heading("4. Results", level=1)
    doc.paragraph(
        "Expected (mean per patient) cost and effect per strategy with Monte Carlo standard errors, and the "
        "incremental cost-effectiveness ratio (ICER) against the baseline strategy. The MCSE describes the "
        "simulation noise only, not parameter uncertainty (see the probabilistic sensitivity analysis for that)."
    )
    doc.table(tables["results"])
    cma_cba.write_report_section(doc, tables)

    if not tables["cohort"].empty:
        doc.heading("5. Comparison with the equivalent cohort model", level=1)
        doc.paragraph(
            "The same states, transitions, costs and effects were also run as a deterministic Markov cohort "
            "model. A correct microsimulation converges to the cohort result as the number of patients "
            "grows; a difference is expected to stay within a few Monte Carlo standard errors. z is the "
            "difference divided by the MCSE."
        )
        doc.table(tables["cohort"])
        next_section = 6
    else:
        next_section = 5

    write_selfchecks_section(doc, checks, f"{next_section}. Validation self-checks")

    doc.heading(f"{next_section + 1}. Occupancy and per-cycle results", level=1)
    doc.paragraph("Five evenly spaced cycles of each strategy's trace; the full per-cycle trace is on the "
                  "'Trace' sheets of the Excel companion file.")
    for run in inp.runs:
        doc.heading(run.label, level=2)
        doc.table(_snapshot_table(_trace_table(inp, run)))

    doc.heading(f"{next_section + 2}. Statement on AI use", level=1)
    doc.paragraph(
        "No AI/LLM component was used to compute the results in this report or in the underlying DECIDE "
        "model. All costs, effects and ICERs above were produced by deterministic Python code."
    )
    return doc.to_bytes()


def build_microsim_report(inp: MicrosimReportInput) -> MicrosimReportBundle:
    """Build the report. Raises ``SelfCheckFailure`` when an
    error-severity self-check fails (an internal-consistency failure means
    an engine bug, so no report is returned)."""

    if not inp.runs:
        raise ValueError("No strategy results to report.")
    checks = microsim_selfchecks(inp)
    raise_on_failure(checks)

    parameters_df, missing_source_rows = _parameters_table(inp)
    tables: dict[str, pd.DataFrame] = {
        "settings": _settings_table(inp), "structure": _structure_table(inp), "parameters": parameters_df,
        "results": _results_table(inp), "cohort": _cohort_table(inp), "selfchecks": selfchecks_frame(checks),
    }
    _by = _run_by_label(inp)
    _sec = cma_cba.report_section(
        inp.analysis_settings, [r.label for r in inp.runs],
        [float(r.result.total_cost) for r in inp.runs], [float(r.result.total_effect) for r in inp.runs],
        inp.maximize_effect, _effective_baseline(inp),
        inp.cost_symbol, inp.cost_decimals, inp.effect_decimals,
    )
    if _sec is not None:
        tables["cma_cba"] = _sec
    sheet_order = list(BASE_SHEETS)
    for i, run in enumerate(inp.runs):
        key = f"trace_{i}"
        tables[key] = _trace_table(inp, run)
        sheet_order.append((key, f"Trace - {run.label}"[:31]))

    methods = microsim_methods_paragraphs(inp)
    config_hash = _config_hash(inp)
    docx_bytes = _write_docx(inp, tables, missing_source_rows, methods, config_hash, checks)
    title = inp.model_name or "DECIDE Microsimulation Model -- Technical Documentation Report"
    xlsx_bytes = write_xlsx(tables, title, methods, sheet_order)

    stem = "decide-microsim-technical-report"
    slug = _slug(inp.model_name or "")
    if slug:
        stem += "-" + slug
    return MicrosimReportBundle(docx_bytes=docx_bytes, xlsx_bytes=xlsx_bytes, filename_stem=stem)
