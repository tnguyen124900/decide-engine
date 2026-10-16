"""Excel export of a cohort Markov model with a live, auditable formula
trail -- the Markov half of the same transparency work as
``tree_excel_export.py`` (see that module's own docstring
for the shared two-tier design: Parameters holds resolved values,
everything the engine does on top of them is a live Excel formula).

Reuses ``tree_excel_export.py``'s generic helpers (``_write_parameters_sheet``,
``_style_header_row``, ``_autosize``, ``_live_ref``, ``_cell_live_or_resolved``,
the fill/comment helpers) rather than duplicating them -- they were
written model-agnostically in the first place (a ``Param`` is a ``Param``
regardless of Tree or Markov), so importing them keeps the two exporters
from silently drifting apart on what "live" vs "resolved" means or looks
like. The one exception is the Read-me cover sheet: Tree's own
``_write_cover_sheet`` names Tree-only sheets ("Tree Paths", "Strategy
Summary / ICER") in its "how to read this workbook" text, so this module
has its own ``_write_markov_cover_sheet`` instead, describing this
workbook's actual sheets -- see that function's own docstring.

Cohort Trace, always both sheets (a fixed design decision, not
a per-export choice): ``Cohort Trace (state x tenure)`` is the
authoritative one -- one column per (state, tenure) pair the engine can
actually occupy (a state with ``tenure_applied=False`` contributes exactly
one column, at tenure 1) -- built the same way ``run_markov()`` itself
advances the cohort: a live ``SUMPRODUCT`` of the previous cycle's row
against the matching column of that cycle's Transition Matrix table (see
``_write_transition_matrix_sheet``). ``Cohort Trace (by state)`` is a live
``SUM`` of the by-tenure sheet's own columns for that state -- i.e. it does
not re-derive occupancy independently, it is *defined* as the by-tenure
sheet's aggregate. That is a deliberate, stronger version of the "both totals must reconcile, self-check at build time" requirement:
a live ``SUM`` formula guarantees the two sheets can never drift apart, at
any point in the file's life (including after a reviewer edits a
Parameters cell and recalculates), which a Python-time-only comparison of
two independently-computed numbers cannot guarantee once the file leaves
DECIDE's hands.

That does not eliminate the risk of *this exporter's own code* having a
bug in how it resolves a state's tenure-based cost split (Initial/
Recurring/Terminal, see ``MarkovState``'s own docstring) -- that math is
non-trivial and doesn't come from a `PathRow`-like already-vetted engine
data structure the way Tree's per-path figures do. So this module DOES
also do exactly the build-time cross-check:
``build_markov_transparency_workbook()`` recomputes total cost/effect
itself from the same resolved numbers written into the "State Rewards"
sheet's cells, applied to ``MarkovResult.tenure_trace`` (the engine's own,
trusted, per-(state,tenure) occupancy -- see that field's docstring in
``markov_models.py``), and compares the result to ``MarkovResult``'s own
``total_cost``/``total_effect`` (independently computed by
``run_markov()``'s ``reward()``/``advance()``). A mismatch beyond a tight
tolerance raises ``MarkovExportError`` and no workbook is returned --
the export is blocked rather than silently shipping a wrong file.

Time-varying transitions are the default case here, not an edge case (by
design): the Transition Matrix sheet is
*always* one stacked table per cycle, even for a model where every
transition happens to be a plain constant -- there is no "simple" single-
table mode to fall back to, because there is no way to tell from the
model alone whether a transition that looks constant today will stay that
way after a reviewer starts editing Parameters cells that feed a formula.

Limits, stated rather than hidden (mirrors the transparency export design):
state cost/effect/extra-outcome rates and transition probabilities are
resolved (and checked for "is this a bare parameter reference") using
each cycle's own ``_cycle``/``_tenure`` variables, so a reward or
probability that genuinely varies with them is still numerically correct
cycle by cycle -- it just cannot be *live* the way a bare parameter
reference is, since Excel has no equivalent of DECIDE's expression
evaluator. Such a cell is filled/commented exactly like Tree's exporter
does for the same situation.
"""

from __future__ import annotations

import hashlib
import io
from collections.abc import Mapping, Sequence
from typing import Any, Callable

from .markov_engine import _engine_variables, _resolve as _base_resolve
from .markov_models import MarkovModel, MarkovResult, MarkovState, MarkovTransition
from .models import Param, is_complement
from .tree_excel_export import (
    _MISSING_SOURCE_FILL,  # noqa: F401 -- re-exported for callers that want it
    _RESOLVED_FILL,
    _autosize,
    _cell_live_or_resolved,
    _excel_quote,
    _live_ref,
    _resolved_comment,
    _style_header_row,
    _write_parameters_sheet,
)

# Deliberately NOT reusing tree_excel_export._write_cover_sheet: that
# function's own "how to read this workbook" bullets name Tree-only sheets
# ("Tree Paths", "Strategy Summary / ICER") that don't exist in a Markov
# workbook -- reusing it verbatim would put factually wrong sheet names in
# every Markov export's Read-me page. Everything else about the two cover
# sheets (title/timestamp/engine-version/config-hash layout, the AI-use
# statement, the bold title + wrapped-text styling) is intentionally kept
# identical, just with Markov's own sheet names in the bullets below.

Unit = tuple[str, int]  # (state_id, tenure)


class MarkovExportError(ValueError):
    """Raised when this module's own build-time self-check finds its
    resolved cost/effect totals don't match ``run_markov()``'s -- see the
    module docstring. Never silently swallowed by app.py's usual
    try/except-and-show-a-caption pattern without at least a clear
    message, since this specifically means "do not trust this export"."""


# -- shared small helpers --------------------------------------------------


def _tenure_units(model: MarkovModel) -> list[Unit]:
    units: list[Unit] = []
    for state in model.states:
        count = state.tenure_count if state.tenure_applied else 1
        for tenure in range(1, max(1, count) + 1):
            units.append((state.id, tenure))
    return units


def _unit_label(states_by_id: Mapping[str, MarkovState], unit: Unit) -> str:
    state_id, tenure = unit
    state = states_by_id[state_id]
    if state.tenure_applied and state.tenure_count > 1:
        return f"{state.name} (tenure {tenure})"
    return state.name


def _next_unit(state: MarkovState, from_tenure: int, to_state_id: str) -> Unit:
    if to_state_id == state.id:
        next_tenure = min(from_tenure + 1, state.tenure_count) if state.tenure_applied else 1
        return (state.id, next_tenure)
    return (to_state_id, 1)


def _grouped_transitions(model: MarkovModel) -> dict[str, list[MarkovTransition]]:
    grouped: dict[str, list[MarkovTransition]] = {state.id: [] for state in model.states}
    for transition in model.transitions:
        grouped[transition.from_state].append(transition)
    return grouped


def _transition_by_pair(model: MarkovModel) -> dict[tuple[str, str], MarkovTransition]:
    return {(t.from_state, t.to_state): t for t in model.transitions}


def _resolve_group_with_raw(
    entries: list[tuple[str, Any]],
    parameters: Mapping[str, Param],
    variables: Mapping[str, Any] | None,
    distributions: Mapping[str, Mapping[str, Any]] | None,
    tables: Mapping[str, Mapping[str, Any]] | None,
) -> dict[str, dict[str, Any]]:
    """Like ``markov_engine._resolve_probability_group``, but keeps each
    destination's raw reference alongside its resolved value so the
    caller can tell live (bare parameter) apart from resolved (everything
    else). The complement entry (``#``/``C``) always has ``raw=None``: its
    value is *derived* from the others, so it can never itself be a bare
    parameter reference -- this exporter always writes it as a resolved
    value (mirrors Tree's own scope decision not to special-case a
    complement into a live ``1-SUM(...)`` formula)."""

    resolved: dict[str, dict[str, Any]] = {}
    complement_key: str | None = None
    known_sum = 0.0
    for key, value in entries:
        if is_complement(value):
            complement_key = key
            continue
        probability = _base_resolve(value, parameters, variables, distributions, tables)
        resolved[key] = {"value": probability, "raw": value}
        known_sum += probability
    if complement_key is not None:
        resolved[complement_key] = {"value": 1.0 - known_sum, "raw": None}
    return resolved


def _outgoing_with_raw(
    state_id: str,
    grouped: Mapping[str, list[MarkovTransition]],
    parameters: Mapping[str, Param],
    variables: Mapping[str, Any] | None,
    distributions: Mapping[str, Mapping[str, Any]] | None,
    tables: Mapping[str, Mapping[str, Any]] | None,
) -> dict[str, dict[str, Any]]:
    entries = grouped.get(state_id, [])
    if not entries:
        # Absorbing state with no declared transitions: implicit
        # self-loop, probability 1 -- a structural fact, not a resolved
        # expression, so raw=None here means "literal", not "flag me".
        return {state_id: {"value": 1.0, "raw": None, "literal": True}}
    resolved = _resolve_group_with_raw(
        [(t.to_state, t.probability) for t in entries], parameters, variables, distributions, tables
    )
    for info in resolved.values():
        info.setdefault("literal", False)
    return resolved


_SHEET_NAME_FORBIDDEN = set('[]:*?/\\')


def _safe_extra_sheet_name(label: str, used: set[str]) -> str:
    """A valid, unique <=31-char Excel sheet name for an extra outcome's
    "Extra - <label>" sheet (Excel forbids ``[]:*?/\\`` in a sheet name and
    caps it at 31 characters -- a plain ``label`` from the Outcomes table
    could hit either)."""

    cleaned = "".join(ch for ch in label if ch not in _SHEET_NAME_FORBIDDEN).strip() or "outcome"
    base = f"Extra - {cleaned}"[:31]
    candidate = base
    suffix = 2
    while candidate in used:
        tail = f" ({suffix})"
        candidate = base[: 31 - len(tail)] + tail
        suffix += 1
    used.add(candidate)
    return candidate


def _write_markov_cover_sheet(
    wb: Any, model_name: str, engine_version: str, config_hash: str, ai_statement: bool
) -> None:
    """Markov's own Read-me cover sheet -- same layout/styling as
    ``tree_excel_export._write_cover_sheet``, but describing this
    workbook's actual sheets (Transition Matrix / Cohort Trace / State
    Rewards / Cost / Effect / Results) instead of Tree's."""

    from datetime import datetime, timezone

    ws = wb.create_sheet("Read me", 0)
    lines = [
        model_name or "DECIDE Markov Model -- transparency export",
        f"Exported: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"DECIDE engine version: {engine_version or 'unknown'}",
        f"Model configuration hash: {config_hash}",
        "",
        "How to read this workbook:",
        "- Parameters: every input's resolved value, distribution and source citation. A parameter "
        "whose value comes from a formula/distribution shows the resolved number here (Excel cannot "
        "represent every DECIDE expression as a native formula) -- see Distribution/Dist args.",
        "- Transition Matrix: one table per cycle, columns are the 'from' state/tenure, rows are the "
        "'to' state/tenure -- amber cells are resolved values (a transition whose probability comes "
        "from a formula); everything else is the actual probability used that cycle.",
        "- Cohort Trace (state x tenure) / Cohort Trace (by state): fully live formulas -- the cohort "
        "occupying each state (split by tenure, or summed across it) at the start of each cycle, "
        "built from the previous cycle's row and that cycle's Transition Matrix table. The 'by state' "
        "sheet is defined as a live sum of the by-tenure sheet's own columns, so the two can never "
        "drift apart, even after you edit Parameters and recalculate.",
        "- State Rewards: each state's per-cycle Initial/Recurring/Terminal cost, exit probability "
        "and Cost/Effect rate, used by the Cost/Effect/Results sheets below.",
        "- Cost / Effect (and any 'Extra - ...' outcome sheet): one row per cycle, fully live "
        "formulas built from Cohort Trace and State Rewards, raw and discounted.",
        "- Results: fully live totals summed from Cost/Effect(/Extra), including the half-cycle "
        "correction term written out explicitly when it applies -- change any Parameters cell and "
        "these recalculate.",
    ]
    if ai_statement:
        lines += ["", "No AI/LLM component was used to compute the results in this file."]
    for i, text in enumerate(lines, start=1):
        cell = ws.cell(row=i, column=1, value=text)
        if i == 1:
            from openpyxl.styles import Font

            cell.font = Font(bold=True, size=14)
    ws.column_dimensions["A"].width = 120
    from openpyxl.styles import Alignment

    for row in ws.iter_rows():
        for cell in row:
            cell.alignment = Alignment(wrap_text=True, vertical="top")


def _config_hash(parameters: Mapping[str, Param], model: MarkovModel) -> str:
    parts: list[str] = []
    for name in sorted(parameters):
        p = parameters[name]
        parts.append(f"{name}={p.base}|{p.low}|{p.high}|{p.dist}|{p.dist_args}|{p.source}")
    for state in model.states:
        parts.append(
            f"state={state.id}|{state.cost}|{state.tenure_applied}|{state.tenure_count}|"
            f"{state.initial_cost}|{state.recurring_cost}|{state.terminal_cost}|{state.effect}|"
            f"{state.initial_prob}|{state.is_absorbing}|{sorted(state.extra_outcomes.items())}"
        )
    for t in model.transitions:
        parts.append(
            f"transition={t.from_state}->{t.to_state}|{t.probability}|{t.cost_increment}|"
            f"{t.effect_increment}|{sorted(t.extra_increment.items())}"
        )
    parts.append(f"cycles={model.cycles}|half_cycle={model.half_cycle_correction}")
    digest = hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()
    return digest[:12]


def _write_value_cell(
    ws: Any, row: int, col: int, info: Mapping[str, Any], param_defined_names: Mapping[str, str],
) -> Any:
    """Write one probability/reward cell from a ``{"value", "raw",
    ["literal"]}`` info dict (see ``_resolve_group_with_raw``/
    ``_outgoing_with_raw``): a bare parameter reference is live, a plain
    literal (``literal=True``, e.g. an absorbing state's implicit
    self-loop, or a raw numeric constant in the model) is an unflagged
    number, and anything else (an expression, or a derived complement) is
    a resolved number flagged as such."""

    if info.get("literal"):
        return ws.cell(row=row, column=col, value=float(info["value"]))
    return _cell_live_or_resolved(
        ws, row, col, info.get("raw"), info["value"], param_defined_names, _resolved_fill(),
    )


_RESOLVED_FILL_OBJ = None


def _resolved_fill() -> Any:
    global _RESOLVED_FILL_OBJ
    if _RESOLVED_FILL_OBJ is None:
        from openpyxl.styles import PatternFill

        _RESOLVED_FILL_OBJ = PatternFill("solid", fgColor=_RESOLVED_FILL)
    return _RESOLVED_FILL_OBJ


# -- Transition Matrix ------------------------------------------------------


def _write_transition_matrix_sheet(
    wb: Any,
    model: MarkovModel,
    units: Sequence[Unit],
    states_by_id: Mapping[str, MarkovState],
    parameters: Mapping[str, Param],
    param_defined_names: Mapping[str, str],
    variables: Mapping[str, Any] | None,
    distributions: Mapping[str, Mapping[str, Any]] | None,
    tables: Mapping[str, Mapping[str, Any]] | None,
) -> dict[str, Any]:
    """One stacked square units-x-units table per cycle (0 .. cycles-1) --
    time-varying transitions are the default case, not a special mode
    (see module docstring). COLUMNS are the "from" unit, ROWS are the "to"
    unit -- deliberately the transpose of how a transition matrix is
    usually drawn on paper -- because it makes each destination's
    coefficients a single ROW (``$B$row:$last_col$row``), the same shape
    as a Cohort Trace cycle's own row of masses; a Cohort Trace cell can
    then be one plain ``SUMPRODUCT`` of two same-shaped row ranges. The
    natural alternative (from = rows, to = columns, so a destination's
    coefficients are a column) would need ``TRANSPOSE`` inside every
    ``SUMPRODUCT`` to line up with a Cohort Trace row -- and plain
    ``SUMPRODUCT(row_range, column_range)`` without it is a dimension
    mismatch (``#VALUE!``), not silently wrong, which is how this
    decision got made (see this module's own test harness).

    Returns bookkeeping the Cohort Trace/Cost/Effect sheets need: per
    cycle, the table's header/data rows and each "from" unit's column
    letter, plus (for the tenure cost-split formula) each unit's own
    "self-loop" cell coordinate -- the probability of staying in the same
    state, which is exactly ``1 - exit_probability``.
    """

    from openpyxl.utils import get_column_letter

    ws = wb.create_sheet("Transition Matrix")
    grouped = _grouped_transitions(model)
    n = len(units)
    unit_index = {unit: i for i, unit in enumerate(units)}

    per_cycle: dict[int, dict[str, Any]] = {}
    row = 1
    for cycle in range(model.cycles):
        ws.cell(row=row, column=1, value=f"Cycle {cycle}")
        header_row = row + 1
        ws.cell(row=header_row, column=1, value="To \\ From")
        for j, unit in enumerate(units):
            ws.cell(row=header_row, column=2 + j, value=_unit_label(states_by_id, unit))
        _style_header_row(ws, header_row, n + 1)
        first_data_row = header_row + 1

        # Row headers: the "to" unit each row represents.
        for i, unit in enumerate(units):
            ws.cell(row=first_data_row + i, column=1, value=_unit_label(states_by_id, unit))

        unit_col: dict[Unit, str] = {unit: get_column_letter(2 + j) for j, unit in enumerate(units)}
        self_loop_coord: dict[Unit, str] = {}
        self_loop_value: dict[Unit, float] = {}
        prob_value: dict[tuple[Unit, Unit], float] = {}

        for j, (state_id, tenure) in enumerate(units):
            col = 2 + j
            state = states_by_id[state_id]
            outgoing = _outgoing_with_raw(
                state_id, grouped, parameters, _engine_variables(cycle, tenure, variables), distributions, tables,
            )
            target_for: dict[Unit, dict[str, Any]] = {}
            for to_state, info in outgoing.items():
                target = _next_unit(state, tenure, to_state)
                target_for[target] = info
            for i, target in enumerate(units):
                r = first_data_row + i
                info = target_for.get(target)
                if info is None:
                    ws.cell(row=r, column=col, value=0.0)
                    continue
                _write_value_cell(ws, r, col, info, param_defined_names)
                prob_value[((state_id, tenure), target)] = float(info["value"])

            self_target = _next_unit(state, tenure, state_id)
            self_loop_row = first_data_row + unit_index[self_target]
            self_loop_coord[(state_id, tenure)] = f"'{ws.title}'!${unit_col[(state_id, tenure)]}${self_loop_row}"
            self_loop_value[(state_id, tenure)] = prob_value.get(((state_id, tenure), self_target), 0.0)

        _autosize(ws, n + 1)
        per_cycle[cycle] = {
            "header_row": header_row,
            "first_data_row": first_data_row,
            "last_data_row": first_data_row + n - 1,
            "unit_col": unit_col,
            "self_loop_coord": self_loop_coord,
            "self_loop_value": self_loop_value,
            "prob_value": prob_value,
        }
        row = first_data_row + n + 1  # one blank row between cycle tables

    return {"sheet_name": ws.title, "per_cycle": per_cycle, "unit_index": unit_index}


# -- Cohort Trace -----------------------------------------------------------


def _write_cohort_trace_by_tenure_sheet(
    wb: Any,
    model: MarkovModel,
    units: Sequence[Unit],
    states_by_id: Mapping[str, MarkovState],
    parameters: Mapping[str, Param],
    param_defined_names: Mapping[str, str],
    matrix_info: Mapping[str, Any],
    variables: Mapping[str, Any] | None,
    distributions: Mapping[str, Mapping[str, Any]] | None,
    tables: Mapping[str, Mapping[str, Any]] | None,
) -> dict[str, Any]:
    """Cycle 0's row is the initial distribution (live/resolved per unit,
    same complement-aware resolution ``run_markov()`` itself uses -- one
    group across every state, since initial probabilities must sum to 1
    across the whole model, not per state). Every later row is a pure
    live formula: ``SUMPRODUCT`` of the previous row against the matching
    Transition Matrix column -- see that function's own docstring."""

    from openpyxl.utils import get_column_letter

    ws = wb.create_sheet("Cohort Trace (state x tenure)")
    headers = ["Cycle"] + [_unit_label(states_by_id, unit) for unit in units]
    ws.append(headers)
    _style_header_row(ws, 1, len(headers))

    unit_col = {unit: get_column_letter(2 + j) for j, unit in enumerate(units)}
    data_row_of_cycle: dict[int, int] = {}

    initial_group = _resolve_group_with_raw(
        [(state.id, state.initial_prob) for state in model.states],
        parameters, _engine_variables(0, 1, variables), distributions, tables,
    )

    r = 2
    ws.cell(row=r, column=1, value=0)
    for j, unit in enumerate(units):
        state_id, tenure = unit
        col = 2 + j
        if tenure == 1:
            _write_value_cell(ws, r, col, initial_group.get(state_id, {"value": 0.0, "raw": None, "literal": True}), param_defined_names)
        else:
            ws.cell(row=r, column=col, value=0.0)
    data_row_of_cycle[0] = r

    for cycle in range(model.cycles):
        prev_row = data_row_of_cycle[cycle]
        r = prev_row + 1
        ws.cell(row=r, column=1, value=cycle + 1)
        cyc_info = matrix_info["per_cycle"][cycle]
        prev_range = f"$B${prev_row}:${get_column_letter(1 + len(units))}${prev_row}"
        last_col_letter = get_column_letter(1 + len(units))
        for j, target in enumerate(units):
            # Transition Matrix has "from" as columns and "to" as rows (see
            # that function's own docstring for why); this target's row,
            # spanning every "from" column, is the same shape as prev_range.
            target_row = cyc_info["first_data_row"] + j
            matrix_row_range = f"'{matrix_info['sheet_name']}'!$B${target_row}:${last_col_letter}${target_row}"
            ws.cell(row=r, column=2 + j, value=f"=SUMPRODUCT({prev_range},{matrix_row_range})")
        data_row_of_cycle[cycle + 1] = r

    _autosize(ws, len(headers))
    return {"sheet_name": ws.title, "unit_col": unit_col, "data_row_of_cycle": data_row_of_cycle}


def _write_cohort_trace_by_state_sheet(
    wb: Any, model: MarkovModel, units: Sequence[Unit], by_tenure: Mapping[str, Any],
) -> dict[str, Any]:
    """A live ``SUM`` of the by-tenure sheet's own columns for each state
    -- deliberately *defined* as that sheet's aggregate rather than
    independently re-derived, so the two can never drift apart at any
    point in the file's life (see module docstring)."""

    ws = wb.create_sheet("Cohort Trace (by state)")
    state_ids = [state.id for state in model.states]
    states_by_id = {state.id: state for state in model.states}
    headers = ["Cycle"] + [states_by_id[sid].name for sid in state_ids]
    ws.append(headers)
    _style_header_row(ws, 1, len(headers))

    tenure_sheet = by_tenure["sheet_name"]
    cols_by_state: dict[str, list[str]] = {sid: [] for sid in state_ids}
    for unit in units:
        cols_by_state[unit[0]].append(by_tenure["unit_col"][unit])

    data_row_of_cycle: dict[int, int] = {}
    r = 2
    for cycle in sorted(by_tenure["data_row_of_cycle"]):
        src_row = by_tenure["data_row_of_cycle"][cycle]
        ws.cell(row=r, column=1, value=cycle)
        for i, sid in enumerate(state_ids):
            cols = cols_by_state[sid]
            refs = "+".join(f"'{tenure_sheet}'!${c}${src_row}" for c in cols)
            ws.cell(row=r, column=2 + i, value=f"={refs}" if refs else 0.0)
        data_row_of_cycle[cycle] = r
        r += 1

    _autosize(ws, len(headers))
    return {"sheet_name": ws.title, "data_row_of_cycle": data_row_of_cycle}


# -- State Rewards (helper sheet backing Cost/Effect) ------------------------


def _write_state_rewards_sheet(
    wb: Any,
    model: MarkovModel,
    units: Sequence[Unit],
    states_by_id: Mapping[str, MarkovState],
    parameters: Mapping[str, Param],
    param_defined_names: Mapping[str, str],
    matrix_info: Mapping[str, Any],
    extra_outcome_ids: Sequence[str],
    extra_labels: Mapping[str, str],
    variables: Mapping[str, Any] | None,
    distributions: Mapping[str, Mapping[str, Any]] | None,
    tables: Mapping[str, Mapping[str, Any]] | None,
) -> dict[str, Any]:
    """One stacked table per cycle (0 .. cycles-1, "apply_terminal=True" --
    matches every regular per-cycle reward in ``run_markov()``), plus one
    final table at "cycle = cycles" with no terminal-cost blend, used only
    by the half-cycle correction (mirrors ``reward()``'s own
    ``apply_terminal=False`` branch). Every column here is a single
    resolved-or-live *rate* (not yet multiplied by occupancy) -- the
    Cost/Effect sheets reference these cells directly, so a downstream
    formula never needs to know whether the cell it points to happens to
    be live or resolved; it just inherits whichever it is (see module
    docstring: this is why nothing here needs a three-way "partially
    live" cell -- only leaf cells like these are ever flagged).

    For a state with ``tenure_applied=False`` only "Cost rate" is filled
    (Initial/Recurring/Terminal/Exit probability stay blank -- not
    applicable, not "zero"). For ``tenure_applied=True``, tenure 1 fills
    Initial+Terminal+Exit probability+Cost rate; tenure>1 fills
    Recurring+Terminal+Exit probability+Cost rate -- mirroring exactly
    which branch of ``reward()``'s ``if not state.tenure_applied: ...
    else: ... if tenure==1 ... else ...`` applies.
    """

    ws = wb.create_sheet("State Rewards")
    extra_headers: list[str] = []
    for oid in extra_outcome_ids:
        extra_headers.append(f"{extra_labels.get(oid, oid)} rate")
    headers = [
        "Unit", "Initial cost", "Recurring cost", "Terminal cost", "Exit probability",
        "Cost rate", "Effect rate",
    ] + extra_headers

    per_cycle: dict[Any, dict[str, Any]] = {}
    row = 1

    def write_table(cycle_label: str, cycle: int, apply_terminal: bool) -> dict[str, Any]:
        nonlocal row
        ws.cell(row=row, column=1, value=cycle_label)
        header_row = row + 1
        for c, h in enumerate(headers, start=1):
            ws.cell(row=header_row, column=c, value=h)
        _style_header_row(ws, header_row, len(headers))
        first_data_row = header_row + 1

        unit_row: dict[Unit, int] = {}
        rate_cost_cell: dict[Unit, str] = {}
        rate_effect_cell: dict[Unit, str] = {}
        rate_extra_cell: dict[Unit, dict[str, str]] = {}
        rate_cost_value: dict[Unit, float] = {}
        rate_effect_value: dict[Unit, float] = {}
        rate_extra_value: dict[Unit, dict[str, float]] = {}

        for i, (state_id, tenure) in enumerate(units):
            r = first_data_row + i
            unit_row[(state_id, tenure)] = r
            state = states_by_id[state_id]
            state_vars = _engine_variables(cycle, tenure, variables)
            ws.cell(row=r, column=1, value=_unit_label(states_by_id, (state_id, tenure)))

            effect_val = _base_resolve(state.effect, parameters, state_vars, distributions, tables)
            effect_cell = _cell_live_or_resolved(
                ws, r, 7, state.effect, effect_val, param_defined_names, _resolved_fill(),
            )
            rate_effect_cell[(state_id, tenure)] = effect_cell.coordinate
            rate_effect_value[(state_id, tenure)] = effect_val

            col = 8
            rate_extra_value[(state_id, tenure)] = {}
            for oid in extra_outcome_ids:
                raw = state.extra_outcomes.get(oid)
                rate_extra_cell.setdefault((state_id, tenure), {})
                if raw is None:
                    ws.cell(row=r, column=col, value=0.0)
                    rate_extra_cell[(state_id, tenure)][oid] = ws.cell(row=r, column=col).coordinate
                    rate_extra_value[(state_id, tenure)][oid] = 0.0
                else:
                    val = _base_resolve(raw, parameters, state_vars, distributions, tables)
                    cell = _cell_live_or_resolved(ws, r, col, raw, val, param_defined_names, _resolved_fill())
                    rate_extra_cell[(state_id, tenure)][oid] = cell.coordinate
                    rate_extra_value[(state_id, tenure)][oid] = val
                col += 1

            if not state.tenure_applied:
                cost_val = _base_resolve(state.cost, parameters, state_vars, distributions, tables)
                cost_cell = _cell_live_or_resolved(
                    ws, r, 6, state.cost, cost_val, param_defined_names, _resolved_fill(),
                )
                rate_cost_cell[(state_id, tenure)] = cost_cell.coordinate
                rate_cost_value[(state_id, tenure)] = cost_val
                continue

            terminal_val = _base_resolve(state.terminal_cost, parameters, state_vars, distributions, tables)
            terminal_cell = _cell_live_or_resolved(
                ws, r, 4, state.terminal_cost, terminal_val, param_defined_names, _resolved_fill(),
            )

            if apply_terminal:
                self_loop_coord = matrix_info["per_cycle"][cycle]["self_loop_coord"][(state_id, tenure)]
                self_loop_val = matrix_info["per_cycle"][cycle]["self_loop_value"][(state_id, tenure)]
                exit_cell = ws.cell(row=r, column=5, value=f"=1-{self_loop_coord}")
                exit_val = 1.0 - self_loop_val
            else:
                exit_cell = None
                exit_val = None

            if tenure == 1:
                initial_val = _base_resolve(state.initial_cost, parameters, state_vars, distributions, tables)
                initial_cell = _cell_live_or_resolved(
                    ws, r, 2, state.initial_cost, initial_val, param_defined_names, _resolved_fill(),
                )
                # Leaving during the first cycle in the state: only Initial cost (no Terminal).
                cost_formula = f"={initial_cell.coordinate}"
                cost_val = initial_val
            else:
                recurring_raw = state.recurring_cost if state.recurring_cost is not None else state.cost
                recurring_val = _base_resolve(recurring_raw, parameters, state_vars, distributions, tables)
                recurring_cell = _cell_live_or_resolved(
                    ws, r, 3, recurring_raw, recurring_val, param_defined_names, _resolved_fill(),
                )
                if apply_terminal:
                    cost_formula = (
                        f"={recurring_cell.coordinate}*(1-{exit_cell.coordinate})"
                        f"+{terminal_cell.coordinate}*{exit_cell.coordinate}"
                    )
                    cost_val = recurring_val * (1.0 - exit_val) + terminal_val * exit_val
                else:
                    cost_formula = f"={recurring_cell.coordinate}"
                    cost_val = recurring_val
            cost_cell = ws.cell(row=r, column=6, value=cost_formula)
            rate_cost_cell[(state_id, tenure)] = cost_cell.coordinate
            rate_cost_value[(state_id, tenure)] = cost_val

        _autosize(ws, len(headers))
        last_data_row = first_data_row + len(units) - 1
        row = last_data_row + 2
        return {
            "header_row": header_row, "first_data_row": first_data_row, "last_data_row": last_data_row,
            "unit_row": unit_row, "cost_cell": rate_cost_cell, "effect_cell": rate_effect_cell,
            "extra_cell": rate_extra_cell, "cost_value": rate_cost_value, "effect_value": rate_effect_value,
            "extra_value": rate_extra_value,
        }

    for cycle in range(model.cycles):
        per_cycle[cycle] = write_table(f"Cycle {cycle}", cycle, apply_terminal=True)
    if model.half_cycle_correction and model.cycles > 0:
        per_cycle["final"] = write_table(
            f"Cycle {model.cycles} (no terminal blend -- half-cycle correction)",
            model.cycles, apply_terminal=False,
        )

    return {"sheet_name": ws.title, "per_cycle": per_cycle}


# -- Cost / Effect / Extra outcomes -----------------------------------------


def _write_reward_sheet(
    wb: Any,
    sheet_name: str,
    model: MarkovModel,
    units: Sequence[Unit],
    by_tenure: Mapping[str, Any],
    rewards: Mapping[str, Any],
    rate_coord: "Callable[[dict, Unit], str]",
    discount_rate: float,
    transition_increment_of: "Callable[[MarkovTransition], Any]",
    param_defined_names: Mapping[str, str],
    parameters: Mapping[str, Param],
    variables: Mapping[str, Any] | None,
    distributions: Mapping[str, Mapping[str, Any]] | None,
    tables: Mapping[str, Mapping[str, Any]] | None,
    matrix_info: Mapping[str, Any],
) -> dict[str, Any]:
    """Shared builder for the Cost, Effect and Extra-outcome sheets.
    ``rate_coord(cyc_rewards, unit)`` picks that unit's rate cell
    coordinate off the State Rewards bookkeeping for this cycle (cost,
    effect, or a specific extra outcome's rate); ``transition_increment_of
    (transition)`` picks the matching increment value (``cost_increment``/
    ``effect_increment``/``extra_increment.get(oid)``) off a
    ``MarkovTransition``. One row per cycle: each unit's contribution is
    ``mass * rate`` (both live cell references -- see
    ``_write_state_rewards_sheet``'s own docstring for why this is always
    a clean live formula regardless of whether the rate itself is live or
    resolved), summed into that cycle's Raw total, then discounted."""

    ws = wb.create_sheet(sheet_name)
    states_by_id = {state.id: state for state in model.states}
    grouped = _grouped_transitions(model)

    unit_headers = [f"{_unit_label(states_by_id, u)}: contribution" for u in units]
    trans_inc_units: list[tuple[Unit, str, MarkovTransition]] = []
    for (state_id, tenure) in units:
        for transition in grouped.get(state_id, []):
            value = transition_increment_of(transition)
            if value not in (None, 0, 0.0):
                trans_inc_units.append(((state_id, tenure), transition.to_state, transition))

    headers = ["Cycle"] + unit_headers
    trans_col_start = len(headers) + 1
    for (unit, to_state, transition) in trans_inc_units:
        headers.append(f"{_unit_label(states_by_id, unit)}->{to_state}: transition increment")
    total_raw_col = len(headers) + 1
    headers += ["Raw total", "Discounted total"]

    ws.append(headers)
    _style_header_row(ws, 1, len(headers))

    trace_col = by_tenure["unit_col"]

    r = 2
    for cycle in range(model.cycles):
        ws.cell(row=r, column=1, value=cycle)
        mass_row = by_tenure["data_row_of_cycle"][cycle]
        cyc_rewards = rewards["per_cycle"][cycle]

        contribution_refs: list[str] = []
        for j, unit in enumerate(units):
            mass_ref = f"'{by_tenure['sheet_name']}'!${trace_col[unit]}${mass_row}"
            rate_ref = f"'{rewards['sheet_name']}'!{rate_coord(cyc_rewards, unit)}"
            cell = ws.cell(row=r, column=2 + j, value=f"={mass_ref}*{rate_ref}")
            contribution_refs.append(cell.coordinate)

        col = trans_col_start
        for (unit, to_state, transition) in trans_inc_units:
            state = states_by_id[unit[0]]
            target = _next_unit(state, unit[1], to_state)
            # Transition Matrix: "from" (this transition's source unit) is
            # the column, "to" (target) is the row -- see that function's
            # own docstring.
            prob_col = matrix_info["per_cycle"][cycle]["unit_col"][unit]
            prob_row = matrix_info["per_cycle"][cycle]["first_data_row"] + matrix_info["unit_index"][target]
            prob_ref = f"'{matrix_info['sheet_name']}'!${prob_col}${prob_row}"
            mass_ref = f"'{by_tenure['sheet_name']}'!${trace_col[unit]}${mass_row}"
            incr_value = transition_increment_of(transition)
            incr_val = _base_resolve(incr_value, parameters, _engine_variables(cycle, unit[1], variables), distributions, tables)
            incr_cell_holder = ws.cell(row=r, column=col)
            live = _live_ref(incr_value, param_defined_names)
            incr_expr = live[1:] if live else None
            if incr_expr is None:
                # Embed the resolved constant directly. Only flag it if it
                # came from an actual expression (a string that wasn't a
                # bare parameter reference) -- a plain numeric literal in
                # the model has nothing hidden behind it, same convention
                # as _cell_live_or_resolved.
                formula = f"={mass_ref}*{prob_ref}*{incr_val}"
                incr_cell_holder.value = formula
                if isinstance(incr_value, str):
                    incr_cell_holder.fill = _resolved_fill()
                    incr_cell_holder.comment = _resolved_comment()
            else:
                incr_cell_holder.value = f"={mass_ref}*{prob_ref}*{incr_expr}"
            contribution_refs.append(incr_cell_holder.coordinate)
            col += 1

        raw_formula = "=" + "+".join(contribution_refs) if contribution_refs else "0"
        ws.cell(row=r, column=total_raw_col, value=raw_formula)
        raw_coord = ws.cell(row=r, column=total_raw_col).coordinate
        ws.cell(row=r, column=total_raw_col + 1, value=f"={raw_coord}/(1+{discount_rate})^{cycle}")
        r += 1

    final_row = None
    if "final" in rewards["per_cycle"]:
        # The half-cycle correction's own "no terminal blend" reward,
        # evaluated on the distribution AFTER the last transition (see
        # module docstring and _write_state_rewards_sheet). No transition
        # increments here -- reward()'s apply_terminal=False branch never
        # computes them either.
        ws.cell(row=r, column=1, value="Final (no terminal blend)")
        mass_row = by_tenure["data_row_of_cycle"][model.cycles]
        cyc_rewards = rewards["per_cycle"]["final"]
        contribution_refs = []
        for j, unit in enumerate(units):
            mass_ref = f"'{by_tenure['sheet_name']}'!${trace_col[unit]}${mass_row}"
            rate_ref = f"'{rewards['sheet_name']}'!{rate_coord(cyc_rewards, unit)}"
            cell = ws.cell(row=r, column=2 + j, value=f"={mass_ref}*{rate_ref}")
            contribution_refs.append(cell.coordinate)
        raw_formula = "=" + "+".join(contribution_refs) if contribution_refs else "0"
        ws.cell(row=r, column=total_raw_col, value=raw_formula)
        raw_coord = ws.cell(row=r, column=total_raw_col).coordinate
        ws.cell(row=r, column=total_raw_col + 1, value=f"={raw_coord}/(1+{discount_rate})^{model.cycles}")
        final_row = r
        r += 1

    _autosize(ws, len(headers))
    return {
        "sheet_name": ws.title, "first_data_row": 2, "last_data_row": r - 1 - (1 if final_row else 0),
        "final_row": final_row, "raw_col": total_raw_col, "discounted_col": total_raw_col + 1,
    }


# -- Results ------------------------------------------------------------


def _reward_sheet_total_formula(info: Mapping[str, Any]) -> str:
    """A live formula for one reward sheet's grand total: the discounted
    column summed over its regular per-cycle rows, plus the half-cycle
    correction term when that sheet has a "final" row -- written out
    explicitly (not folded into one opaque number), so the correction term is
    never hidden inside a single sum."""

    from openpyxl.utils import get_column_letter

    col = get_column_letter(info["discounted_col"])
    disc_range = f"'{info['sheet_name']}'!${col}${info['first_data_row']}:${col}${info['last_data_row']}"
    base = f"SUM({disc_range})"
    if info.get("final_row"):
        final_coord = f"'{info['sheet_name']}'!${col}${info['final_row']}"
        cycle0_coord = f"'{info['sheet_name']}'!${col}${info['first_data_row']}"
        return f"={base}+0.5*({final_coord}-{cycle0_coord})"
    return f"={base}"


def _write_results_sheet(
    wb: Any,
    model: MarkovModel,
    cost_info: Mapping[str, Any],
    effect_info: Mapping[str, Any],
    extra_info: Mapping[str, Mapping[str, Any]],
    extra_labels: Mapping[str, str],
) -> None:
    ws = wb.create_sheet("Results")
    ws.append(["Metric", "Value"])
    _style_header_row(ws, 1, 2)

    r = 2
    ws.cell(row=r, column=1, value="Total cost")
    ws.cell(row=r, column=2, value=_reward_sheet_total_formula(cost_info))
    r += 1
    ws.cell(row=r, column=1, value="Total effect")
    ws.cell(row=r, column=2, value=_reward_sheet_total_formula(effect_info))
    r += 1
    for oid, info in extra_info.items():
        ws.cell(row=r, column=1, value=f"Total {extra_labels.get(oid, oid)}")
        ws.cell(row=r, column=2, value=_reward_sheet_total_formula(info))
        r += 1

    ws.cell(row=r, column=1, value="Half-cycle correction applied")
    ws.cell(row=r, column=2, value="Yes" if (model.half_cycle_correction and model.cycles > 0) else "No")
    r += 1
    ws.cell(row=r, column=1, value="Number of cycles")
    ws.cell(row=r, column=2, value=model.cycles)

    _autosize(ws, 2)


# -- Build-time self-check ---------------------------------------------------


def _self_check(
    model: MarkovModel,
    units: Sequence[Unit],
    parameters: Mapping[str, Param],
    result: MarkovResult,
    matrix_info: Mapping[str, Any],
    rewards_info: Mapping[str, Any],
    discount_cost_per_cycle: float,
    discount_effect_per_cycle: float,
    extra_discount_per_cycle: Mapping[str, float],
    variables: Mapping[str, Any] | None,
    distributions: Mapping[str, Mapping[str, Any]] | None,
    tables: Mapping[str, Mapping[str, Any]] | None,
    tolerance: float = 1e-6,
) -> None:
    """Recompute total cost/effect/extra-outcomes from the SAME resolved
    numbers written into the State Rewards sheet, applied to
    ``result.tenure_trace`` (the engine's own trusted per-(state,tenure)
    occupancy), and compare to ``result``'s own totals (independently
    computed by ``run_markov()``'s ``reward()``/``advance()``). Raises
    ``MarkovExportError`` -- blocking the export -- on any mismatch beyond
    ``tolerance``. See the module docstring for why this, rather than the
    Excel formulas themselves, is where the build-time
    cross-check (block export on mismatch) lives."""

    def cycle_total(reward_key: str, cycle: int, table: Mapping[str, Any]) -> float:
        mass = result.tenure_trace[cycle]
        base = sum(mass.get(u, 0.0) * table["per_cycle"][cycle][reward_key][u] for u in units)
        return base

    def transition_increments(attr: str, cycle: int, extra_oid: str | None = None) -> float:
        mass = result.tenure_trace[cycle]
        prob_value = matrix_info["per_cycle"][cycle]["prob_value"]
        total = 0.0
        states_by_id = {state.id: state for state in model.states}
        for transition in model.transitions:
            if extra_oid is None:
                incr_raw = getattr(transition, attr)
            else:
                incr_raw = transition.extra_increment.get(extra_oid)
            if incr_raw in (None, 0, 0.0):
                continue
            state = states_by_id[transition.from_state]
            tenures = range(1, max(1, state.tenure_count) + 1) if state.tenure_applied else [1]
            for tenure in tenures:
                source = (transition.from_state, tenure)
                target = _next_unit(state, tenure, transition.to_state)
                probability = prob_value.get((source, target), 0.0)
                if probability == 0.0:
                    continue
                weight = mass.get(source, 0.0) * probability
                if weight == 0.0:
                    continue
                incr_val = _base_resolve(
                    incr_raw, parameters, _engine_variables(cycle, tenure, variables), distributions, tables,
                )
                total += weight * incr_val
        return total

    own_cost_per_cycle: list[float] = []
    own_effect_per_cycle: list[float] = []
    own_extra_per_cycle: dict[str, list[float]] = {oid: [] for oid in extra_discount_per_cycle}
    for t in range(model.cycles):
        raw_cost = cycle_total("cost_value", t, rewards_info) + transition_increments("cost_increment", t)
        raw_effect = cycle_total("effect_value", t, rewards_info) + transition_increments("effect_increment", t)
        own_cost_per_cycle.append(raw_cost / (1.0 + discount_cost_per_cycle) ** t)
        own_effect_per_cycle.append(raw_effect / (1.0 + discount_effect_per_cycle) ** t)
        mass = result.tenure_trace[t]
        for oid in extra_discount_per_cycle:
            raw_extra = sum(
                mass.get(u, 0.0) * rewards_info["per_cycle"][t]["extra_value"][u].get(oid, 0.0) for u in units
            )
            raw_extra += transition_increments("extra_increment", t, extra_oid=oid)
            own_extra_per_cycle[oid].append(raw_extra / (1.0 + extra_discount_per_cycle[oid]) ** t)

    own_total_cost = sum(own_cost_per_cycle)
    own_total_effect = sum(own_effect_per_cycle)
    own_extra_totals = {oid: sum(values) for oid, values in own_extra_per_cycle.items()}

    if model.half_cycle_correction and model.cycles > 0:
        mass_final = result.tenure_trace[model.cycles]
        final_raw_cost = sum(mass_final.get(u, 0.0) * rewards_info["per_cycle"]["final"]["cost_value"][u] for u in units)
        final_raw_effect = sum(mass_final.get(u, 0.0) * rewards_info["per_cycle"]["final"]["effect_value"][u] for u in units)
        own_total_cost += 0.5 * (final_raw_cost / (1.0 + discount_cost_per_cycle) ** model.cycles - own_cost_per_cycle[0])
        own_total_effect += 0.5 * (final_raw_effect / (1.0 + discount_effect_per_cycle) ** model.cycles - own_effect_per_cycle[0])
        for oid in extra_discount_per_cycle:
            final_raw_extra = sum(
                mass_final.get(u, 0.0) * rewards_info["per_cycle"]["final"]["extra_value"][u].get(oid, 0.0)
                for u in units
            )
            own_extra_totals[oid] += 0.5 * (
                final_raw_extra / (1.0 + extra_discount_per_cycle[oid]) ** model.cycles - own_extra_per_cycle[oid][0]
            )

    def close(a: float, b: float) -> bool:
        return abs(a - b) <= tolerance * max(1.0, abs(b))

    mismatches: list[str] = []
    if not close(own_total_cost, result.total_cost):
        mismatches.append(f"total cost: exporter={own_total_cost!r} vs engine={result.total_cost!r}")
    if not close(own_total_effect, result.total_effect):
        mismatches.append(f"total effect: exporter={own_total_effect!r} vs engine={result.total_effect!r}")
    for oid, own_value in own_extra_totals.items():
        engine_value = result.extra_totals.get(oid, 0.0)
        if not close(own_value, engine_value):
            mismatches.append(f"extra outcome '{oid}': exporter={own_value!r} vs engine={engine_value!r}")

    if mismatches:
        raise MarkovExportError(
            "Markov transparency export self-check failed -- the exported workbook's own numbers "
            "would not match this model's Run results, so the export was blocked rather than shipped "
            "wrong: " + "; ".join(mismatches)
        )


# -- Top-level orchestrator ---------------------------------------------------


def build_markov_transparency_workbook(
    model: MarkovModel,
    parameters: Mapping[str, Param],
    result: MarkovResult,
    *,
    discount_cost_per_cycle: float = 0.0,
    discount_effect_per_cycle: float = 0.0,
    model_name: str = "",
    engine_version: str = "",
    variables: Mapping[str, Any] | None = None,
    distributions: Mapping[str, Mapping[str, Any]] | None = None,
    tables: Mapping[str, Mapping[str, Any]] | None = None,
    extra_outcome_labels: Mapping[str, str] | None = None,
) -> bytes:
    """Build the transparency Excel workbook for one cohort Markov model
    (one strategy -- see ``build_markov_strategies_workbook`` for a
    multi-strategy comparison with an ICER sheet, mirroring
    ``tree_excel_export.build_tree_transparency_workbook``).

    ``model``/``parameters``/``result`` are exactly what the app already
    has in hand after a Run: ``result`` must be `run_markov(model, ...)`'s
    own return value (needed for ``tenure_trace`` and the totals this
    function cross-checks itself against -- see module docstring), run
    with the SAME ``discount_cost_per_cycle``/``discount_effect_per_cycle``
    passed here.
    """

    import openpyxl

    if result.errors:
        raise MarkovExportError(f"Cannot export a model with validation errors: {result.errors}")

    extra_outcome_labels = extra_outcome_labels or {}
    extra_outcome_ids = [row["id"] for row in model.extra_outcomes]
    extra_outcome_types = {row["id"]: row["type"] for row in model.extra_outcomes}
    extra_discount_per_cycle = {
        oid: (discount_cost_per_cycle if extra_outcome_types[oid] == "cost" else discount_effect_per_cycle)
        for oid in extra_outcome_ids
    }

    states_by_id = {state.id: state for state in model.states}
    units = _tenure_units(model)

    wb = openpyxl.Workbook()
    param_info = _write_parameters_sheet(wb, parameters)
    param_defined_names = {name: info["defined_name"] for name, info in param_info.items()}

    matrix_info = _write_transition_matrix_sheet(
        wb, model, units, states_by_id, parameters, param_defined_names, variables, distributions, tables,
    )
    by_tenure = _write_cohort_trace_by_tenure_sheet(
        wb, model, units, states_by_id, parameters, param_defined_names, matrix_info, variables, distributions, tables,
    )
    _write_cohort_trace_by_state_sheet(wb, model, units, by_tenure)
    rewards_info = _write_state_rewards_sheet(
        wb, model, units, states_by_id, parameters, param_defined_names, matrix_info,
        extra_outcome_ids, extra_outcome_labels, variables, distributions, tables,
    )

    cost_info = _write_reward_sheet(
        wb, "Cost", model, units, by_tenure, rewards_info, lambda cyc, u: cyc["cost_cell"][u],
        discount_cost_per_cycle, lambda t: t.cost_increment, param_defined_names, parameters,
        variables, distributions, tables, matrix_info,
    )
    effect_info = _write_reward_sheet(
        wb, "Effect", model, units, by_tenure, rewards_info, lambda cyc, u: cyc["effect_cell"][u],
        discount_effect_per_cycle, lambda t: t.effect_increment, param_defined_names, parameters,
        variables, distributions, tables, matrix_info,
    )
    extra_rewards_info: dict[str, dict[str, Any]] = {}
    used_sheet_names = {"Parameters", "Transition Matrix", "Cohort Trace (state x tenure)",
                         "Cohort Trace (by state)", "State Rewards", "Cost", "Effect"}
    for oid in extra_outcome_ids:
        sheet_name = _safe_extra_sheet_name(extra_outcome_labels.get(oid, oid), used_sheet_names)
        extra_rewards_info[oid] = _write_reward_sheet(
            wb, sheet_name, model, units, by_tenure, rewards_info,
            (lambda cyc, u, _oid=oid: cyc["extra_cell"][u][_oid]),
            extra_discount_per_cycle[oid], (lambda t, _oid=oid: t.extra_increment.get(_oid)),
            param_defined_names, parameters, variables, distributions, tables, matrix_info,
        )

    _self_check(
        model, units, parameters, result, matrix_info, rewards_info,
        discount_cost_per_cycle, discount_effect_per_cycle, extra_discount_per_cycle,
        variables, distributions, tables,
    )

    _write_results_sheet(wb, model, cost_info, effect_info, extra_rewards_info, extra_outcome_labels)

    config_hash = _config_hash(parameters, model)
    _write_markov_cover_sheet(wb, model_name, engine_version, config_hash, ai_statement=True)
    wb.active = 0

    buffer = io.BytesIO()
    wb.save(buffer)
    return buffer.getvalue()
