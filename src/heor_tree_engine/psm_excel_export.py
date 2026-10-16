"""Live-formula sheets for Partitioned Survival (PSM) nodes in the transparency workbook.

One sheet per PSM node ("PSM <node id>"), built from `psm_trace.compute_psm_trace` (the engine's own per-cycle
replay).  Everything the engine does on top of the parameters is an Excel formula:

  * the PFS / OS survival curve at every cycle (closed forms for exponential, Weibull, log-normal, log-logistic,
    Gompertz; generalized gamma through GAMMADIST with the Stacy conversion used by `des_sampling`),
  * state occupancy (pre-progression = PFS, post-progression = max(0, OS - PFS), dead = 1 - OS),
  * each role's per-cycle cost / effect rate -- a live translation of the branch + terminal expression
    (`excel_expr`, including `_cycle`), or, only when an expression has no Excel equivalent, a resolved
    (amber, commented) value per cycle,
  * per-cycle reward (raw and discounted), the half-cycle correction and the node totals.

The totals are then linked from the "Tree Paths" sheet's leaf cells of every path ending at this PSM node, so the
Strategy Summary / ICER sheets stay live end to end.  The number of cycles is structural (one table row per cycle):
changing "Cycles" in Excel does not add rows -- re-export from DECIDE to change it.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from .excel_expr import expression_to_excel
from .models import PSM_CURVE_TIME_UNITS, PSM_STATE_ROLES, PSMNode, SurvivalCurve
from .psm_trace import PSMTrace, psm_branches_by_role

_HEADER_FILL = "1E3A8A"
_RESOLVED_FILL = "FFF3CD"
_INPUT_FILL = "E8F0FE"

ROLE_LABELS = {"pre_progression": "Pre-progression", "post_progression": "Post-progression", "dead": "Dead"}
CURVE_PARAM_ORDER = {
    "exponential": ("rate",),
    "weibull": ("shape", "scale"),
    "log_normal": ("meanlog", "sdlog"),
    "log_logistic": ("shape", "scale"),
    "gompertz": ("shape", "rate"),
    "generalized_gamma": ("mu", "sigma", "Q"),
}


class PSMExportError(ValueError):
    """The PSM sheets could not be built, or the Excel replay does not reconcile with the engine."""


def psm_sheet_name(node_id: str, used: set[str]) -> str:
    base = "PSM " + re.sub(r"[\[\]:*?/\\]", "_", str(node_id)).strip()
    base = base[:31] or "PSM"
    name, k = base, 2
    while name.lower() in used:
        suffix = f" {k}"
        name = base[: 31 - len(suffix)] + suffix
        k += 1
    used.add(name.lower())
    return name


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def survival_excel(distribution: str, t: str, p: Mapping[str, str]) -> str:
    """Excel formula body (no '=') for S(t); `t` and `p[name]` are cell references."""

    if distribution == "exponential":
        return f"EXP(-{p['rate']}*{t})"
    if distribution == "weibull":
        return f"IF({t}<=0,1,EXP(-((({t})/{p['scale']})^{p['shape']})))"
    if distribution == "log_normal":
        return f"IF({t}<=0,1,NORMSDIST(-(LN({t})-{p['meanlog']})/{p['sdlog']}))"
    if distribution == "log_logistic":
        return f"IF({t}<=0,1,1/(1+(({t})/{p['scale']})^{p['shape']}))"
    if distribution == "gompertz":
        return f"IF({t}<=0,1,EXP(-({p['rate']}/{p['shape']})*(EXP({p['shape']}*{t})-1)))"
    if distribution == "generalized_gamma":
        if "ln_scale" in p:  # Prentice mu/sigma/Q, converted to Stacy a, c, scale in helper cells
            x = f"(({t})/{p['scale']})^{p['c']}"
            return f"IF({t}<=0,1,IF({p['c']}>0,1-GAMMADIST({x},{p['a']},1,TRUE),GAMMADIST({x},{p['a']},1,TRUE)))"
        if "mu" in p:  # Q == 0: log-normal
            return f"IF({t}<=0,1,NORMSDIST(-(LN({t})-{p['mu']})/{p['sigma']}))"
    raise PSMExportError(f"No Excel survival formula for distribution '{distribution}'")


def _style_header(ws: Any, row: int, n_cols: int, start: int = 1) -> None:
    from openpyxl.styles import Alignment, Font, PatternFill

    for col in range(start, start + n_cols):
        cell = ws.cell(row=row, column=col)
        cell.fill = PatternFill("solid", fgColor=_HEADER_FILL)
        cell.font = Font(bold=True, color="FFFFFF")
        cell.alignment = Alignment(wrap_text=True, vertical="center")


def _resolved_comment(text: str | None = None) -> Any:
    from openpyxl.comments import Comment

    return Comment(
        text or "Resolved value, not a live formula: DECIDE evaluated this from the model's own expression "
        "(it has no Excel equivalent).", "DECIDE",
    )


def _amber(cell: Any, text: str | None = None) -> None:
    from openpyxl.styles import PatternFill

    cell.fill = PatternFill("solid", fgColor=_RESOLVED_FILL)
    cell.comment = _resolved_comment(text)


def _ref_live(raw: Any, param_names: Mapping[str, str]) -> str | None:
    """Excel body for a numeric reference that is a literal or a bare parameter name."""

    if _is_number(raw):
        return repr(float(raw)) if not float(raw).is_integer() else str(int(raw))
    if isinstance(raw, str) and raw.strip() in param_names:
        return param_names[raw.strip()]
    return None


def _write_curve_block(ws: Any, col0: int, label: str, curve: SurvivalCurve, resolved: Mapping[str, float],
                       param_names: Mapping[str, str], first_row: int) -> dict[str, Any]:
    """Curve inputs in columns col0 (label) / col0+1 (value).  Returns {'refs': {...}, 'unit_factor_ref': ...}."""

    from openpyxl.utils import get_column_letter

    lc, vc = get_column_letter(col0), get_column_letter(col0 + 1)
    r = first_row
    ws.cell(row=r, column=col0, value=label)
    _style_header(ws, r, 2, start=col0)
    ws.cell(row=r + 1, column=col0, value="Distribution")
    ws.cell(row=r + 1, column=col0 + 1, value=curve.distribution)
    ws.cell(row=r + 2, column=col0, value="Time unit of the parameters")
    ws.cell(row=r + 2, column=col0 + 1, value=curve.time_unit)
    ws.cell(row=r + 3, column=col0, value="Curve time units per year")
    factor = PSM_CURVE_TIME_UNITS.get(curve.time_unit, 1.0)
    ws.cell(row=r + 3, column=col0 + 1, value=float(factor))
    unit_ref = f"${vc}${r + 3}"

    refs: dict[str, str] = {}
    row = r + 4
    for name in CURVE_PARAM_ORDER[curve.distribution]:
        ws.cell(row=row, column=col0, value=name)
        raw = curve.params.get(name)
        live = _ref_live(raw, param_names)
        if live is not None:
            cell = ws.cell(row=row, column=col0 + 1, value=("=" + live) if not _is_number(raw) else float(raw))
        else:
            cell = ws.cell(row=row, column=col0 + 1, value=float(resolved[name]))
            _amber(cell)
        refs[name] = f"${vc}${row}"
        row += 1
    if curve.distribution == "generalized_gamma":
        q_value = float(resolved["Q"])
        if q_value == 0.0:
            pass  # reduces to log-normal(mu, sigma): survival_excel uses mu/sigma only
        else:
            mu, sigma, q = refs["mu"], refs["sigma"], refs["Q"]
            for label_, formula, key in (
                ("a = 1/Q^2", f"=1/({q})^2", "a"),
                ("c = Q/sigma", f"=({q})/({sigma})", "c"),
                ("ln(scale) = mu + 2 sigma/Q * ln|Q|", f"={mu}+(2*{sigma}/({q}))*LN(ABS({q}))", "ln_scale"),
                ("scale = exp(ln scale)", f"=EXP({vc}{row + 2})", "scale"),
            ):
                ws.cell(row=row, column=col0, value=label_)
                ws.cell(row=row, column=col0 + 1, value=formula)
                refs[key] = f"${vc}${row}"
                row += 1
    return {"refs": refs, "unit_ref": unit_ref, "next_row": row}


def write_psm_sheet(
    wb: Any,
    node: PSMNode,
    trace: PSMTrace,
    param_names: Mapping[str, str],
    sheet_name: str,
) -> dict[str, str]:
    """Write one PSM node's sheet.  Returns the cell references of its four totals (for the Tree Paths links)."""

    from openpyxl.styles import Alignment, Font

    by_role = psm_branches_by_role(node)
    ws = wb.create_sheet(sheet_name)
    n = trace.cycles

    ws["A1"] = f"Partitioned survival node '{node.id}'"
    ws["A1"].font = Font(bold=True, size=14)
    ws["A2"] = ("Blue = input, white = live formula, amber = resolved value (no Excel equivalent; see the cell comment). "
                "The cycle table has one row per cycle: re-export from DECIDE to change the number of cycles.")
    ws["A2"].alignment = Alignment(wrap_text=False)

    setup = [
        ("Cycles", float(n)), ("Cycles per year", float(trace.cycles_per_year)),
        ("Half-cycle correction (1 = on, 0 = off)", 1.0 if trace.half_cycle_correction else 0.0),
        ("Cost discount rate (per year)", float(trace.cost_discount_rate)),
        ("Effect discount rate (per year)", float(trace.effect_discount_rate)),
    ]
    from openpyxl.styles import PatternFill

    for i, (label, value) in enumerate(setup):
        ws.cell(row=4 + i, column=1, value=label)
        c = ws.cell(row=4 + i, column=2, value=value)
        c.fill = PatternFill("solid", fgColor=_INPUT_FILL)
    cpy_ref, hcc_ref, rc_ref, re_ref = "$B$5", "$B$6", "$B$7", "$B$8"

    pfs = _write_curve_block(ws, 1, "PFS curve", node.pfs_curve, trace.curve_params["pfs"], param_names, 10)
    os_ = _write_curve_block(ws, 4, "OS curve", node.os_curve, trace.curve_params["os"], param_names, 10)
    table_header = max(pfs["next_row"], os_["next_row"]) + 2
    first = table_header + 1
    last_body = first + n - 1
    endpoint = first + n

    headers = ["Cycle", "Time (years)", "PFS curve time", "OS curve time", "S_PFS(t)", "S_OS(t)",
               "Occupancy: Pre-progression", "Occupancy: Post-progression", "Occupancy: Dead",
               "Cost rate: Pre", "Cost rate: Post", "Cost rate: Dead",
               "Effect rate: Pre", "Effect rate: Post", "Effect rate: Dead",
               "Cost (undiscounted)", "Effect (undiscounted)", "Cost (discounted)", "Effect (discounted)"]
    for i, h in enumerate(headers, start=1):
        ws.cell(row=table_header, column=i, value=h)
    _style_header(ws, table_header, len(headers))

    roles = list(PSM_STATE_ROLES)
    translate_failed: set[tuple[str, str]] = set()
    for k, row_data in enumerate(trace.rows):
        r = first + k
        cyc = f"$A{r}"
        ws.cell(row=r, column=1, value=k)
        ws.cell(row=r, column=2, value=f"=A{r}/{cpy_ref}")
        ws.cell(row=r, column=3, value=f"=B{r}*{pfs['unit_ref']}")
        ws.cell(row=r, column=4, value=f"=B{r}*{os_['unit_ref']}")
        ws.cell(row=r, column=5, value="=" + survival_excel(node.pfs_curve.distribution, f"C{r}", pfs["refs"]))
        ws.cell(row=r, column=6, value="=" + survival_excel(node.os_curve.distribution, f"D{r}", os_["refs"]))
        ws.cell(row=r, column=7, value=f"=E{r}")
        ws.cell(row=r, column=8, value=f"=MAX(0,F{r}-E{r})")
        ws.cell(row=r, column=9, value=f"=1-F{r}")
        col = 10
        for outcome in ("cost", "effect"):
            for role in roles:
                branch = by_role[role]
                inc = branch.cost_increment if outcome == "cost" else branch.effect_increment
                leaf = branch.child.cost if outcome == "cost" else branch.child.effect
                parts: list[str] = []
                for raw in (inc, leaf):
                    if raw is None:
                        continue
                    if _is_number(raw):
                        parts.append(_ref_live(raw, param_names))
                    elif isinstance(raw, str):
                        body = expression_to_excel(raw, param_names, cyc)
                        if body is None:
                            parts = []
                            translate_failed.add((outcome, role))
                            break
                        parts.append(body)
                    else:
                        parts.append("0")
                if (outcome, role) in translate_failed:
                    cell = ws.cell(row=r, column=col, value=float(row_data[f"rate_{outcome}_{role}"]))
                    _amber(cell, "Resolved value, not a live formula: this per-cycle rate is built from an expression "
                                 "DECIDE evaluated itself (no Excel equivalent for part of it).")
                else:
                    ws.cell(row=r, column=col, value="=" + ("+".join(parts) if parts else "0"))
                col += 1
        ws.cell(row=r, column=16, value=f"=G{r}*J{r}+H{r}*K{r}+I{r}*L{r}")
        ws.cell(row=r, column=17, value=f"=G{r}*M{r}+H{r}*N{r}+I{r}*O{r}")
        ws.cell(row=r, column=18, value=f"=P{r}/(1+{rc_ref})^B{r}")
        ws.cell(row=r, column=19, value=f"=Q{r}/(1+{re_ref})^B{r}")
    ws.cell(row=endpoint, column=20,
            value="End-of-horizon point: used only by the half-cycle correction (not a reward period)")

    # Totals (E4:E7), next to the setup block
    total_specs = [
        ("Total cost - undiscounted", "P"), ("Total cost - discounted", "R"),
        ("Total effect - undiscounted", "Q"), ("Total effect - discounted", "S"),
    ]
    refs: dict[str, str] = {}
    quoted = "'" + sheet_name.replace("'", "''") + "'"
    for i, (label, colx) in enumerate(total_specs):
        ws.cell(row=4 + i, column=4, value=label).font = Font(bold=True)
        ws.cell(
            row=4 + i, column=5,
            value=f"=SUM({colx}{first}:{colx}{last_body})+{hcc_ref}*0.5*({colx}{endpoint}-{colx}{first})",
        ).font = Font(bold=True)
    refs = {
        "cost_undiscounted": f"{quoted}!$E$4", "cost_discounted": f"{quoted}!$E$5",
        "effect_undiscounted": f"{quoted}!$E$6", "effect_discounted": f"{quoted}!$E$7",
    }
    if translate_failed:
        ws["A3"] = ("Note: " + ", ".join(f"{o} rate ({ROLE_LABELS[r]})" for o, r in sorted(translate_failed))
                    + " use resolved per-cycle values (amber): the rest of the sheet is live.")
    from openpyxl.utils import get_column_letter

    for i in range(1, len(headers) + 1):
        ws.column_dimensions[get_column_letter(i)].width = 16 if i > 1 else 36
    ws.column_dimensions["D"].width = 30
    ws.freeze_panes = ws.cell(row=table_header + 1, column=2)
    return refs
