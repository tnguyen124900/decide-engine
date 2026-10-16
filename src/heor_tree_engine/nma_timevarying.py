"""Time-varying hazard ratios in NMA (C1a, C1b): piecewise HR network meta-analysis and proportional-hazards checks.

Input rows (one per study x time interval x contrast): study, treatment, versus, t_from, t_to and either
log_hr + se, or hr + lower + upper (95% CI, natural scale); `baseline_se` (SE of the log hazard in the shared
comparator arm) is only needed for a multi-arm study that reports two or more contrasts in the same interval.

C1b: each interval gets its own contrast-based NMA (GLS, as nma_freq): d[k, interval] and tau^2[interval]. The
intervals of one study are treated as independent (an approximation: they share patients); state this whenever
the results are reported.
C1a: the proportional-hazards assumption predicts one log HR over all intervals. For a study with >= 2 intervals
for the same contrast, and for each treatment across the network estimates, Cochran's Q of the interval log HRs
(inverse-variance weights, independence assumed) on (intervals - 1) df is reported: a small p-value suggests the
HR changes over time. A test with low power: absence of a flag is not evidence of PH."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .models import ValidationError
from .nma_freq import chi2_sf, run_nma_frequentist
from .nma_parser import resolve_nma


@dataclass
class IntervalEstimate:
    treatment: str
    t_from: float
    t_to: float
    hr: float
    lower: float
    upper: float
    log_hr: float
    se: float
    n_studies: int
    tau2: float


@dataclass
class PHRow:
    scope: str  # "study" or "network"
    label: str  # "study: B vs A" or treatment
    n_intervals: int
    q: float
    df: int
    p: float | None
    flagged: bool  # p < 0.10


@dataclass
class PiecewiseResult:
    reference: str | None = None
    effects: str = "random"
    intervals: list[tuple[float, float]] = field(default_factory=list)
    estimates: list[IntervalEstimate] = field(default_factory=list)
    ph: list[PHRow] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[ValidationError] = field(default_factory=list)


def normalize_rows(rows) -> tuple[list[dict], list[str]]:
    """Clean rows; returns (rows, problems). Problem rows are skipped."""

    clean, problems = [], []
    for n, raw in enumerate(rows, start=1):
        try:
            study = str(raw.get("study", "")).strip()
            treatment, versus = str(raw.get("treatment", "")).strip(), str(raw.get("versus", "")).strip()
            t0, t1 = float(raw["t_from"]), float(raw["t_to"])
            if not (study and treatment and versus) or treatment == versus or not t1 > t0 >= 0:
                raise ValueError("incomplete row")
            log_hr, se = raw.get("log_hr"), raw.get("se")
            if log_hr in (None, "") or se in (None, "") or (isinstance(log_hr, float) and math.isnan(log_hr)):
                hr, lo, up = float(raw["hr"]), float(raw["lower"]), float(raw["upper"])
                if min(hr, lo, up) <= 0 or not lo < up:
                    raise ValueError("bad CI")
                log_hr, se = math.log(hr), (math.log(up) - math.log(lo)) / (2 * 1.959963984540054)
            log_hr, se = float(log_hr), float(se)
            if not se > 0:
                raise ValueError("se must be positive")
            base = raw.get("baseline_se")
            base = None if base in (None, "") or (isinstance(base, float) and math.isnan(base)) else float(base)
            clean.append({"study": study, "treatment": treatment, "versus": versus, "t_from": t0, "t_to": t1,
                          "log_hr": log_hr, "se": se, "baseline_se": base})
        except (KeyError, TypeError, ValueError):
            if any(str(v).strip() not in ("", "None", "nan") for v in raw.values()):
                problems.append(f"Row {n}: skipped (incomplete or invalid)")
    return clean, problems


def _q(estimates: list[tuple[float, float]]) -> tuple[float, int, float | None]:
    w = [1.0 / se ** 2 for _, se in estimates]
    mean = sum(wi * y for wi, (y, _) in zip(w, estimates)) / sum(w)
    q = sum(wi * (y - mean) ** 2 for wi, (y, _) in zip(w, estimates))
    df = len(estimates) - 1
    return q, df, (chi2_sf(q, df) if df > 0 else None)


def run_piecewise_nma(rows, reference: str | None = None, effects: str = "random",
                      small_values: str = "bad") -> PiecewiseResult:
    out = PiecewiseResult(effects=effects)
    data, problems = normalize_rows(rows)
    out.warnings.extend(problems)
    if not data:
        out.errors.append(ValidationError(code="no_rows", message="No valid rows: enter study, treatment, versus, interval and HR"))
        return out
    reference = reference or data[0]["versus"]
    out.reference = reference
    intervals = sorted({(r["t_from"], r["t_to"]) for r in data})
    out.intervals = intervals
    names = []
    for r in data:
        for name in (r["versus"], r["treatment"]):
            if name not in names:
                names.append(name)
    if reference not in names:
        out.errors.append(ValidationError(code="reference", message=f"Reference '{reference}' does not occur in the data"))
        return out
    for t0, t1 in intervals:
        subset = [r for r in data if (r["t_from"], r["t_to"]) == (t0, t1)]
        studies: dict[str, list[dict]] = {}
        for r in subset:
            studies.setdefault(r["study"], []).append(r)
        docs, skip = [], False
        for sid, items in studies.items():
            entry = {"id": sid, "contrasts": [{"treatment": r["treatment"], "versus": r["versus"],
                                               "estimate": r["log_hr"], "se": r["se"]} for r in items]}
            if len(items) > 1:
                bases = {r["baseline_se"] for r in items if r["baseline_se"] is not None}
                versus = {r["versus"] for r in items}
                if not bases or len(versus) > 1:
                    out.warnings.append(f"{sid} ({t0:g}-{t1:g}): a multi-arm study needs one shared comparator and baseline_se; skipped")
                    continue
                entry["baseline_se"] = min(bases)
            docs.append(entry)
        used = sorted({n for d in docs for c in d["contrasts"] for n in (c["treatment"], c["versus"])},
                      key=names.index)
        if len(docs) < 1 or reference not in used:
            out.warnings.append(f"Interval {t0:g}-{t1:g}: the reference is not in this interval's studies; not estimated")
            continue
        try:
            model = resolve_nma({"spec": {"outcome": "contrast", "measure": "HR", "reference": reference,
                                          "small_values": small_values},
                                 "treatments": [{"name": n} for n in used], "studies": docs})
            result = run_nma_frequentist(model)
        except ValueError as exc:
            out.warnings.append(f"Interval {t0:g}-{t1:g}: {exc}")
            continue
        if result.errors:
            out.warnings.append(f"Interval {t0:g}-{t1:g}: " + "; ".join(e.message for e in result.errors[:2]))
            continue
        fit = result.random if effects == "random" else result.fixed
        for e in fit.effects:
            out.estimates.append(IntervalEstimate(e.treatment, t0, t1, e.natural, e.natural_lower, e.natural_upper,
                                                  e.estimate, e.se, len(docs), fit.tau2))
    if not out.estimates:
        out.errors.append(ValidationError(code="not_estimated", message="No interval could be analysed"))
        return out
    # C1a: proportional-hazards checks
    groups: dict[tuple[str, str, str], list[tuple[float, float]]] = {}
    for r in data:
        groups.setdefault((r["study"], r["treatment"], r["versus"]), []).append((r["log_hr"], r["se"]))
    for (study, a, b), values in groups.items():
        if len(values) >= 2:
            q, df, p = _q(values)
            out.ph.append(PHRow("study", f"{study}: {a} vs {b}", len(values), q, df, p, p is not None and p < 0.10))
    for name in names:
        values = [(e.log_hr, e.se) for e in out.estimates if e.treatment == name]
        if len(values) >= 2:
            q, df, p = _q(values)
            out.ph.append(PHRow("network", name, len(values), q, df, p, p is not None and p < 0.10))
    return out
