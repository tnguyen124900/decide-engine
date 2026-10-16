"""Extrapolation, plausibility checks and a candidate-curve library for PSM survival curves.

Works on the output of `psm_km_fit` (KMData + FitResult) -- everything here is pure numpy/math, no Streamlit.

* `extrapolate`      -- per fitted distribution: S(t) at landmark years, median, restricted mean survival time over the
                        observed follow-up and over the model horizon, and how much of the horizon mean lies beyond the data
                        (the "extrapolation share" NICE DSU TSD 14/19 asks to be reported and justified).
* `hazard_curve`     -- h(t) = -d ln S / dt for a fit (the shape plot used to judge plausibility).
* `plausibility_checks` -- flags: long-run survival above the general population, PFS curve above OS, a large share of the
                        mean beyond follow-up, a non-converged fit, a Gompertz plateau (negative shape, never-ending survival).
* `background_adjusted_survival` -- S(t) with h(t) = max(h_fit(t), h_background(t)) after a cut-off age/time, plus its mean.
* library helpers    -- `library_add / library_select / library_remove` keep several fitted candidates per PFS/OS curve
                        dict (`curve["library"]`) with a name and a justification note, and copy a chosen one into the
                        curve (`distribution`, `params`, `time_unit`).

All times in the public API are YEARS unless a name says otherwise; a curve's own parameters stay in its `time_unit`
(`models.PSM_CURVE_TIME_UNITS` = how many of that unit fit in one year).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .models import PSM_CURVE_TIME_UNITS
from .psm_km_fit import FitResult, KMData, km_steps, survival as _native_survival

DEFAULT_LANDMARKS_YEARS = (1.0, 2.0, 5.0, 10.0)


def _factor(unit: str) -> float:
    return PSM_CURVE_TIME_UNITS.get(unit, 1.0)


def survival_years(distribution: str, params: dict[str, float], unit: str, t_years) -> np.ndarray:
    """S at times given in YEARS, for parameters stored in `unit`."""

    return _native_survival(distribution, params, np.asarray(t_years, dtype=float) * _factor(unit))


def hazard_curve(distribution: str, params: dict[str, float], unit: str, t_years) -> np.ndarray:
    """Hazard per YEAR, -d ln S/dt by a central difference on ln S (accurate to ~1e-6 relative for smooth curves)."""

    t = np.asarray(t_years, dtype=float)
    h = np.maximum(1e-6, 1e-4 * np.maximum(t, 1e-3))
    lo = np.maximum(t - h, 0.0)
    s_hi = np.maximum(survival_years(distribution, params, unit, t + h), 1e-300)
    s_lo = np.maximum(survival_years(distribution, params, unit, lo), 1e-300)
    return -(np.log(s_hi) - np.log(s_lo)) / (t + h - lo)


def _grid(horizon_years: float, n: int = 4001) -> np.ndarray:
    return np.linspace(0.0, float(horizon_years), n)


def _simpson(y: np.ndarray, x: np.ndarray) -> float:
    h = x[1] - x[0]
    w = np.ones(len(x))
    w[1:-1:2] = 4.0
    w[2:-1:2] = 2.0
    return float(h / 3.0 * np.sum(w * y))


def restricted_mean_years(distribution: str, params: dict[str, float], unit: str, horizon_years: float) -> float:
    """Area under S(t) from 0 to horizon (years)."""

    if horizon_years <= 0:
        return 0.0
    x = _grid(horizon_years)  # odd number of points -> Simpson-valid
    return _simpson(survival_years(distribution, params, unit, x), x)


def median_years(distribution: str, params: dict[str, float], unit: str, horizon_years: float = 100.0) -> float | None:
    """First time S(t) <= 0.5 within `horizon_years`, or None if the curve never gets there."""

    x = _grid(horizon_years, 20001)
    s = survival_years(distribution, params, unit, x)
    below = np.nonzero(s <= 0.5)[0]
    if not len(below):
        return None
    i = int(below[0])
    if i == 0:
        return 0.0
    x0, x1, s0, s1 = x[i - 1], x[i], s[i - 1], s[i]
    return float(x0 + (0.5 - s0) * (x1 - x0) / (s1 - s0)) if s1 != s0 else float(x1)


def km_restricted_mean_years(data: KMData, unit: str, tau_years: float | None = None) -> float:
    """Area under the Kaplan-Meier step curve up to `tau` (default: last observed time), in years."""

    steps = km_steps(data)
    f = _factor(unit)
    tau = (tau_years * f) if tau_years is not None else float(np.max(data.times))
    area, t_prev, s_prev = 0.0, 0.0, 1.0
    for t, s in steps:
        if t > tau:
            break
        area += s_prev * (t - t_prev)
        t_prev, s_prev = t, s
    area += s_prev * max(0.0, tau - t_prev)
    return area / f


@dataclass
class ExtrapolationRow:
    distribution: str
    params: dict[str, float]
    aic: float
    bic: float
    survival_at: dict[float, float] = field(default_factory=dict)  # landmark year -> S
    median_years: float | None = None
    rmst_followup_years: float = 0.0
    rmst_horizon_years: float = 0.0
    extrapolation_share: float = 0.0  # fraction of the horizon mean that lies beyond the observed follow-up
    survival_at_horizon: float = 0.0
    name: str = ""  # label shown in tables/plots (defaults to the distribution)
    unit: str = ""  # the unit this row's params are in ("" = the summary's unit)
    converged: bool = True


@dataclass
class ExtrapolationSummary:
    unit: str
    followup_years: float
    horizon_years: float
    km_rmst_followup_years: float
    km_survival_at_followup: float
    rows: list[ExtrapolationRow] = field(default_factory=list)


def extrapolate(
    data: KMData, fits: list[FitResult], unit: str, horizon_years: float,
    landmarks: tuple[float, ...] = DEFAULT_LANDMARKS_YEARS,
) -> ExtrapolationSummary:
    """Extrapolation table for every fit that has parameters (order of `fits` kept)."""

    f = _factor(unit)
    followup = float(np.max(data.times)) / f
    steps = km_steps(data)
    summary = ExtrapolationSummary(
        unit=unit, followup_years=followup, horizon_years=float(horizon_years),
        km_rmst_followup_years=km_restricted_mean_years(data, unit),
        km_survival_at_followup=float(steps[-1][1]) if steps else 1.0,
    )
    for fit in fits:
        if not fit.params:
            continue
        sv = {float(t): float(survival_years(fit.distribution, fit.params, unit, [t])[0]) for t in landmarks}
        rm_f = restricted_mean_years(fit.distribution, fit.params, unit, min(followup, horizon_years))
        rm_h = restricted_mean_years(fit.distribution, fit.params, unit, horizon_years)
        summary.rows.append(ExtrapolationRow(
            distribution=fit.distribution, params=dict(fit.params), aic=fit.aic, bic=fit.bic, survival_at=sv,
            median_years=median_years(fit.distribution, fit.params, unit),
            rmst_followup_years=rm_f, rmst_horizon_years=rm_h,
            extrapolation_share=(rm_h - rm_f) / rm_h if rm_h > 0 else 0.0,
            survival_at_horizon=float(survival_years(fit.distribution, fit.params, unit, [horizon_years])[0]),
        ))
    return summary


@dataclass
class Candidate:
    """A curve to extrapolate: a fit, a library entry or the curve currently in the model (params in `unit`)."""

    name: str
    distribution: str
    params: dict[str, float]
    unit: str = "year"
    aic: float = float("nan")
    bic: float = float("nan")
    converged: bool = True


def km_steps_rmst_years(steps: list[tuple[float, float]], unit: str) -> tuple[float, float, float]:
    """(follow-up in years, area under the KM steps in years, last KM survival) from stored `[time, survival]`
    steps in `unit` (what `curve["fit_info"]["km"]` holds)."""

    f = _factor(unit)
    pts = [(float(t), float(s)) for t, s in steps if float(t) > 0]
    if not pts:
        return 0.0, 0.0, 1.0
    tau = max(t for t, _ in pts)
    area, t_prev, s_prev = 0.0, 0.0, 1.0
    for t, s in sorted(pts):
        area += s_prev * (t - t_prev)
        t_prev, s_prev = t, s
    return tau / f, area / f, s_prev


def extrapolate_candidates(
    candidates: list[Candidate], horizon_years: float, followup_years: float,
    km_rmst_followup_years: float = float("nan"), km_survival_at_followup: float = float("nan"),
    landmarks: tuple[float, ...] = DEFAULT_LANDMARKS_YEARS,
) -> ExtrapolationSummary:
    """Same table as `extrapolate`, but for arbitrary candidates (each with its own time unit)."""

    summary = ExtrapolationSummary(
        unit="year", followup_years=float(followup_years), horizon_years=float(horizon_years),
        km_rmst_followup_years=km_rmst_followup_years, km_survival_at_followup=km_survival_at_followup,
    )
    for c in candidates:
        if not c.params:
            continue
        sv = {float(t): float(survival_years(c.distribution, c.params, c.unit, [t])[0]) for t in landmarks}
        rm_f = restricted_mean_years(c.distribution, c.params, c.unit, min(followup_years, horizon_years))
        rm_h = restricted_mean_years(c.distribution, c.params, c.unit, horizon_years)
        summary.rows.append(ExtrapolationRow(
            distribution=c.distribution, params=dict(c.params), aic=c.aic, bic=c.bic, survival_at=sv,
            median_years=median_years(c.distribution, c.params, c.unit),
            rmst_followup_years=rm_f, rmst_horizon_years=rm_h,
            extrapolation_share=(rm_h - rm_f) / rm_h if rm_h > 0 else 0.0,
            survival_at_horizon=float(survival_years(c.distribution, c.params, c.unit, [horizon_years])[0]),
            name=c.name, unit=c.unit, converged=c.converged,
        ))
    return summary


# ---------------------------------------------------------------- background mortality

@dataclass
class BackgroundMortality:
    """General-population mortality: `ages[i]` -> annual probability of death `qx[i]` (step function, last value carried)."""

    ages: list[float]
    qx: list[float]

    def hazard(self, age) -> np.ndarray:
        age = np.asarray(age, dtype=float)
        idx = np.clip(np.searchsorted(np.asarray(self.ages), age, side="right") - 1, 0, len(self.ages) - 1)
        q = np.clip(np.asarray(self.qx, dtype=float)[idx], 0.0, 1.0 - 1e-12)
        return -np.log1p(-q)


def parse_life_table(text: str) -> BackgroundMortality:
    """Rows `age, annual death probability` (0-1); header lines ignored; ages must be increasing."""

    import re

    ages: list[float] = []
    qx: list[float] = []
    for line in str(text or "").splitlines():
        parts = [p for p in re.split(r"[\s,;\t]+", line.strip()) if p]
        if len(parts) < 2:
            continue
        try:
            a, q = float(parts[0]), float(parts[1])
        except ValueError:
            continue
        ages.append(a)
        qx.append(q)
    if len(ages) < 2:
        raise ValueError("Need at least 2 rows of 'age, annual probability of death'")
    if any(b <= a for a, b in zip(ages, ages[1:])):
        raise ValueError("Ages must be strictly increasing")
    if any(q < 0 or q > 1 for q in qx):
        raise ValueError("Probabilities of death must be between 0 and 1 (not percentages)")
    return BackgroundMortality(ages, qx)


def background_survival(bg: BackgroundMortality, start_age: float, t_years) -> np.ndarray:
    t = np.asarray(t_years, dtype=float)
    out = np.ones_like(t)
    if not len(t):
        return out
    order = np.argsort(t)
    ts = t[order]
    grid = np.unique(np.concatenate([[0.0], ts, np.arange(0.0, float(ts[-1]) + 1.0, 0.05)]))
    h = bg.hazard(start_age + grid)
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (h[1:] + h[:-1]) * np.diff(grid))])
    out[order] = np.exp(-np.interp(ts, grid, cum))
    return out


def background_adjusted_survival(
    distribution: str, params: dict[str, float], unit: str, bg: BackgroundMortality, start_age: float,
    horizon_years: float, cutoff_years: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """(t, S) with h(t) = h_fit(t) before `cutoff_years` and max(h_fit(t), h_background(t)) after it."""

    t = _grid(horizon_years)
    h_fit = hazard_curve(distribution, params, unit, t)
    h_bg = bg.hazard(start_age + t)
    h = np.where(t >= cutoff_years, np.maximum(h_fit, h_bg), h_fit)
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (h[1:] + h[:-1]) * np.diff(t))])
    return t, np.exp(-cum)


def background_adjusted_mean_years(*args, **kwargs) -> float:
    t, s = background_adjusted_survival(*args, **kwargs)
    return _simpson(s, t)


# ---------------------------------------------------------------- plausibility

@dataclass
class PlausibilityIssue:
    level: str  # "warning" | "info"
    code: str
    message: str
    distribution: str = ""


def plausibility_checks(
    summary: ExtrapolationSummary, fits: list[FitResult], *, bg: BackgroundMortality | None = None,
    start_age: float | None = None, max_extrapolation_share: float = 0.5, max_survival_at_horizon: float = 0.05,
    pfs_params: tuple[str, dict[str, float], str] | None = None,
) -> list[PlausibilityIssue]:
    """Flags worth reading before a curve is chosen.  `pfs_params` = (distribution, params, unit) of the PFS curve to
    compare with an OS fit (OS must not drop below PFS)."""

    issues: list[PlausibilityIssue] = []
    by_dist = {f.distribution: f for f in fits if f.params}
    horizon = summary.horizon_years
    for row in summary.rows:
        fit = by_dist.get(row.distribution)
        d = row.name or row.distribution
        dist = row.distribution
        unit = row.unit or summary.unit
        if (fit is not None and not fit.converged) or (fit is None and not row.converged):
            issues.append(PlausibilityIssue("warning", "not_converged", f"{d}: the optimiser did not converge; check the fit.", d))
        if row.extrapolation_share > max_extrapolation_share:
            issues.append(PlausibilityIssue(
                "warning", "extrapolation_share",
                f"{d}: {row.extrapolation_share:.0%} of the mean survival over {horizon:g} years lies beyond the "
                f"{summary.followup_years:.2f} years of follow-up; justify the tail (external data, expert opinion).", d))
        if row.survival_at_horizon > max_survival_at_horizon:
            issues.append(PlausibilityIssue(
                "warning", "alive_at_horizon",
                f"{d}: {row.survival_at_horizon:.1%} still alive at the {horizon:g}-year horizon; the mean is truncated by the horizon.", d))
        if dist == "gompertz" and row.params.get("shape", 1.0) < 0:
            issues.append(PlausibilityIssue(
                "warning", "gompertz_plateau",
                "gompertz: negative shape gives a survival plateau (a cured fraction that never dies); check this is intended.", d))
        if bg is not None and start_age is not None:
            t = np.linspace(0.0, horizon, 801)
            s_fit = survival_years(dist, row.params, unit, t)
            s_bg = background_survival(bg, start_age, t)
            over = np.nonzero(s_fit > s_bg + 1e-9)[0]
            over = over[t[over] > 0]
            if len(over):
                issues.append(PlausibilityIssue(
                    "warning", "above_general_population",
                    f"{d}: predicted survival exceeds general-population survival from year {t[over[0]]:.1f} "
                    f"(patients outliving people of the same age). Consider a background-mortality floor.", d))
            h_fit = hazard_curve(dist, row.params, unit, t[1:])
            h_bg = bg.hazard(start_age + t[1:])
            below = np.nonzero((h_fit < h_bg) & (t[1:] > summary.followup_years))[0]
            if len(below):
                issues.append(PlausibilityIssue(
                    "info", "hazard_below_background",
                    f"{d}: after follow-up the hazard drops below background mortality from year {t[1:][below[0]]:.1f}.", d))
        if pfs_params is not None:
            p_dist, p_params, p_unit = pfs_params
            t = np.linspace(0.0, horizon, 801)
            s_os = survival_years(dist, row.params, unit, t)
            s_pfs = survival_years(p_dist, p_params, p_unit, t)
            cross = np.nonzero(s_pfs > s_os + 1e-9)[0]
            if len(cross):
                issues.append(PlausibilityIssue(
                    "warning", "pfs_above_os",
                    f"{d}: the PFS curve is above this OS curve from year {t[cross[0]]:.1f}; the engine floors post-progression at 0.", d))
    return issues


# ---------------------------------------------------------------- candidate library

def _entry_from_fit(name: str, fit: dict[str, Any], unit: str, note: str = "") -> dict[str, Any]:
    return {
        "name": name, "distribution": fit["distribution"], "params": {k: float(v) for k, v in fit["params"].items()},
        "time_unit": unit, "aic": fit.get("aic"), "bic": fit.get("bic"), "note": note,
    }


def library_add(curve: dict[str, Any], name: str, fit: dict[str, Any], unit: str, note: str = "") -> dict[str, Any]:
    """Add (or replace by name) a candidate in `curve["library"]`.  `fit` needs `distribution` and `params`."""

    name = str(name).strip()
    if not name:
        raise ValueError("Give the candidate a name")
    if not fit.get("params"):
        raise ValueError("Cannot store a fit without parameters")
    entry = _entry_from_fit(name, fit, unit, note)
    lib = curve.setdefault("library", [])
    for i, existing in enumerate(lib):
        if existing["name"] == name:
            lib[i] = entry
            return entry
    lib.append(entry)
    return entry


def library_remove(curve: dict[str, Any], name: str) -> bool:
    lib = curve.get("library", [])
    for i, entry in enumerate(lib):
        if entry["name"] == name:
            del lib[i]
            if curve.get("library_selected") == name:
                curve.pop("library_selected", None)
            return True
    return False


def library_select(curve: dict[str, Any], name: str) -> dict[str, Any]:
    """Make candidate `name` the curve in use (distribution, params, time unit) and remember the choice."""

    for entry in curve.get("library", []):
        if entry["name"] == name:
            curve["distribution"] = entry["distribution"]
            curve["params"] = dict(entry["params"])
            curve["time_unit"] = entry["time_unit"]
            curve["library_selected"] = name
            curve.pop("fit_info", None)
            return entry
    raise ValueError(f"No candidate named '{name}' in the library")


def library_from_fit_info(curve: dict[str, Any], prefix: str = "") -> int:
    """Seed the library from `fit_info["candidates"]` (what 'Use this fit' already stores); returns how many were added."""

    info = curve.get("fit_info") or {}
    unit = info.get("time_unit") or curve.get("time_unit", "year")
    added = 0
    for cand in info.get("candidates", []):
        if not cand.get("params"):
            continue
        name = f"{prefix}{cand['distribution']}"
        if not any(e["name"] == name for e in curve.get("library", [])):
            library_add(curve, name, cand, unit)
            added += 1
    return added
