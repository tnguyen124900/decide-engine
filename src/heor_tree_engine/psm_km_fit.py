"""Fit PSM survival curves straight from Kaplan-Meier / patient-level data (no NMA needed).

Input (either one):
  * patient-level rows `time, event` (event 1 = event, 0 = censored), or
  * a digitised KM curve `time, survival` + an at-risk table `time, n_at_risk`, turned into pseudo patient-level
    data with `nma_km_reconstruct.reconstruct_arm` (Guyot et al. 2012).

Each of the six PSM distributions (exponential, weibull, log_normal, log_logistic, gompertz, generalized_gamma) is
fitted by maximum likelihood with right censoring: log L = sum_events log h(t) + sum_all log S(t).  Parameters are
returned in the *same time unit as the data* and use the exact names/parameterisation of `PSM_CURVE_DISTRIBUTIONS`
(weibull S = exp(-(t/scale)^shape), log-logistic S = 1/(1+(t/scale)^shape), Gompertz S = exp(-(rate/shape)(e^{shape t}-1)),
generalized gamma = Prentice mu/sigma/Q as in `des_sampling`), so a fit can be copied straight into a PSM curve.
Optimiser: Nelder-Mead on unconstrained parameters with several starts (scipy's if present, otherwise a built-in
implementation); the generalized gamma needs scipy for its likelihood.  No standard errors are reported (AIC/BIC only).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import numpy as np

from .nma_km_reconstruct import kaplan_meier, reconstruct_arm

FIT_DISTRIBUTIONS = ("exponential", "weibull", "log_normal", "log_logistic", "gompertz", "generalized_gamma")
_TINY = 1e-300

_erfc = np.frompyfunc(math.erfc, 1, 1)


@dataclass
class FitResult:
    distribution: str
    params: dict[str, float] = field(default_factory=dict)
    loglik: float = float("nan")
    k: int = 0
    aic: float = float("nan")
    bic: float = float("nan")
    converged: bool = False
    message: str = ""
    max_gap: float = float("nan")  # max |S_fit - S_KM| at the observed event times
    rmse: float = float("nan")  # RMS of the same gap


@dataclass
class KMData:
    times: np.ndarray
    events: np.ndarray
    n: int = 0
    n_events: int = 0
    source: str = ""  # "patient-level" | "reconstructed"
    warnings: list[str] = field(default_factory=list)


# ------------------------------------------------------------------ parsing

def _rows(text: str) -> list[list[float]]:
    out: list[list[float]] = []
    for line in str(text or "").splitlines():
        parts = [p for p in re.split(r"[\s,;\t]+", line.strip()) if p]
        if not parts:
            continue
        try:
            out.append([float(p) for p in parts])
        except ValueError:
            continue  # header or junk line
    return out


def parse_patient_data(text: str) -> KMData:
    """Rows `time, event` (event 1/0).  A header line is ignored."""

    rows = [r for r in _rows(text) if len(r) >= 2]
    if len(rows) < 5:
        raise ValueError("Need at least 5 rows of 'time, event' (event 1 = event, 0 = censored)")
    times = np.array([r[0] for r in rows], dtype=float)
    events = np.array([r[1] for r in rows], dtype=float)
    if np.any(times < 0) or not np.all(np.isfinite(times)):
        raise ValueError("Times must be finite and >= 0")
    if not np.all(np.isin(events, (0.0, 1.0))):
        raise ValueError("The event column must contain only 0 (censored) or 1 (event)")
    if events.sum() < 3:
        raise ValueError("Need at least 3 events to fit a survival curve")
    return KMData(times, events, len(times), int(events.sum()), "patient-level")


def reconstruct_from_km(curve_text: str, risk_text: str, total_events: int | None = None) -> KMData:
    """Digitised KM points `time, survival` + at-risk table `time, n` (first row at time 0)."""

    curve = [(r[0], r[1]) for r in _rows(curve_text) if len(r) >= 2]
    risk = [(r[0], r[1]) for r in _rows(risk_text) if len(r) >= 2]
    if len(curve) < 3:
        raise ValueError("Need at least 3 digitised KM points 'time, survival'")
    if any(s > 1.0 + 1e-9 for _, s in curve):
        raise ValueError("Survival must be a proportion between 0 and 1 (not a percentage)")
    if not risk or risk[0][0] != 0:
        raise ValueError("The at-risk table must start at time 0 with the number of patients")
    arm = reconstruct_arm(curve, [t for t, _ in risk], [n for _, n in risk], total_events)
    if not arm.times:
        raise ValueError("; ".join(arm.warnings) or "Could not reconstruct patient-level data")
    events = np.array(arm.events, dtype=float)
    if events.sum() < 3:
        raise ValueError("The reconstruction produced fewer than 3 events: check the curve and at-risk table")
    return KMData(np.array(arm.times, dtype=float), events, arm.n, int(events.sum()), "reconstructed", list(arm.warnings))


# ------------------------------------------------------------ distributions

def _scipy_stats():
    try:
        from scipy import stats
    except ImportError:  # pragma: no cover - environment dependent
        return None
    return stats


def survival(distribution: str, params: dict[str, float], t) -> np.ndarray:
    """Vectorised S(t) for t >= 0 (same formulas as the engine's `_curve_survival`)."""

    t = np.asarray(t, dtype=float)
    out = np.ones_like(t)
    pos = t > 0
    x = t[pos]
    if distribution == "exponential":
        out[pos] = np.exp(-params["rate"] * x)
    elif distribution == "weibull":
        out[pos] = np.exp(-((x / params["scale"]) ** params["shape"]))
    elif distribution == "log_normal":
        z = (np.log(x) - params["meanlog"]) / (params["sdlog"] * math.sqrt(2.0))
        out[pos] = 0.5 * _erfc(z).astype(float)
    elif distribution == "log_logistic":
        out[pos] = 1.0 / (1.0 + (x / params["scale"]) ** params["shape"])
    elif distribution == "gompertz":
        shape, rate = params["shape"], params["rate"]
        if abs(shape) < 1e-12:
            out[pos] = np.exp(-rate * x)
        else:
            out[pos] = np.exp(-(rate / shape) * np.expm1(np.clip(shape * x, -700, 700)))
    elif distribution == "generalized_gamma":
        stats = _scipy_stats()
        if stats is None:
            raise ValueError("Generalized Gamma requires scipy")
        out[pos] = _gengamma_frozen(params).sf(x)
    else:
        raise ValueError(f"Unsupported distribution '{distribution}'")
    return out


def _gengamma_frozen(params: dict[str, float]):
    from .des_sampling import _gengamma

    q = params["Q"]
    if abs(q) < 1e-4:
        q = 1e-4 if q >= 0 else -1e-4
    return _gengamma({"mu": params["mu"], "sigma": params["sigma"], "Q": q})


def _log_surv_and_log_haz(distribution: str, p: dict[str, float], t: np.ndarray):
    """(log S(t), log h(t)) for t > 0."""

    lt = np.log(t)
    if distribution == "exponential":
        return -p["rate"] * t, np.full_like(t, math.log(p["rate"]))
    if distribution == "weibull":
        k, s = p["shape"], p["scale"]
        z = lt - math.log(s)
        return -np.exp(k * z), math.log(k) - math.log(s) + (k - 1.0) * z
    if distribution == "log_normal":
        mu, sd = p["meanlog"], p["sdlog"]
        z = (lt - mu) / sd
        s = 0.5 * _erfc(z / math.sqrt(2.0)).astype(float)
        log_s = np.log(np.maximum(s, _TINY))
        log_pdf = -0.5 * z * z - 0.5 * math.log(2 * math.pi) - math.log(sd) - lt
        return log_s, log_pdf - log_s
    if distribution == "log_logistic":
        k, s = p["shape"], p["scale"]
        z = lt - math.log(s)
        log1p_u = np.logaddexp(0.0, k * z)
        return -log1p_u, math.log(k) - math.log(s) + (k - 1.0) * z - log1p_u
    if distribution == "gompertz":
        b, r = p["shape"], p["rate"]
        if abs(b) < 1e-12:
            return -r * t, np.full_like(t, math.log(r))
        return -(r / b) * np.expm1(np.clip(b * t, -700, 700)), math.log(r) + b * t
    if distribution == "generalized_gamma":
        # Stacy form of the Prentice (mu, sigma, Q) parameterisation, evaluated with scipy.special (fast, vectorised)
        from scipy.special import gammainc, gammaincc, gammaln

        a, c, log_scale = _gg_stacy(p)
        y = np.exp(c * (lt - log_scale))
        s = gammaincc(a, y) if c > 0 else gammainc(a, y)
        log_s = np.log(np.maximum(s, _TINY))
        log_pdf = math.log(abs(c)) + (a * c - 1.0) * lt - y - a * c * log_scale - gammaln(a)
        return log_s, log_pdf - log_s
    raise ValueError(distribution)


def _gg_stacy(p: dict[str, float]) -> tuple[float, float, float]:
    """(a, c, log scale) with X = scale * Gamma(a)^(1/c); same conversion as des_sampling._gengamma."""

    q = p["Q"]
    if abs(q) < 1e-4:
        q = 1e-4 if q >= 0 else -1e-4
    return 1.0 / (q * q), q / p["sigma"], p["mu"] + (2.0 * p["sigma"] / q) * math.log(abs(q))


def _neg_loglik(distribution: str, p: dict[str, float], times: np.ndarray, events: np.ndarray) -> float:
    pos = times > 0
    t, e = times[pos], events[pos]
    with np.errstate(all="ignore"):
        log_s, log_h = _log_surv_and_log_haz(distribution, p, t)
    ll = float(np.sum(log_s) + np.sum(np.where(e > 0, log_h, 0.0)))
    return -ll if math.isfinite(ll) else 1e30


# ----------------------------------------------------------------- fitting

# unconstrained vector <-> parameter dict
def _to_params(distribution: str, v) -> dict[str, float] | None:
    try:
        if distribution == "exponential":
            return {"rate": math.exp(v[0])}
        if distribution in ("weibull", "log_logistic"):
            return {"shape": math.exp(v[0]), "scale": math.exp(v[1])}
        if distribution == "log_normal":
            return {"meanlog": v[0], "sdlog": math.exp(v[1])}
        if distribution == "gompertz":
            return {"shape": v[0], "rate": math.exp(v[1])}
        if distribution == "generalized_gamma":
            return {"mu": v[0], "sigma": math.exp(v[1]), "Q": v[2]}
    except OverflowError:
        return None
    raise ValueError(distribution)


def _nelder_mead(fn, x0, max_iter=1500, tol=1e-9):
    n = len(x0)
    pts = [np.array(x0, dtype=float)]
    for i in range(n):
        p = np.array(x0, dtype=float)
        p[i] += 0.3 if abs(p[i]) < 1e-8 else 0.2 * abs(p[i]) + 0.1
        pts.append(p)
    vals = [fn(p) for p in pts]
    for _ in range(max_iter):
        order = np.argsort(vals)
        pts = [pts[i] for i in order]
        vals = [vals[i] for i in order]
        if abs(vals[-1] - vals[0]) < tol and max(np.max(np.abs(p - pts[0])) for p in pts[1:]) < 1e-7:
            return pts[0], vals[0], True
        centroid = np.mean(pts[:-1], axis=0)
        xr = centroid + (centroid - pts[-1])
        fr = fn(xr)
        if fr < vals[0]:
            xe = centroid + 2.0 * (centroid - pts[-1])
            fe = fn(xe)
            pts[-1], vals[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < vals[-2]:
            pts[-1], vals[-1] = xr, fr
        else:
            xc = centroid + 0.5 * ((xr if fr < vals[-1] else pts[-1]) - centroid)
            fc = fn(xc)
            if fc < min(fr, vals[-1]):
                pts[-1], vals[-1] = xc, fc
            else:
                for i in range(1, len(pts)):
                    pts[i] = pts[0] + 0.5 * (pts[i] - pts[0])
                    vals[i] = fn(pts[i])
    best = int(np.argmin(vals))
    return pts[best], vals[best], False


def _minimise(fn, x0):
    try:
        from scipy.optimize import minimize
    except ImportError:
        return _nelder_mead(fn, x0)
    res = minimize(fn, x0, method="Nelder-Mead", options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 4000, "maxfev": 8000})
    return res.x, float(res.fun), bool(res.success)


def _starts(distribution: str, data: KMData) -> list[list[float]]:
    t, e = data.times, data.events
    pos = t[t > 0]
    total = float(np.sum(t))
    lam = max(float(np.sum(e)) / max(total, 1e-12), 1e-9)
    med = float(np.median(pos)) if len(pos) else 1.0
    ev_t = t[(e > 0) & (t > 0)]
    lt = np.log(ev_t) if len(ev_t) else np.log(pos)
    ml, sl = float(np.mean(lt)), max(float(np.std(lt)), 0.1)
    tscale = float(np.max(t)) or 1.0
    if distribution == "exponential":
        return [[math.log(lam)]]
    if distribution == "weibull":
        return [[math.log(k), math.log(1.0 / lam)] for k in (0.8, 1.0, 1.5, 2.5)]
    if distribution == "log_normal":
        return [[ml, math.log(sl)], [ml + 0.5, math.log(sl * 1.5)]]
    if distribution == "log_logistic":
        return [[math.log(k), math.log(max(med, 1e-9))] for k in (1.0, 1.5, 2.5)]
    if distribution == "gompertz":
        return [[b / tscale, math.log(lam)] for b in (0.0001, 1.0, -0.5, 3.0)]
    if distribution == "generalized_gamma":
        return [[ml, math.log(sl), q] for q in (0.5, -0.5, 1.5)]
    raise ValueError(distribution)


_N_PARAMS = {"exponential": 1, "weibull": 2, "log_normal": 2, "log_logistic": 2, "gompertz": 2, "generalized_gamma": 3}


def fit_distribution(distribution: str, data: KMData) -> FitResult:
    res = FitResult(distribution, k=_N_PARAMS[distribution])
    if distribution == "generalized_gamma" and _scipy_stats() is None:
        res.message = "Needs scipy (not installed here)"
        return res

    def fn(v):
        p = _to_params(distribution, v)
        return 1e30 if p is None else _neg_loglik(distribution, p, data.times, data.events)

    best = None
    for x0 in _starts(distribution, data):
        try:
            x, f, ok = _minimise(fn, x0)
            x, f2, ok2 = _minimise(fn, x)  # restart from the optimum to escape a premature simplex collapse
        except Exception as exc:  # noqa: BLE001 - a bad start must not kill the other starts
            res.message = str(exc)
            continue
        if best is None or f2 < best[1]:
            best = (x, f2, ok or ok2)
    if best is None or best[1] >= 1e29:
        res.message = res.message or "Optimiser failed"
        return res
    x, f, ok = best
    params = _to_params(distribution, x)
    res.params = {k: float(v) for k, v in params.items()}
    res.loglik = -f
    n_obs = int(np.sum(data.times > 0))
    res.aic = 2 * res.k - 2 * res.loglik
    res.bic = res.k * math.log(max(n_obs, 1)) - 2 * res.loglik
    res.converged = bool(ok)
    res.message = "" if ok else "Optimiser stopped at its iteration limit; check the fit"
    gap = km_gap(res, data)
    res.max_gap, res.rmse = gap
    return res


def fit_all(data: KMData, distributions=FIT_DISTRIBUTIONS) -> list[FitResult]:
    """All requested distributions, best (lowest AIC) first; failed fits last."""

    fits = [fit_distribution(d, data) for d in distributions]
    return sorted(fits, key=lambda r: (not math.isfinite(r.aic), r.aic if math.isfinite(r.aic) else 0.0))


def km_steps(data: KMData) -> list[tuple[float, float]]:
    return kaplan_meier(list(data.times), [int(x) for x in data.events])


def km_gap(fit: FitResult, data: KMData) -> tuple[float, float]:
    """(max, RMS) of |S_fit(t) - S_KM(t)| at the KM step times (a fit-vs-KM check, not a formal test)."""

    steps = km_steps(data)
    t = np.array([s[0] for s in steps if s[0] > 0], dtype=float)
    km = np.array([s[1] for s in steps if s[0] > 0], dtype=float)
    if not len(t) or not fit.params:
        return float("nan"), float("nan")
    d = np.abs(survival(fit.distribution, fit.params, t) - km)
    return float(np.max(d)), float(math.sqrt(float(np.mean(d * d))))


def fit_summary_text(fit: FitResult, data: KMData, unit: str, label: str = "") -> str:
    """The Source string stored next to a fitted curve."""

    where = "patient-level data" if data.source == "patient-level" else "reconstructed KM data (Guyot)"
    head = f"{label}: " if label else ""
    return (f"{head}{fit.distribution} fitted by maximum likelihood to {where} "
            f"(n={data.n}, events={data.n_events}, time unit {unit}); AIC={fit.aic:.2f}, BIC={fit.bic:.2f}")
