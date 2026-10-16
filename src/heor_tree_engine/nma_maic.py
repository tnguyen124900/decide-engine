"""Matching-adjusted indirect comparison (MAIC; Signorini 2010, Phillippo et al. NICE DSU TSD 18).

Individual patient data (IPD) of one trial are re-weighted so that the weighted covariate means (optionally also the
standard deviations) equal those reported for the comparator (aggregate-data) trial. Weights come from the method
of moments: w_i = exp(x_i' a) with centred covariates x_i - target, a = argmin sum_i exp((x_i - target)' a), which
makes the weighted means equal the target exactly (a unique minimum when the target lies inside the data). The
outcome is then analysed with the weights in the IPD trial:
  binary      weighted logistic regression, log odds ratio
  continuous  weighted linear regression, mean difference
  survival    weighted Cox model (Breslow ties), log hazard ratio
Standard errors: robust sandwich (binary, continuous) or bootstrap that re-estimates the weights (all outcomes).
An anchored comparison combines the adjusted IPD-trial contrast with the aggregate-trial contrast through the
common comparator (Bucher); an unanchored one compares arm outcomes directly and rests on the (untestable)
assumption that all prognostic factors and effect modifiers are matched: state this whenever it is used.
The effective sample size, ESS = (sum w)^2 / sum w^2, measures the information kept by the weighting."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .models import ValidationError

Z95 = 1.959963984540054


@dataclass
class MaicWeights:
    weights: np.ndarray
    alpha: np.ndarray
    ess: float
    n: int
    balance: list[dict] = field(default_factory=list)
    converged: bool = True
    warnings: list[str] = field(default_factory=list)


@dataclass
class MaicEffect:
    outcome: str
    estimate: float  # log OR / MD / log HR of `treatment` vs `comparator` (adjusted)
    se: float
    unadjusted: float
    unadjusted_se: float
    ess: float
    n: int
    se_method: str
    ratio: bool
    weights: MaicWeights | None = None
    warnings: list[str] = field(default_factory=list)
    errors: list[ValidationError] = field(default_factory=list)

    @property
    def natural(self) -> float:
        return math.exp(self.estimate) if self.ratio else self.estimate

    @property
    def interval(self) -> tuple[float, float]:
        lo, hi = self.estimate - Z95 * self.se, self.estimate + Z95 * self.se
        return (math.exp(lo), math.exp(hi)) if self.ratio else (lo, hi)


def estimate_weights(X: np.ndarray, target_means, target_sds=None, names=None) -> MaicWeights:
    """Method-of-moments weights. X (n, p) covariates, target_means (p,), optional target_sds (p,) (None or NaN
    entries: match the mean only). Matching the SD adds the squared covariate with target sd^2 + mean^2."""

    X = np.asarray(X, float)
    n, p = X.shape
    names = list(names) if names is not None else [f"x{i + 1}" for i in range(p)]
    means = np.asarray(target_means, float)
    cols, targets, labels = [X], [means], list(names)
    if target_sds is not None:
        sds = np.asarray(target_sds, float)
        for j in range(p):
            if np.isfinite(sds[j]) and sds[j] > 0:
                cols.append((X[:, j] ** 2)[:, None])
                targets.append(np.array([sds[j] ** 2 + means[j] ** 2]))
                labels.append(f"{names[j]}²")
    Xm = np.hstack(cols)
    tm = np.concatenate(targets)
    Xc = Xm - tm
    out = MaicWeights(weights=np.ones(n), alpha=np.zeros(Xm.shape[1]), ess=float(n), n=n)
    alpha = np.zeros(Xm.shape[1])
    ok = False
    for _ in range(100):
        eta = np.clip(Xc @ alpha, -50, 50)
        w = np.exp(eta)
        grad = Xc.T @ w
        hess = (Xc * w[:, None]).T @ Xc + 1e-12 * np.eye(len(alpha))
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            break
        # damped Newton: halve until the objective decreases
        f0, t = float(w.sum()), 1.0
        while t > 1e-8 and float(np.exp(np.clip(Xc @ (alpha - t * step), -50, 50)).sum()) > f0 + 1e-12:
            t /= 2.0
        alpha = alpha - t * step
        if float(np.max(np.abs(grad) / w.sum())) < 1e-9:
            ok = True
            break
    w = np.exp(np.clip(Xc @ alpha, -50, 50))
    grad = Xc.T @ w
    ok = float(np.max(np.abs(grad) / w.sum() / (1.0 + np.abs(tm)))) < 1e-6
    w = w * n / w.sum()
    out.weights, out.alpha, out.converged = w, alpha, ok
    out.ess = float(w.sum() ** 2 / (w ** 2).sum())
    if not ok:
        out.warnings.append("The weights did not converge: the target population is probably outside the range of the IPD covariates")
    for k, label in enumerate(labels):
        out.balance.append({"covariate": label, "unweighted": float(Xm[:, k].mean()),
                            "weighted": float((w * Xm[:, k]).sum() / w.sum()), "target": float(tm[k])})
    if out.ess < 0.2 * n or out.ess < 20:
        out.warnings.append(f"Effective sample size {out.ess:.0f} of {n}: the weighting keeps little information; results are unstable")
    return out


# --------------------------------------------------------------------------
# Weighted outcome models
# --------------------------------------------------------------------------


def _weighted_glm(y, z, w, family):
    """Intercept + arm indicator; returns (coef_arm, robust_se). family: 'logit' or 'linear'."""

    X = np.column_stack([np.ones(len(y)), z])
    beta = np.zeros(2)
    if family == "logit":
        for _ in range(50):
            p = 1 / (1 + np.exp(-np.clip(X @ beta, -30, 30)))
            W = w * p * (1 - p)
            info = (X * W[:, None]).T @ X + 1e-10 * np.eye(2)
            step = np.linalg.solve(info, X.T @ (w * (y - p)))
            beta = beta + step
            if np.max(np.abs(step)) < 1e-10:
                break
        p = 1 / (1 + np.exp(-np.clip(X @ beta, -30, 30)))
        info = (X * (w * p * (1 - p))[:, None]).T @ X
        score = X * (w * (y - p))[:, None]
    else:
        info = (X * w[:, None]).T @ X
        beta = np.linalg.solve(info, X.T @ (w * y))
        score = X * (w * (y - X @ beta))[:, None]
    inv = np.linalg.inv(info)
    cov = inv @ (score.T @ score) @ inv
    return float(beta[1]), float(math.sqrt(max(cov[1, 1], 0.0)))


def _weighted_cox(time, event, z, w):
    """Weighted Cox log HR for a binary arm indicator (Breslow ties); returns (beta, model-based se)."""

    order = np.argsort(time, kind="stable")
    t, d, zz, ww = time[order], event[order], z[order], w[order]
    first = np.searchsorted(t, t, side="left")
    beta = 0.0
    for _ in range(50):
        e = ww * np.exp(beta * zz)
        s0 = np.cumsum(e[::-1])[::-1]
        s1 = np.cumsum((e * zz)[::-1])[::-1]
        r0, r1 = s0[first], s1[first]
        score = float(np.sum(ww * d * (zz - r1 / r0)))
        info = float(np.sum(ww * d * (r1 / r0 - (r1 / r0) ** 2)))  # z is 0/1 so S2 = S1
        if info <= 0:
            break
        step = score / info
        beta += step
        if abs(step) < 1e-10:
            break
    e = ww * np.exp(beta * zz)
    s0 = np.cumsum(e[::-1])[::-1]
    s1 = np.cumsum((e * zz)[::-1])[::-1]
    info = float(np.sum(ww * d * (s1[first] / s0[first] - (s1[first] / s0[first]) ** 2)))
    return beta, (1.0 / math.sqrt(info) if info > 0 else float("nan"))


def _effect(outcome, data, z, w):
    if outcome == "binary":
        return _weighted_glm(data["event"], z, w, "logit")
    if outcome == "continuous":
        return _weighted_glm(data["value"], z, w, "linear")
    return _weighted_cox(data["time"], data["event"], z, w)


def run_maic(covariates: np.ndarray, arm: np.ndarray, outcome: str, data: dict, target_means, target_sds=None,
             names=None, treatment=1, comparator=0, se_method: str = "auto", n_boot: int = 300, seed: int = 1) -> MaicEffect:
    """Adjusted contrast of `treatment` versus `comparator` in the IPD trial, re-weighted to the target population.
    `arm` holds arm labels; `data` has 'event' (binary, survival), 'value' (continuous) or 'time' + 'event'."""

    ratio = outcome in ("binary", "survival")
    res = MaicEffect(outcome=outcome, estimate=float("nan"), se=float("nan"), unadjusted=float("nan"),
                     unadjusted_se=float("nan"), ess=0.0, n=0, se_method="", ratio=ratio)
    arm = np.asarray(arm)
    keep = np.isin(arm, [treatment, comparator])
    X = np.asarray(covariates, float)[keep]
    z = (arm[keep] == treatment).astype(float)
    d = {k: np.asarray(v, float)[keep] for k, v in data.items()}
    if outcome not in ("binary", "continuous", "survival"):
        res.errors.append(ValidationError(code="outcome", message="Outcome must be binary, continuous or survival"))
        return res
    if len(z) < 10 or z.sum() < 3 or (1 - z).sum() < 3:
        res.errors.append(ValidationError(code="small", message="Each arm needs at least 3 patients"))
        return res
    if np.isnan(X).any() or any(np.isnan(v).any() for v in d.values()):
        res.errors.append(ValidationError(code="missing", message="Missing values: remove or impute patients with missing data"))
        return res
    weights = estimate_weights(X, target_means, target_sds, names)
    res.weights, res.ess, res.n = weights, weights.ess, len(z)
    res.warnings.extend(weights.warnings)
    ones = np.ones(len(z))
    res.unadjusted, res.unadjusted_se = _effect(outcome, d, z, ones)
    est, se = _effect(outcome, d, z, weights.weights)
    method = se_method if se_method != "auto" else ("bootstrap" if outcome == "survival" else "robust")
    if method == "bootstrap":
        rng = np.random.default_rng(seed)
        idx1, idx0 = np.where(z == 1)[0], np.where(z == 0)[0]
        draws = []
        for _ in range(n_boot):
            pick = np.concatenate([rng.choice(idx1, len(idx1)), rng.choice(idx0, len(idx0))])
            wb = estimate_weights(X[pick], target_means, target_sds, names)
            if not wb.converged:
                continue
            draws.append(_effect(outcome, {k: v[pick] for k, v in d.items()}, z[pick], wb.weights)[0])
        if len(draws) < 20:
            res.errors.append(ValidationError(code="bootstrap", message="Too few bootstrap samples with converged weights"))
            return res
        se = float(np.std(draws, ddof=1))
    res.estimate, res.se, res.se_method = est, se, method
    return res


def anchored_indirect(adjusted: float, adjusted_se: float, aggregate: float, aggregate_se: float) -> tuple[float, float]:
    """Bucher: IPD-trial contrast (A vs B, adjusted) minus aggregate-trial contrast (C vs B) gives A vs C."""

    return adjusted - aggregate, math.sqrt(adjusted_se ** 2 + aggregate_se ** 2)


def unanchored_binary(ipd_events: np.ndarray, weights: np.ndarray, agg_events: float, agg_n: float):
    """Log OR of the re-weighted IPD arm versus an aggregate arm (unanchored). Returns (estimate, se)."""

    w = np.asarray(weights, float)
    y = np.asarray(ipd_events, float)
    p1 = float((w * y).sum() / w.sum())
    ess = float(w.sum() ** 2 / (w ** 2).sum())
    p0 = agg_events / agg_n
    est = math.log(p1 / (1 - p1)) - math.log(p0 / (1 - p0))
    se = math.sqrt(1 / (ess * p1 * (1 - p1)) + 1 / (agg_n * p0 * (1 - p0)))
    return est, se
