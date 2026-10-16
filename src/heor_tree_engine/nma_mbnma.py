"""Dose-response model-based network meta-analysis (MBNMA; Mawdsley et al. 2016, CPT Pharmacometrics Syst Pharmacol
5:393-401) on arm-level data, by maximum likelihood (fixed effect).

Arm i.a of study i gives agent, dose and either (mean, se) [normal likelihood, mean difference] or (events, n)
[binomial, logit link]. The linear predictor is  eta = mu_i + f(dose; theta_group(agent)),  f(0) = 0 (a dose-0 arm is
the placebo whatever its agent label). Study intercepts mu_i are nuisance parameters, so only within-study dose
contrasts inform the curves. Agents in the same *group* share one set of dose-response parameters (a common class
effect); by default every agent has its own curve.
Functions (theta parameters, log scale for the positive ones):
  linear      theta0 d
  loglinear   theta0 log(1 + d)
  quadratic   theta0 d + theta1 d^2
  emax        theta0 d / (ED50 + d),                       ED50 = exp(theta1)
  emax_hill   theta0 d^h / (ED50^h + d^h),                 ED50 = exp(theta1), h = exp(theta2)
  exp_sat     theta0 (1 - exp(-d / ED))                    ED = exp(theta1)
Fitting: Fisher scoring with Levenberg-Marquardt damping; covariance = inverse information, inflated by the
residual dispersion (quasi-likelihood) when the normal model shows more variation than the reported standard errors
allow. AIC compares function forms. Limits: no between-study random effects (a study-level random effect on the dose
response is not included); arms of one study are treated as independent given the study intercept."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .models import ValidationError
from .nma_freq import chi2_sf

FUNCTIONS = {"linear": 1, "loglinear": 1, "quadratic": 2, "emax": 2, "emax_hill": 3, "exp_sat": 2}


def dose_response(name: str, dose, theta) -> np.ndarray:
    d = np.asarray(dose, float)
    if name == "linear":
        return theta[0] * d
    if name == "loglinear":
        return theta[0] * np.log1p(d)
    if name == "quadratic":
        return theta[0] * d + theta[1] * d ** 2
    if name == "emax":
        return theta[0] * d / (math.exp(theta[1]) + d)
    if name == "emax_hill":
        h, ed = math.exp(theta[2]), math.exp(theta[1])
        with np.errstate(over="ignore", invalid="ignore"):
            return np.where(d > 0, theta[0] * d ** h / (ed ** h + d ** h), 0.0)
    if name == "exp_sat":
        return theta[0] * (1.0 - np.exp(-d / math.exp(theta[1])))
    raise ValueError(f"Unknown dose-response function '{name}'")


@dataclass
class DoseArm:
    study: str
    agent: str
    dose: float
    mean: float | None = None  # continuous
    se: float | None = None
    events: float | None = None  # binary
    n: float | None = None


@dataclass
class MbnmaResult:
    function: str = "emax"
    outcome: str = "continuous"
    groups: dict = field(default_factory=dict)  # agent -> group
    group_names: list[str] = field(default_factory=list)
    parameter_names: list[str] = field(default_factory=list)
    theta: np.ndarray | None = None
    cov: np.ndarray | None = None
    deviance: float = float("nan")
    aic: float = float("nan")
    n_arms: int = 0
    n_studies: int = 0
    df: int = 0
    dispersion: float = 1.0
    gof_p: float | None = None
    max_dose: dict = field(default_factory=dict)  # agent -> highest dose observed
    warnings: list[str] = field(default_factory=list)
    errors: list[ValidationError] = field(default_factory=list)
    _slices: dict = field(default_factory=dict)

    def se(self, name: str) -> float:
        i = self.parameter_names.index(name)
        return float(math.sqrt(max(self.cov[i, i], 0.0)))

    def curve(self, agent: str, doses, level: float = 0.95):
        """Effect versus dose 0 (mean difference, or log odds ratio) with a delta-method interval:
        (effect, lower, upper) arrays."""

        g = self.groups.get(agent, agent)
        sl = self._slices[g]
        th = self.theta[sl]
        doses = np.asarray(doses, float)
        f = dose_response(self.function, doses, th)
        grad = np.zeros((len(doses), len(th)))
        for k in range(len(th)):
            step = 1e-6 * (1.0 + abs(th[k]))
            up, dn = th.copy(), th.copy()
            up[k] += step
            dn[k] -= step
            grad[:, k] = (dose_response(self.function, doses, up) - dose_response(self.function, doses, dn)) / (2 * step)
        cov = self.cov[sl, sl]
        var = np.einsum("ik,kl,il->i", grad, cov, grad)
        z = 1.959963984540054 if abs(level - 0.95) < 1e-9 else 1.959963984540054
        half = z * np.sqrt(np.maximum(var, 0.0))
        return f, f - half, f + half


def run_mbnma(arms: list[DoseArm], function: str = "emax", groups: dict | None = None, quasi: bool = True) -> MbnmaResult:
    res = MbnmaResult(function=function)
    if function not in FUNCTIONS:
        res.errors.append(ValidationError(code="function", message=f"Unknown function '{function}'"))
        return res
    binary = all(a.events is not None for a in arms) and arms != []
    normal = all(a.mean is not None and a.se not in (None, 0) for a in arms) and arms != []
    if binary == normal:
        res.errors.append(ValidationError(code="data", message="Every arm needs (mean, se) or every arm (events, n)"))
        return res
    res.outcome = "binary" if binary else "continuous"
    arms = [a for a in arms if a.study]
    studies = sorted({a.study for a in arms})
    per_study = {s: [a for a in arms if a.study == s] for s in studies}
    keep = {s for s, v in per_study.items() if len(v) >= 2}
    arms = [a for a in arms if a.study in keep]
    studies = sorted(keep)
    if not studies:
        res.errors.append(ValidationError(code="studies", message="Each study needs at least two arms"))
        return res
    active = [a for a in arms if a.dose > 0]
    if not active:
        res.errors.append(ValidationError(code="dose", message="No arm has a dose above 0"))
        return res
    agents = sorted({a.agent for a in active})
    groups = {**{ag: ag for ag in agents}, **(groups or {})}
    res.groups = groups
    group_names = sorted({groups[ag] for ag in agents})
    res.group_names = group_names
    P = FUNCTIONS[function]
    res._slices = {g: slice(i * P, (i + 1) * P) for i, g in enumerate(group_names)}
    res.parameter_names = [f"{g}: theta{k}" for g in group_names for k in range(P)]
    nθ = len(group_names) * P
    J = len(studies)
    sidx = {s: i for i, s in enumerate(studies)}
    N = len(arms)
    res.n_arms, res.n_studies = N, J
    res.max_dose = {ag: max(a.dose for a in active if a.agent == ag) for ag in agents}
    doses = np.array([a.dose for a in arms])
    arm_group = [groups.get(a.agent) if a.dose > 0 else None for a in arms]
    study_of = np.array([sidx[a.study] for a in arms])
    if binary:
        e = np.array([a.events for a in arms], float)
        nn = np.array([a.n for a in arms], float)
    else:
        y = np.array([a.mean for a in arms], float)
        w = 1.0 / np.array([a.se for a in arms], float) ** 2

    def fvec(theta):
        f = np.zeros(N)
        for i, g in enumerate(arm_group):
            if g is not None:
                f[i] = dose_response(function, doses[i], theta[res._slices[g]])
        return f

    def jac(theta):
        Jm = np.zeros((N, J + nθ))
        Jm[np.arange(N), study_of] = 1.0
        for i, g in enumerate(arm_group):
            if g is None:
                continue
            sl = res._slices[g]
            th = theta[sl]
            for k in range(P):
                step = 1e-6 * (1.0 + abs(th[k]))
                up, dn = th.copy(), th.copy()
                up[k] += step
                dn[k] -= step
                Jm[i, J + sl.start + k] = (dose_response(function, doses[i], up) - dose_response(function, doses[i], dn)) / (2 * step)
        return Jm

    def loglik_parts(psi):
        mu, theta = psi[:J], psi[J:]
        eta = mu[study_of] + fvec(theta)
        if binary:
            p = 1 / (1 + np.exp(-np.clip(eta, -35, 35)))
            p = np.clip(p, 1e-12, 1 - 1e-12)
            ll = float(np.sum(e * np.log(p) + (nn - e) * np.log(1 - p)))
            return ll, e - nn * p, nn * p * (1 - p)
        r = y - eta
        return -0.5 * float(np.sum(w * r ** 2)), w * r, w

    # starting values: study intercepts from the lowest-dose arm; slopes from crude effects at the top dose
    base = np.zeros(J + nθ)
    for s in studies:
        lowest = min((a for a in per_study[s]), key=lambda a: a.dose)
        if binary:
            r_ = min(max(lowest.events / lowest.n, 0.01), 0.99)
            base[sidx[s]] = math.log(r_ / (1 - r_))
        else:
            base[sidx[s]] = lowest.mean

    def crude_effect(group):
        vals = []
        for s in studies:
            arms_s = per_study[s]
            zero = [a for a in arms_s if a.dose == 0]
            top = [a for a in arms_s if a.dose > 0 and groups.get(a.agent) == group]
            if zero and top:
                t_ = max(top, key=lambda a: a.dose)
                if binary:
                    lg = lambda ev, nn_: math.log(min(max(ev / nn_, 0.01), 0.99) / (1 - min(max(ev / nn_, 0.01), 0.99)))  # noqa: E731
                    vals.append(lg(t_.events, t_.n) - lg(zero[0].events, zero[0].n))
                else:
                    vals.append(t_.mean - zero[0].mean)
        return float(np.median(vals)) if vals else 0.1

    def optimise(psi):
        ll, u, W = loglik_parts(psi)
        lam = 1e-3
        for _ in range(400):
            Jm = jac(psi[J:])
            info = Jm.T @ (Jm * W[:, None])
            grad = Jm.T @ u
            improved = False
            for _try in range(30):
                try:
                    step = np.linalg.solve(info + lam * np.diag(np.maximum(np.diag(info), 1e-9)), grad)
                except np.linalg.LinAlgError:
                    lam *= 10
                    continue
                new = psi + step
                new_ll, new_u, new_W = loglik_parts(new)
                if np.isfinite(new_ll) and new_ll >= ll - 1e-10:
                    psi, ll, u, W = new, new_ll, new_u, new_W
                    lam = max(lam / 5, 1e-9)
                    improved = True
                    break
                lam *= 5
            if not improved or float(np.max(np.abs(step))) < 1e-9:
                break
        return psi, ll, u, W

    starts = [1.0 / 10, 1.0 / 3, 1.0] if function in ("emax", "emax_hill", "exp_sat") else [1.0]
    best = None
    for frac in starts:
        psi = base.copy()
        for g in group_names:
            top = max(res.max_dose[ag] for ag in agents if groups[ag] == g)
            sl = res._slices[g]
            eff = crude_effect(g)
            th = np.zeros(P)
            if function in ("linear", "loglinear"):
                th[0] = eff / (top if function == "linear" else math.log1p(top))
            elif function == "quadratic":
                th[0], th[1] = eff / top, 0.0
            else:
                th[0] = 1.5 * eff if eff != 0 else 0.1
                th[1] = math.log(top * frac)
                if function == "emax_hill":
                    th[2] = 0.0
            psi[J + sl.start:J + sl.stop] = th
        cand = optimise(psi)
        if best is None or cand[1] > best[1]:
            best = cand
    psi, ll, u, W = best
    Jm = jac(psi[J:])
    info = Jm.T @ (Jm * W[:, None])
    dscale = 1.0 / np.sqrt(np.maximum(np.diag(info), 1e-300))
    if np.linalg.matrix_rank(info * dscale[:, None] * dscale[None, :], tol=1e-9) < info.shape[0]:
        res.errors.append(ValidationError(code="singular", message="The dose-response parameters are not identified by these data: use a simpler function or share parameters within classes"))
        return res
    cov = np.linalg.inv(info)
    res.theta = psi[J:]
    res.df = N - J - nθ
    if binary:
        pfit = 1 / (1 + np.exp(-np.clip(psi[:J][study_of] + fvec(psi[J:]), -35, 35)))
        with np.errstate(divide="ignore", invalid="ignore"):
            dev = 2 * np.sum(np.where(e > 0, e * np.log(e / (nn * pfit)), 0.0) + np.where(nn - e > 0, (nn - e) * np.log((nn - e) / (nn * (1 - pfit))), 0.0))
        res.deviance = float(dev)
        res.aic = -2 * ll + 2 * (J + nθ)
    else:
        res.deviance = float(np.sum(w * (y - psi[:J][study_of] - fvec(psi[J:])) ** 2))
        res.aic = res.deviance + 2 * (J + nθ)
    if res.df > 0:
        res.gof_p = chi2_sf(res.deviance, res.df)
        res.dispersion = res.deviance / res.df
        if quasi and res.dispersion > 1.0:
            cov = cov * res.dispersion
            res.warnings.append(f"Residual dispersion {res.dispersion:.2f}: standard errors were inflated (quasi-likelihood) for extra variation between studies")
    res.cov = cov[J:, J:]
    return res
