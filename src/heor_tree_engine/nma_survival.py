"""Parametric / fractional-polynomial survival NMA on interval event counts (C1c; Jansen 2011 BMC Med Res
Methodol 11:61; Ouwens 2010; NICE DSU TSD 14 and 19).

Data: for every study arm, per time interval, the number of events and the person-time (from published counts
or reconstructed from Kaplan-Meier curves with nma_km_reconstruct). Likelihood per interval:
    events ~ Poisson(person_time * r),   r = (H(t1) - H(t0)) / (t1 - t0)  (mean hazard over the interval),
exact for a constant hazard and a close approximation for narrow intervals.

Hazard model (time is divided by `scale` = the longest follow-up):
    log h(t) = theta_0 + theta_1 phi_1(t) + ... + theta_M phi_M(t)
with fractional-polynomial terms phi (Royston-Altman): t^p, or log t when p = 0; a repeated power gives
phi * log t. Special cases: exponential (no terms), Weibull (p = 0), Gompertz (p = 1). Powers are chosen from
{-2, -1, -0.5, 0, 0.5, 1, 2, 3}. Every coefficient carries a treatment effect d_k (relative to the reference).
Study arm coefficients: theta_ia = mu_i + delta_ia, delta_ia = d_ta - d_tb in the fixed-effect model, or drawn
from N(d_ta - d_tb, tau_m^2 (I + J)/2) per coefficient in the random-effects model.

Sampling: Metropolis-adjusted Langevin steps preconditioned by the local information matrix (analytic gradient
and Hessian of the Poisson likelihood) for each study block and, in the fixed-effect model, for d; Gibbs for d
and a random-walk step for log tau_m in the random-effects model. Reported: split R-hat, ESS, DIC (Poisson
deviance).
Limits: a very small number of studies leaves the shape coefficients weakly identified: check R-hat, look at the
fitted hazards and treat extrapolation beyond the follow-up with care."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .models import ValidationError
from .nma_bayes import ParameterDiagnostics, _summary

FP_POWERS = (-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0)
MODEL_LABELS = {"exponential": "Exponential", "weibull": "Weibull", "gompertz": "Gompertz"}
_GL_X, _GL_W = np.polynomial.legendre.leggauss(6)
_GL_W = _GL_W / 2.0  # weights of the mean over an interval


def _interval_nodes(a: np.ndarray, b: np.ndarray):
    """Quadrature nodes t (J, Q) and weights w (J, Q) of the *mean* of a function over each interval [a, b].
    An interval starting at 0 uses t = b x^2 (x uniform on [0, 1]) so that log t and small powers, which are
    singular at 0, are integrated accurately."""

    mid, half = (a + b) / 2.0, (b - a) / 2.0
    t = mid[:, None] + half[:, None] * _GL_X[None, :]
    w = np.tile(_GL_W, (len(a), 1))
    zero = a <= 0
    if zero.any():
        x = (_GL_X + 1.0) / 2.0
        t[zero] = b[zero, None] * x[None, :] ** 2
        w[zero] = (2.0 * x * _GL_W)[None, :]  # density 2x on [0, 1] (the weights of the mean sum to 1)
    return t, w


def model_powers(name: str, powers=None) -> list[float]:
    if name == "exponential":
        return []
    if name == "weibull":
        return [0.0]
    if name == "gompertz":
        return [1.0]
    if name in ("fp1", "fp2"):
        n = 1 if name == "fp1" else 2
        p = [float(x) for x in (powers or [])]
        if len(p) != n or any(x not in FP_POWERS for x in p):
            raise ValueError(f"{name} needs {n} power(s) from {list(FP_POWERS)}")
        return sorted(p)
    raise ValueError(f"Unknown survival model '{name}'")


def basis(t: np.ndarray, powers: list[float]) -> np.ndarray:
    """(..., P) matrix [1, phi_1(t), ...] for scaled positive times t."""

    t = np.asarray(t, dtype=float)
    cols = [np.ones_like(t)]
    for i, p in enumerate(powers):
        if i > 0 and p == powers[i - 1]:
            cols.append(cols[-1] * np.log(t))
        else:
            cols.append(np.log(t) if p == 0 else t ** p)
    return np.stack(cols, axis=-1)


@dataclass
class SurvivalArmData:
    study: str
    treatment: str
    t_from: list[float]
    t_to: list[float]
    events: list[float]
    person_time: list[float]


@dataclass
class SurvivalSettings:
    model: str = "weibull"  # exponential | weibull | gompertz | fp1 | fp2
    powers: list[float] = field(default_factory=list)
    effects: str = "random"
    chains: int = 3
    warmup: int = 800
    draws_per_chain: int = 800
    seed: int = 1
    prior_sd: float = 100.0  # N(0, prior_sd^2) on d and mu
    tau_scale: float = 2.0  # half-normal scale of tau_m
    ci_level: float = 0.95


@dataclass
class SurvivalResult:
    model: str = "weibull"
    powers: list[float] = field(default_factory=list)
    effects: str = "random"
    reference: str | None = None
    treatments: list[str] = field(default_factory=list)
    scale: float = 1.0
    coefficient_names: list[str] = field(default_factory=list)
    d_draws: np.ndarray | None = None  # (S, K, P) with the reference row zero
    tau_draws: np.ndarray | None = None  # (S, P) random effects
    theta_draws: dict = field(default_factory=dict)  # (study, treatment) -> (S, P)
    study_reference_theta: dict = field(default_factory=dict)  # study -> (S, P) log-hazard coefficients of the reference
    diagnostics: list[ParameterDiagnostics] = field(default_factory=list)
    dbar: float = float("nan")
    p_d: float = float("nan")
    dic: float = float("nan")
    n_data: int = 0
    max_time: float = 0.0
    warnings: list[str] = field(default_factory=list)
    errors: list[ValidationError] = field(default_factory=list)


class _Arm:
    def __init__(self, data: SurvivalArmData, scale: float, powers: list[float], index: int):
        self.study, self.treatment, self.index = data.study, data.treatment, index
        a = np.asarray(data.t_from, float) / scale
        b = np.asarray(data.t_to, float) / scale
        self.e = np.asarray(data.events, float)
        self.py = np.asarray(data.person_time, float) / scale
        t, self.w = _interval_nodes(a, b)  # (J, Q)
        self.Phi = basis(t, powers)  # (J, Q, P)
        self.dev_const = 0.0

    def stats(self, theta: np.ndarray, need_hessian: bool = True):
        eta = np.clip(self.Phi @ theta, -40.0, 25.0)  # (J, Q)
        E = self.w * np.exp(eta)
        r = np.maximum(E.sum(axis=1), 1e-300)
        mu = self.py * r
        ll = float(np.sum(self.e * np.log(np.maximum(mu, 1e-300)) - mu))
        R1 = np.einsum("jq,jqp->jp", E, self.Phi)
        a = self.e / r - self.py
        grad = a @ R1
        if not need_hessian:
            return ll, grad, None
        R2 = np.einsum("jq,jqp,jqr->jpr", E, self.Phi, self.Phi)
        hess = np.einsum("j,jpr->pr", a, R2) - np.einsum("j,jp,jr->pr", self.e / r ** 2, R1, R1)
        return ll, grad, hess

    def deviance(self, theta: np.ndarray) -> float:
        eta = np.clip(self.Phi @ theta, -40.0, 25.0)
        mu = np.maximum(self.py * (self.w * np.exp(eta)).sum(axis=1), 1e-300)
        pos = self.e > 0
        term = np.where(pos, self.e * np.log(np.where(pos, self.e, 1.0) / mu), 0.0) - (self.e - mu)
        return float(2.0 * term.sum())


def _psd_cov(A: np.ndarray):
    w, V = np.linalg.eigh((A + A.T) / 2.0)
    w = np.maximum(w, 1e-8 + 1e-9 * max(w.max(), 1.0))
    return w, V


def _mala(psi: np.ndarray, fn, h: float, rng: np.random.Generator, current=None):
    """One preconditioned MALA step. fn(psi) -> (logpost, grad, hessian_of_logpost). Returns (psi, accepted, state)."""

    lp, g, H = current if current is not None else fn(psi)
    w, V = _psd_cov(-H)
    mean = psi + 0.5 * h * h * (V @ ((V.T @ g) / w))
    prop = mean + h * (V @ (rng.standard_normal(len(psi)) / np.sqrt(w)))
    lp2, g2, H2 = fn(prop)
    if not np.isfinite(lp2):
        return psi, False, (lp, g, H)
    w2, V2 = _psd_cov(-H2)
    mean2 = prop + 0.5 * h * h * (V2 @ ((V2.T @ g2) / w2))
    fwd = -0.5 * np.sum((V.T @ (prop - mean)) ** 2 * w) / h ** 2 + 0.5 * np.sum(np.log(w)) - len(psi) * math.log(h)
    rev = -0.5 * np.sum((V2.T @ (psi - mean2)) ** 2 * w2) / h ** 2 + 0.5 * np.sum(np.log(w2)) - len(psi) * math.log(h)
    if math.log(rng.random()) < lp2 - lp + rev - fwd:
        return prop, True, (lp2, g2, H2)
    return psi, False, (lp, g, H)


def run_survival_nma(arms: list[SurvivalArmData], reference: str, settings: SurvivalSettings | None = None) -> SurvivalResult:
    st = settings or SurvivalSettings()
    out = SurvivalResult(model=st.model, effects=st.effects, reference=reference)
    try:
        powers = model_powers(st.model, st.powers)
    except ValueError as exc:
        out.errors.append(ValidationError(code="model", message=str(exc)))
        return out
    out.powers = powers
    P = 1 + len(powers)
    out.coefficient_names = ["log-hazard (intercept)"] + [
        ("log t" if p == 0 else f"t^{p:g}") + (" x log t" if i > 0 and p == powers[i - 1] else "") for i, p in enumerate(powers)]
    # --- validation and structure
    clean = []
    for arm in arms:
        rows = [(a, b, e, py) for a, b, e, py in zip(arm.t_from, arm.t_to, arm.events, arm.person_time)
                if b > a and py > 0 and e >= 0]
        if rows:
            clean.append(SurvivalArmData(arm.study, arm.treatment, *[list(x) for x in zip(*rows)]))
    studies: dict[str, list[SurvivalArmData]] = {}
    for arm in clean:
        studies.setdefault(arm.study, []).append(arm)
    studies = {k: v for k, v in studies.items() if len(v) >= 2 and len({a.treatment for a in v}) == len(v)}
    if not studies:
        out.errors.append(ValidationError(code="no_studies", message="No study with two or more arms and valid interval data"))
        return out
    names: list[str] = [reference]
    for arms_ in studies.values():
        for arm in arms_:
            if arm.treatment not in names:
                names.append(arm.treatment)
    if not any(a.treatment == reference for v in studies.values() for a in v):
        out.errors.append(ValidationError(code="reference", message=f"The reference '{reference}' is in no study"))
        return out
    # connectivity
    seen, changed = {reference}, True
    while changed:
        changed = False
        for arms_ in studies.values():
            ts = {a.treatment for a in arms_}
            if ts & seen and not ts <= seen:
                seen |= ts
                changed = True
    if seen != set(names):
        out.errors.append(ValidationError(code="disconnected", message="The network is not connected"))
        return out
    K = len(names)
    out.treatments = names
    tindex = {n: i for i, n in enumerate(names)}
    scale = max(b for v in studies.values() for a in v for b in a.t_to)
    out.scale, out.max_time = scale, scale
    sid = list(studies)
    arm_objs: dict[str, list[_Arm]] = {}
    n_data = 0
    for s, v in studies.items():
        arm_objs[s] = [_Arm(a, scale, powers, i) for i, a in enumerate(v)]
        n_data += sum(len(a.e) for a in arm_objs[s])
    out.n_data = n_data
    if st.effects == "random" and len(studies) < 3:
        out.warnings.append("Random effects with fewer than 3 studies: heterogeneity is poorly identified; consider fixed effects")
    rng_master = np.random.SeedSequence(st.seed)
    free = [i for i in range(K) if i != 0]  # non-reference treatment indices
    D = len(free) * P
    col = {k: j for j, k in enumerate(free)}  # treatment -> position among free

    # per-arm initial coefficients
    def initial_mu(s: str) -> np.ndarray:
        a = arm_objs[s][0]
        mu = np.zeros(P)
        mu[0] = math.log(max(a.e.sum(), 0.5) / max(a.py.sum(), 1e-9))
        return mu

    S_keep = st.chains * st.draws_per_chain
    d_store = np.zeros((st.chains, st.draws_per_chain, K, P))
    tau_store = np.zeros((st.chains, st.draws_per_chain, P))
    theta_store = {(s, a.treatment): np.zeros((st.chains, st.draws_per_chain, P)) for s, v in arm_objs.items() for a in v}
    dev_store = np.zeros((st.chains, st.draws_per_chain))
    s0 = st.prior_sd
    random_effects = st.effects == "random"
    accept = {"n": 0, "a": 0}

    for chain in range(st.chains):
        rng = np.random.default_rng(rng_master.spawn(1)[0])
        d = np.zeros((K, P))
        mu = {s: initial_mu(s) + 0.01 * rng.standard_normal(P) for s in sid}
        delta = {s: np.zeros((len(arm_objs[s]) - 1, P)) for s in sid}
        tau = np.full(P, 0.3)
        h_study, h_d = 0.9, 0.9
        for it in range(st.warmup + st.draws_per_chain):
            adapt = it < st.warmup
            # ---- study blocks
            for s in sid:
                objs = arm_objs[s]
                nA = len(objs)
                tb = tindex[studies[s][0].treatment]
                if not random_effects:
                    offsets = [d[tindex[studies[s][i].treatment]] - d[tb] for i in range(nA)]

                    def fn(psi, objs=objs, offsets=offsets):
                        lp = -0.5 * float(psi @ psi) / s0 ** 2
                        g = -psi / s0 ** 2
                        H = -np.eye(P) / s0 ** 2
                        for o, off in zip(objs, offsets):
                            ll, gg, hh = o.stats(psi + off)
                            lp, g, H = lp + ll, g + gg, H + hh
                        return lp, g, H
                    mu[s], acc, _ = _mala(mu[s], fn, h_study, rng)
                else:
                    c = nA - 1
                    dvec = np.array([d[tindex[studies[s][i].treatment]] - d[tb] for i in range(1, nA)])  # (c, P)
                    Sinv = 2.0 * (np.eye(c) - np.ones((c, c)) / (c + 1))
                    Lam = np.zeros((c * P, c * P))
                    for m in range(P):
                        idx = np.arange(c) * P + m
                        Lam[np.ix_(idx, idx)] = Sinv / tau[m] ** 2
                    dim = P * nA
                    psi0 = np.concatenate([mu[s], delta[s].reshape(-1)])
                    mean_delta = dvec.reshape(-1)

                    def fn(psi, objs=objs, Lam=Lam, mean_delta=mean_delta, nA=nA, dim=dim):
                        m_, dl = psi[:P], psi[P:]
                        r = dl - mean_delta
                        lp = -0.5 * float(m_ @ m_) / s0 ** 2 - 0.5 * float(r @ Lam @ r)
                        g = np.zeros(dim)
                        H = np.zeros((dim, dim))
                        g[:P] = -m_ / s0 ** 2
                        H[:P, :P] = -np.eye(P) / s0 ** 2
                        g[P:] = -Lam @ r
                        H[P:, P:] = -Lam
                        for k_, o in enumerate(objs):
                            th = m_ + (dl[(k_ - 1) * P:k_ * P] if k_ > 0 else 0.0)
                            ll, gg, hh = o.stats(th)
                            lp += ll
                            g[:P] += gg
                            H[:P, :P] += hh
                            if k_ > 0:
                                sl = slice(P + (k_ - 1) * P, P + k_ * P)
                                g[sl] += gg
                                H[sl, sl] += hh
                                H[:P, sl] += hh
                                H[sl, :P] += hh
                        return lp, g, H
                    psi, acc, _ = _mala(psi0, fn, h_study, rng)
                    mu[s], delta[s] = psi[:P], psi[P:].reshape(c, P)
                if adapt:
                    h_study = float(np.clip(h_study * (1.05 if acc else 0.98), 0.2, 1.5))
                accept["n"] += 1
                accept["a"] += int(acc)
            # ---- treatment effects
            if not random_effects:
                def fn_d(psi):
                    lp = -0.5 * float(psi @ psi) / s0 ** 2
                    g = -psi / s0 ** 2
                    H = -np.eye(D) / s0 ** 2
                    for s in sid:
                        tb = tindex[studies[s][0].treatment]
                        for o, arm in zip(arm_objs[s], studies[s]):
                            ta = tindex[arm.treatment]
                            Sel = np.zeros((P, D))
                            for k_, sign in ((ta, 1.0), (tb, -1.0)):
                                if k_ != 0:
                                    Sel[:, col[k_] * P:(col[k_] + 1) * P] += sign * np.eye(P)
                            th = mu[s] + Sel @ psi
                            ll, gg, hh = o.stats(th)
                            lp += ll
                            g = g + Sel.T @ gg
                            H = H + Sel.T @ hh @ Sel
                    return lp, g, H
                new, acc, _ = _mala(d[1:].reshape(-1), fn_d, h_d, rng)
                d[1:] = new.reshape(K - 1, P)
                if adapt:
                    h_d = float(np.clip(h_d * (1.05 if acc else 0.98), 0.05, 1.5))
            else:
                for m in range(P):
                    prec = np.eye(K - 1) / s0 ** 2
                    rhs = np.zeros(K - 1)
                    for s in sid:
                        nA = len(arm_objs[s])
                        c = nA - 1
                        Sinv = 2.0 * (np.eye(c) - np.ones((c, c)) / (c + 1)) / tau[m] ** 2
                        X = np.zeros((c, K - 1))
                        tb = tindex[studies[s][0].treatment]
                        for i in range(1, nA):
                            ta = tindex[studies[s][i].treatment]
                            if ta != 0:
                                X[i - 1, col[ta]] += 1.0
                            if tb != 0:
                                X[i - 1, col[tb]] -= 1.0
                        prec += X.T @ Sinv @ X
                        rhs += X.T @ Sinv @ delta[s][:, m]
                    cov = np.linalg.inv(prec)
                    cov = (cov + cov.T) / 2.0
                    d[1:, m] = rng.multivariate_normal(cov @ rhs, cov)
                # tau_m random walk on log tau_m, half-normal(0, tau_scale) prior (with Jacobian)
                for m in range(P):
                    def logp(t):
                        lp = -0.5 * (t / st.tau_scale) ** 2 + math.log(t)
                        for s in sid:
                            nA = len(arm_objs[s])
                            c = nA - 1
                            tb = tindex[studies[s][0].treatment]
                            r = np.array([delta[s][i - 1, m] - (d[tindex[studies[s][i].treatment], m] - d[tb, m])
                                          for i in range(1, nA)])
                            Sinv = 2.0 * (np.eye(c) - np.ones((c, c)) / (c + 1))
                            lp += -c * math.log(t) - 0.5 * float(r @ Sinv @ r) / t ** 2
                        return lp
                    new = tau[m] * math.exp(0.3 * rng.standard_normal())
                    if math.log(rng.random()) < logp(new) - logp(tau[m]):
                        tau[m] = new
            if not adapt:
                j = it - st.warmup
                d_store[chain, j] = d
                tau_store[chain, j] = tau
                dev = 0.0
                for s in sid:
                    tb = tindex[studies[s][0].treatment]
                    for i, o in enumerate(arm_objs[s]):
                        if random_effects:
                            th = mu[s] + (delta[s][i - 1] if i > 0 else 0.0)
                        else:
                            th = mu[s] + d[tindex[studies[s][i].treatment]] - d[tb]
                        theta_store[(s, o.treatment)][chain, j] = th
                        dev += o.deviance(th)
                dev_store[chain, j] = dev
    # ---- summaries
    out.d_draws = d_store.reshape(S_keep, K, P)
    out.tau_draws = tau_store.reshape(S_keep, P) if random_effects else None
    out.theta_draws = {k: v.reshape(S_keep, P) for k, v in theta_store.items()}
    for s, v in arm_objs.items():
        first = studies[s][0].treatment
        out.study_reference_theta[s] = out.theta_draws[(s, first)] - out.d_draws[:, tindex[first], :]
    level = st.ci_level
    for k in range(1, K):
        for m in range(P):
            out.diagnostics.append(_summary(f"d[{names[k]}] {out.coefficient_names[m]}", d_store[:, :, k, m], level))
    if random_effects:
        for m in range(P):
            out.diagnostics.append(_summary(f"tau {out.coefficient_names[m]}", tau_store[:, :, m], level))
    bad = [x.name for x in out.diagnostics if not (x.rhat < 1.1)]
    if bad:
        out.warnings.append("R-hat above 1.1 for: " + ", ".join(bad[:6]) + ". Run longer or use a simpler model.")
    dbar = float(dev_store.mean())
    dev_hat = 0.0
    for s, v in arm_objs.items():
        for o in v:
            dev_hat += o.deviance(out.theta_draws[(s, o.treatment)].mean(axis=0))
    out.dbar, out.p_d = dbar, dbar - dev_hat
    out.dic = dbar + out.p_d
    return out


# --------------------------------------------------------------------------
# Survival curves and exports
# --------------------------------------------------------------------------


def baseline_draws(result: SurvivalResult, baseline: str = "pooled") -> np.ndarray:
    """(S, P) log-hazard coefficients of the reference treatment: from one study, or the mean over studies that
    have a reference arm ("pooled")."""

    if baseline != "pooled":
        if baseline not in result.study_reference_theta:
            raise ValueError(f"Unknown baseline study '{baseline}'")
        return result.study_reference_theta[baseline]
    keep = [v for (s, t), v in result.theta_draws.items() if t == result.reference]
    if not keep:
        return next(iter(result.study_reference_theta.values()))
    return np.mean(keep, axis=0)


def cumulative_hazard(theta: np.ndarray, times: np.ndarray, powers: list[float], scale: float) -> np.ndarray:
    """H(t) on a grid for coefficient draws theta (S, P) -> (S, T); times (original units) start at 0 and increase.
    The model hazard h_s is per unit of scaled time, so H(t) = integral of h_s up to t / scale."""

    times = np.asarray(times, float)
    a, b = times[:-1] / scale, times[1:] / scale
    t, w = _interval_nodes(a, b)
    Phi = basis(np.maximum(t, 1e-12), powers)  # (T-1, Q, P)
    haz = np.exp(np.clip(np.einsum("tqp,sp->stq", Phi, theta), -40, 25))
    steps = (haz * w[None, :, :]).sum(axis=2) * (b - a)[None, :]  # H is the integral of h_s over scaled time
    return np.concatenate([np.zeros((theta.shape[0], 1)), np.cumsum(steps, axis=1)], axis=1)


def survival_draws(result: SurvivalResult, times, baseline: str = "pooled", max_draws: int = 1000) -> dict[str, np.ndarray]:
    """Survival curves S_k(t) = exp(-H_k(t)) per treatment for a thinned set of posterior draws: {treatment: (S, T)}.
    Draws are aligned across treatments, so PSA can sample one row for all arms."""

    if result.d_draws is None:
        raise ValueError("No draws")
    base = baseline_draws(result, baseline)
    n = base.shape[0]
    idx = np.linspace(0, n - 1, min(max_draws, n)).astype(int)
    out = {}
    for k, name in enumerate(result.treatments):
        theta = base[idx] + result.d_draws[idx, k, :]
        out[name] = np.exp(-cumulative_hazard(theta, times, result.powers, result.scale))
    return out


def survival_summary(result: SurvivalResult, times, baseline: str = "pooled", level: float = 0.95) -> dict[str, dict[str, np.ndarray]]:
    draws = survival_draws(result, times, baseline)
    alpha = (1 - level) / 2
    return {k: {"median": np.median(v, axis=0), "lower": np.quantile(v, alpha, axis=0),
                "upper": np.quantile(v, 1 - alpha, axis=0)} for k, v in draws.items()}


# --------------------------------------------------------------------------
# Export to the economic model (PSM / Markov)
# --------------------------------------------------------------------------


def survival_table(result: SurvivalResult, times, baseline: str = "pooled", level: float = 0.95):
    """Long table (pandas): time, treatment, survival median, lower, upper. A lookup table for models that
    cannot take the fitted curve in closed form (fractional polynomials of order 2)."""

    import pandas as pd

    times = np.asarray(times, float)
    summary = survival_summary(result, times, baseline, level)
    rows = []
    for name, v in summary.items():
        for i, t in enumerate(times):
            rows.append({"time": float(t), "treatment": name, "survival": float(v["median"][i]),
                         "lower": float(v["lower"][i]), "upper": float(v["upper"][i])})
    return pd.DataFrame(rows)


def psa_survival_draws(result: SurvivalResult, times, baseline: str = "pooled", n_draws: int = 500):
    """Posterior survival curves for PSA: a long table with columns draw, time, treatment, survival. The same
    draw index refers to the same posterior sample for every treatment (correlation is preserved)."""

    import pandas as pd

    times = np.asarray(times, float)
    curves = survival_draws(result, times, baseline, max_draws=n_draws)
    rows = []
    for name, v in curves.items():
        for i in range(v.shape[0]):
            for j, t in enumerate(times):
                rows.append((i, float(t), name, float(v[i, j])))
    return pd.DataFrame(rows, columns=["draw", "time", "treatment", "survival"])


def psm_parameter_draws(result: SurvivalResult, baseline: str = "pooled", time_factor: float = 1.0,
                        n_draws: int = 500):
    """Parameters of the PSM curve of every treatment, one row per posterior draw (a pandas table with columns
    draw, treatment and the PSM parameter names). Only for the exponential, Weibull and Gompertz models, whose
    curves the PSM node has in closed form; `time_factor` converts the analysis time unit to the PSM unit
    (years): 1/12 for months. Weibull: S = exp(-(t / scale)^shape); Gompertz: S = exp(-(rate / shape)
    (exp(shape t) - 1)); exponential: S = exp(-rate t)."""

    import pandas as pd

    if result.model not in ("exponential", "weibull", "gompertz"):
        raise ValueError("Closed-form PSM parameters exist only for the exponential, Weibull and Gompertz models; "
                         "export the survival table instead")
    base = baseline_draws(result, baseline)
    n = base.shape[0]
    idx = np.linspace(0, n - 1, min(n_draws, n)).astype(int)
    T, f = result.scale, float(time_factor)
    rows = []
    for k, name in enumerate(result.treatments):
        theta = base[idx] + result.d_draws[idx, k, :]
        for i, th in enumerate(theta):
            a = float(th[0])
            if result.model == "exponential":
                params = {"rate": math.exp(a) / (T * f)}
            elif result.model == "weibull":
                kappa = float(th[1]) + 1.0
                if kappa <= 0:
                    continue  # hazard not integrable at 0: no Weibull form
                params = {"shape": kappa, "scale": T * f * (kappa * math.exp(-a)) ** (1.0 / kappa)}
            else:
                b = float(th[1])
                if abs(b) < 1e-12:
                    params = {"rate": math.exp(a) / (T * f), "shape": 1e-12}
                else:
                    params = {"shape": b / (T * f), "rate": math.exp(a) / (T * f)}
            rows.append({"draw": int(i), "treatment": name, **params})
    return pd.DataFrame(rows)
