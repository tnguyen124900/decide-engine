"""Multilevel network meta-regression for a binary outcome (ML-NMR; Phillippo et al. 2020, JRSS A 183:1189-1210;
NICE DSU TSD 18), by maximum likelihood with a normal approximation.

Individual-level model for arm a of study j with treatment t (reference treatment: no treatment terms)
    logit p = mu_j + x' beta + d_t + x_EM' gamma_t ,           x centred at c,
with beta the (common) prognostic coefficients, d_t the effect of t versus the reference at x = c and gamma_t the
effect-modifier interactions (shared by all active treatments or one set per treatment).
  * IPD arms enter with the individual Bernoulli likelihood.
  * Aggregate arms (events, n) enter with the *integrated* likelihood: the arm risk is the mean of expit(eta) over the
    arm's covariate distribution, approximated by quasi-Monte Carlo points (Halton, Gaussian copula with the
    correlation of the pooled IPD; continuous covariates normal, binary covariates Bernoulli). This integration is
    what keeps the model free of aggregation bias, unlike a meta-regression on the mean covariate.
Estimation: Newton's method with the exact gradient and Hessian, covariance = inverse of the observed information.
Effects are reported at any covariate profile: d_t + (x* - c)' gamma_t (a conditional log odds ratio).
Limits: maximum likelihood with study-specific intercepts (a Bayesian or random-effects version is not
included); with no IPD the interactions are estimated from between-study variation only (ecological bias)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .models import ValidationError
from .nma_parser import _normal_quantile

_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]


def halton(n: int, dim: int) -> np.ndarray:
    out = np.zeros((n, dim))
    for j in range(dim):
        base = _PRIMES[j]
        for i in range(n):
            f, r, k = 1.0, 0.0, i + 1
            while k > 0:
                f /= base
                r += f * (k % base)
                k //= base
            out[i, j] = r
    return out


@dataclass
class IPDArm:
    study: str
    treatment: str
    x: np.ndarray  # (n, p) covariates
    y: np.ndarray  # (n,) 0/1


@dataclass
class AggArm:
    study: str
    treatment: str
    events: float
    n: float
    mean: np.ndarray  # (p,) covariate means (proportion for binary)
    sd: np.ndarray  # (p,) SDs (ignored for binary)


@dataclass
class MlnmrResult:
    covariates: list[str] = field(default_factory=list)
    effect_modifiers: list[str] = field(default_factory=list)
    treatments: list[str] = field(default_factory=list)
    reference: str | None = None
    center: np.ndarray | None = None
    names: list[str] = field(default_factory=list)
    theta: np.ndarray | None = None
    cov: np.ndarray | None = None
    loglik: float = float("nan")
    n_ipd: int = 0
    n_agg: int = 0
    warnings: list[str] = field(default_factory=list)
    errors: list[ValidationError] = field(default_factory=list)
    _layout: dict = field(default_factory=dict)
    _evaluate: object = None  # (theta) -> (loglik, gradient, hessian); exposed for testing

    def se(self, name: str) -> float:
        i = self.names.index(name)
        return float(math.sqrt(self.cov[i, i]))

    def effect(self, treatment: str, profile) -> tuple[float, float]:
        """Conditional log OR of `treatment` versus the reference at covariate values `profile` (original scale)."""

        if treatment == self.reference:
            return 0.0, 0.0
        L = self._layout
        v = np.zeros(len(self.theta))
        v[L["d"] + L["tindex"][treatment]] = 1.0
        xs = np.asarray(profile, float) - self.center
        em = [self.covariates.index(c) for c in self.effect_modifiers]
        for g, j in enumerate(em):
            pos = L["g"] + g if L["shared"] else L["g"] + L["tindex"][treatment] * len(em) + g
            v[pos] = xs[j]
        return float(v @ self.theta), float(math.sqrt(max(v @ self.cov @ v, 0.0)))


def _copula_points(mean, sd, binary, corr, u):
    p = len(mean)
    z = np.vectorize(_normal_quantile)(np.clip(u, 1e-6, 1 - 1e-6))
    try:
        chol = np.linalg.cholesky(corr + 1e-9 * np.eye(p))
        z = z @ chol.T
    except np.linalg.LinAlgError:
        pass
    x = np.zeros_like(z)
    for j in range(p):
        if binary[j]:
            thresh = _normal_quantile(min(max(1 - mean[j], 1e-6), 1 - 1e-6))
            x[:, j] = (z[:, j] > thresh).astype(float)
        else:
            x[:, j] = mean[j] + sd[j] * z[:, j]
    return x


def _expit(a):
    return 1.0 / (1.0 + np.exp(-np.clip(a, -35, 35)))


def run_mlnmr(ipd: list[IPDArm], agg: list[AggArm], covariates: list[str], binary: list[bool],
              effect_modifiers: list[str], reference: str, shared: bool = True, n_points: int = 512,
              ci_level: float = 0.95) -> MlnmrResult:
    res = MlnmrResult(covariates=list(covariates), effect_modifiers=list(effect_modifiers), reference=reference)
    p = len(covariates)
    if any(c not in covariates for c in effect_modifiers):
        res.errors.append(ValidationError(code="em", message="Effect modifiers must be among the covariates"))
        return res
    arms = [(a.study, a.treatment) for a in ipd] + [(a.study, a.treatment) for a in agg]
    studies = sorted({s for s, _ in arms})
    treatments = [reference] + sorted({t for _, t in arms if t != reference})
    if reference not in {t for _, t in arms} or len(treatments) < 2:
        res.errors.append(ValidationError(code="reference", message="The reference and at least one other treatment are needed"))
        return res
    res.treatments, res.n_ipd, res.n_agg = treatments, sum(len(a.y) for a in ipd), len(agg)
    tindex = {t: i for i, t in enumerate(treatments[1:])}
    K1, J, G = len(treatments) - 1, len(studies), len(effect_modifiers)
    sindex = {s: i for i, s in enumerate(studies)}
    n_gamma = G if shared else G * K1
    dim = J + p + n_gamma + K1
    off = {"mu": 0, "b": J, "g": J + p, "d": J + p + n_gamma}
    res._layout = {"d": off["d"], "g": off["g"], "shared": shared, "tindex": tindex}
    res.names = ([f"mu[{s}]" for s in studies] + [f"beta[{c}]" for c in covariates]
                 + ([f"gamma[{c}]" for c in effect_modifiers] if shared else
                    [f"gamma[{t}, {c}]" for t in treatments[1:] for c in effect_modifiers])
                 + [f"d[{t}]" for t in treatments[1:]])
    # centring and copula correlation
    allx = [a.x for a in ipd]
    if allx:
        pooled = np.vstack(allx)
        centre = pooled.mean(axis=0)
        corr = np.corrcoef(pooled, rowvar=False) if p > 1 else np.ones((1, 1))
        corr = np.nan_to_num(corr) + 0.0
    else:
        centre = np.mean([a.mean for a in agg], axis=0)
        corr = np.eye(p)
        res.warnings.append("No IPD: the interactions are estimated from between-study variation only (ecological bias); "
                            "individual data are needed to separate within- from between-study effects")
    if p == 0:
        centre, corr = np.zeros(0), np.ones((0, 0))
    res.center = centre
    em_idx = [covariates.index(c) for c in effect_modifiers]
    u = halton(n_points, max(p, 1))[:, :p] if p > 0 else np.zeros((n_points, 0))

    def design(study, treatment, x):
        n = x.shape[0]
        D = np.zeros((n, dim))
        D[:, off["mu"] + sindex[study]] = 1.0
        xc = x - centre
        D[:, off["b"]:off["b"] + p] = xc
        if treatment != reference:
            k = tindex[treatment]
            D[:, off["d"] + k] = 1.0
            for g, j in enumerate(em_idx):
                col = off["g"] + (g if shared else k * G + g)
                D[:, col] = xc[:, j]
        return D

    blocks = []  # (kind, D, payload)
    for a in ipd:
        if a.x.shape[1] != p:
            res.errors.append(ValidationError(code="cols", message="IPD covariate columns differ"))
            return res
        blocks.append(("ipd", design(a.study, a.treatment, a.x), a.y.astype(float)))
    for a in agg:
        x = _copula_points(np.asarray(a.mean, float), np.asarray(a.sd, float), binary, corr, u) if p > 0 else np.zeros((n_points, 0))
        blocks.append(("agg", design(a.study, a.treatment, x), (float(a.events), float(a.n))))

    def evaluate(theta, need_hessian=True):
        ll, grad, hess = 0.0, np.zeros(dim), np.zeros((dim, dim))
        for kind, D, payload in blocks:
            pr = _expit(D @ theta)
            if kind == "ipd":
                y = payload
                ll += float(np.sum(y * np.log(np.maximum(pr, 1e-300)) + (1 - y) * np.log(np.maximum(1 - pr, 1e-300))))
                grad += D.T @ (y - pr)
                if need_hessian:
                    hess -= (D * (pr * (1 - pr))[:, None]).T @ D
            else:
                e, n = payload
                pbar = float(np.clip(pr.mean(), 1e-12, 1 - 1e-12))
                ll += e * math.log(pbar) + (n - e) * math.log(1 - pbar)
                w1 = pr * (1 - pr)
                dp = D.T @ w1 / len(pr)
                coef = e / pbar - (n - e) / (1 - pbar)
                grad += coef * dp
                if need_hessian:
                    d2 = (D * (w1 * (1 - 2 * pr))[:, None]).T @ D / len(pr)
                    hess += coef * d2 - (e / pbar ** 2 + (n - e) / (1 - pbar) ** 2) * np.outer(dp, dp)
        return ll, grad, hess

    res._evaluate = evaluate
    theta = np.zeros(dim)
    for s in studies:  # start each intercept at the logit of the study's event rate
        e = sum(float(a.y.sum()) for a in ipd if a.study == s) + sum(a.events for a in agg if a.study == s)
        n = sum(len(a.y) for a in ipd if a.study == s) + sum(a.n for a in agg if a.study == s)
        r = min(max(e / max(n, 1.0), 0.01), 0.99)
        theta[sindex[s]] = math.log(r / (1 - r))
    ll, grad, hess = evaluate(theta)
    for _ in range(100):
        try:
            step = np.linalg.solve(-hess - 1e-9 * np.eye(dim), grad)
        except np.linalg.LinAlgError:
            res.errors.append(ValidationError(code="singular", message="The information matrix is singular: some parameters are not identified"))
            return res
        t = 1.0
        while t > 1e-6:
            new_ll = evaluate(theta + t * step, need_hessian=False)[0]
            if np.isfinite(new_ll) and new_ll >= ll - 1e-10:
                break
            t /= 2.0
        theta = theta + t * step
        ll, grad, hess = evaluate(theta)
        if float(np.max(np.abs(t * step))) < 1e-9:
            break
    else:
        res.warnings.append("Newton iterations did not fully converge")
    info = -hess
    if np.linalg.matrix_rank(info, tol=1e-8) < dim:
        res.errors.append(ValidationError(code="singular", message="The information matrix is singular: some parameters are not identified "
                                                                    "(check that every treatment is connected and every effect modifier varies)"))
        return res
    res.theta, res.cov, res.loglik = theta, np.linalg.inv(info), float(ll)
    return res
