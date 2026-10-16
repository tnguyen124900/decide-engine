"""Bayesian network meta-analysis (contrast-based, normal likelihood).

Model (same structure as nma_freq, with priors):

    y | d, tau ~ N(X d, V + tau^2 C)        (random effects marginalised)
    d_k ~ N(0, prior_sd^2)                  (vague; NICE DSU TSD 2 uses sd 100)
    tau ~ Uniform(0, tau_max)  or  Half-Normal(scale)

y are the study contrasts and V their within-study covariance (log OR,
log HR, MD, ... - built exactly as in nma_freq, multi-arm correlation
included), C the between-study structure (1 on the diagonal, 0.5 within a
multi-arm study).

Sampling is exact rather than MCMC ("DIRECT" algorithm, as in the R package
bayesmeta; Roever 2020, J Stat Softw 93(6)):
  1. the marginal posterior of tau is evaluated on a fine grid
     p(tau | y) ∝ p(tau) |S|^-1/2 |A|^-1/2 exp(-(y'S^-1 y - b'A^-1 b) / 2),
     S = V + tau^2 C, A = X'S^-1 X + I/prior_sd^2, b = X'S^-1 y,
     and tau is drawn by inverse-CDF with linear interpolation;
  2. d | tau, y ~ N(A^-1 b, A^-1) is drawn exactly.
Draws are therefore independent (no burn-in, no autocorrelation). "Chains"
are independent random streams, kept so the usual R-hat / ESS diagnostics
can be shown and checked (they should be ~1.00 and ~the number of draws).

Limitation (stated in the UI too): arm-level binary / count data enter
through their log-odds / log-rate contrasts with normal within-study
likelihood - the same approximation as the frequentist engine - not the
exact binomial / Poisson likelihood of NICE TSD 2. With very small event
counts the exact-likelihood (MCMC) model can differ somewhat.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .models import ValidationError
from .nma_freq import _assemble, build_study_blocks
from .nma_models import NetworkMetaAnalysis, NMAFit, NMAResult, RelativeEffect, is_ratio_measure
from .nma_validation import validate_nma

TAU_PRIORS = ("uniform", "half-normal", "lognormal")


@dataclass
class BayesSettings:
    effects: str = "random"  # "random" | "fixed"
    chains: int = 4
    draws_per_chain: int = 1000
    seed: int = 2026
    prior_sd: float = 100.0  # N(0, prior_sd^2) on each d_k
    tau_prior: str = "uniform"  # "uniform" | "half-normal" | "lognormal" (informative, on tau^2)
    tau_max: float = 5.0  # Uniform(0, tau_max)
    tau_scale: float = 0.5  # Half-Normal(0, tau_scale)
    tau_mu: float = -2.56  # lognormal: log tau^2 ~ N(tau_mu, tau_sigma^2) (or t with tau_df df)
    tau_sigma: float = 1.74
    tau_df: float | None = None
    tau_prior_label: str = ""  # free text for reports (e.g. "Turner 2015: all-cause mortality, pharma vs placebo")
    # Treatment classes (exact-likelihood engine only): "none", "common" (d_k = class effect) or
    # "exchangeable" (d_k ~ N(class mean, sigma_class^2)); sigma_class ~ Half-Normal(0, class_sd_scale)
    class_model: str = "none"
    class_sd_scale: float = 1.0
    grid_points: int = 2000


@dataclass
class ParameterDiagnostics:
    name: str
    mean: float
    sd: float
    median: float
    lower: float
    upper: float
    rhat: float
    ess: float


@dataclass
class BayesResult:
    reference: str | None = None
    measure: str = "OR"
    treatments: list[str] = field(default_factory=list)
    settings: BayesSettings = field(default_factory=BayesSettings)
    fit: NMAFit | None = None  # effects/league from posterior quantiles; p_scores = SUCRA
    tau: ParameterDiagnostics | None = None
    diagnostics: list[ParameterDiagnostics] = field(default_factory=list)
    # rank_probabilities[treatment][r] = P(rank r + 1), rank 1 = best
    rank_probabilities: dict[str, list[float]] = field(default_factory=dict)
    sucra: dict[str, float] = field(default_factory=dict)
    mean_rank: dict[str, float] = field(default_factory=dict)
    residual_deviance: float | None = None
    n_data_points: int = 0
    p_d: float | None = None
    dic: float | None = None
    # Posterior draws on the analysis scale: rows = draws (chains stacked),
    # columns = parameter_names (every treatment except the reference).
    draws: np.ndarray | None = None
    tau_draws: np.ndarray | None = None
    warnings: list[str] = field(default_factory=list)
    errors: list[ValidationError] = field(default_factory=list)


# --------------------------------------------------------------------------
# Posterior pieces
# --------------------------------------------------------------------------


def _log_tau_prior(tau: np.ndarray, settings: BayesSettings) -> np.ndarray:
    if settings.tau_prior == "half-normal":
        return -0.5 * (tau / settings.tau_scale) ** 2
    if settings.tau_prior == "lognormal":
        with np.errstate(divide="ignore"):
            z = (2.0 * np.log(np.where(tau > 0, tau, np.nan)) - settings.tau_mu) / settings.tau_sigma
        if settings.tau_df is None:
            out = -0.5 * z * z
        else:
            out = -0.5 * (settings.tau_df + 1.0) * np.log1p(z * z / settings.tau_df)
        return np.where(tau > 0, np.nan_to_num(out, nan=-np.inf), -np.inf)
    return np.where(tau <= settings.tau_max, 0.0, -np.inf)


def tau_prior_upper(settings: BayesSettings) -> float:
    """Practical upper end of the tau range for grids / slice samplers."""

    if settings.tau_prior == "uniform":
        return settings.tau_max
    if settings.tau_prior == "half-normal":
        return settings.tau_scale * 5.0
    spread = 4.0 if settings.tau_df is None else 6.0  # heavier tails for log-t
    return float(min(50.0, math.exp(0.5 * (settings.tau_mu + spread * settings.tau_sigma))))


class _Spectral:
    """S(tau) = V + tau^2 C diagonalised once: with V = L L' and
    L^-1 C L^-T = Q diag(lam) Q', S^-1 = L^-T Q diag(1/(1+tau^2 lam)) Q' L^-1
    and log|S| = log|V| + sum log(1 + tau^2 lam). Every tau then costs only
    O(n p^2) - cheap enough to evaluate thousands of tau values exactly."""

    def __init__(self, y, X, V, C, prior_sd: float):
        L = np.linalg.cholesky(V)
        Linv_C = np.linalg.solve(L, C)
        M = np.linalg.solve(L, Linv_C.T).T
        M = (M + M.T) / 2.0
        lam, Q = np.linalg.eigh(M)
        self.lam = np.maximum(lam, 0.0)
        self.Xt = Q.T @ np.linalg.solve(L, X)
        self.yt = Q.T @ np.linalg.solve(L, y)
        self.log_det_V = 2.0 * float(np.sum(np.log(np.diag(L))))
        self.prior_precision = 1.0 / prior_sd ** 2
        self.p = X.shape[1]

    def conditional(self, tau: float):
        w = 1.0 / (1.0 + tau * tau * self.lam)
        A = (self.Xt * w[:, None]).T @ self.Xt + np.eye(self.p) * self.prior_precision
        b = self.Xt.T @ (w * self.yt)
        LA = np.linalg.cholesky(A)
        z = np.linalg.solve(LA, b)
        log_det_S = self.log_det_V - float(np.sum(np.log(w)))
        log_det_A = 2.0 * float(np.sum(np.log(np.diag(LA))))
        log_like = -0.5 * (log_det_S + log_det_A + float(np.sum(w * self.yt ** 2)) - float(z @ z))
        return A, b, LA, log_like


def tau_grid_posterior(y, X, V, C, settings: BayesSettings) -> tuple[np.ndarray, np.ndarray]:
    """Grid over tau and normalised posterior density values on it."""

    spectral = _Spectral(y, X, V, C, settings.prior_sd)
    upper = tau_prior_upper(settings)
    grid = np.linspace(0.0, upper, settings.grid_points)
    log_post = np.array([spectral.conditional(float(t))[3] for t in grid])
    log_post = log_post + _log_tau_prior(grid, settings)
    log_post -= np.max(log_post)
    density = np.exp(log_post)
    area = float(np.sum(0.5 * (density[:-1] + density[1:]) * np.diff(grid)))
    return grid, density / area


def _sample_tau(grid: np.ndarray, density: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Inverse-CDF sampling from the piecewise-linear density on the grid."""

    widths = np.diff(grid)
    segment_mass = 0.5 * (density[:-1] + density[1:]) * widths
    total = float(segment_mass.sum())
    cdf = np.concatenate([[0.0], np.cumsum(segment_mass)]) / total
    u = rng.random(n)
    index = np.clip(np.searchsorted(cdf, u, side="right") - 1, 0, len(widths) - 1)
    f0, f1, w = density[index], density[index + 1], widths[index]
    target = (u - cdf[index]) * total  # mass needed inside the segment
    slope = (f1 - f0) / w
    safe_slope = np.where(np.abs(slope) > 1e-12, slope, 1.0)
    quadratic = (-f0 + np.sqrt(np.maximum(f0 * f0 + 2.0 * safe_slope * target, 0.0))) / safe_slope
    linear = target / np.where(f0 > 0, f0, 1.0)
    offset = np.where(np.abs(slope) > 1e-12, quadratic, linear)
    return grid[index] + np.clip(offset, 0.0, w)


# --------------------------------------------------------------------------
# Diagnostics
# --------------------------------------------------------------------------


def _split_rhat(chains: np.ndarray) -> float:
    """Split-R-hat (Gelman et al. BDA3) for an array (n_chains, n_draws)."""

    n_chains, n = chains.shape
    if n < 4:
        return float("nan")
    half = n // 2
    split = np.concatenate([chains[:, :half], chains[:, half:2 * half]], axis=0)
    m, n_half = split.shape
    means = split.mean(axis=1)
    within = split.var(axis=1, ddof=1).mean()
    between = n_half * means.var(ddof=1)
    if within <= 0:
        return 1.0
    var_plus = (n_half - 1) / n_half * within + between / n_half
    return float(math.sqrt(var_plus / within))


def _ess(chains: np.ndarray) -> float:
    """Effective sample size with Geyer's initial positive sequence."""

    n_chains, n = chains.shape
    centred = chains - chains.mean(axis=1, keepdims=True)
    variance = centred.var(axis=1).mean()
    if variance <= 0:
        return float(n_chains * n)
    rho_sum = 0.0
    lag = 1
    while lag < n - 1:
        acov = np.mean([np.dot(c[:-lag], c[lag:]) / n for c in centred])
        acov_next = np.mean([np.dot(c[:-(lag + 1)], c[lag + 1:]) / n for c in centred])
        pair = (acov + acov_next) / variance
        if pair <= 0:
            break
        rho_sum += pair
        lag += 2
    return float(n_chains * n / max(1.0, 1.0 + 2.0 * rho_sum - 0.0))


def _summary(name: str, chains: np.ndarray, level: float) -> ParameterDiagnostics:
    flat = chains.reshape(-1)
    alpha = (1.0 - level) / 2.0
    return ParameterDiagnostics(
        name=name, mean=float(flat.mean()), sd=float(flat.std(ddof=1)), median=float(np.median(flat)),
        lower=float(np.quantile(flat, alpha)), upper=float(np.quantile(flat, 1.0 - alpha)),
        rhat=_split_rhat(chains), ess=_ess(chains),
    )


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def run_nma_bayesian(model: NetworkMetaAnalysis, settings: BayesSettings | None = None) -> BayesResult:
    settings = settings or BayesSettings()
    result = BayesResult(reference=model.reference, measure=model.spec.measure, settings=settings)
    result.errors = validate_nma(model)
    if settings.tau_prior not in TAU_PRIORS:
        result.errors.append(ValidationError(code="unknown_tau_prior", message=f"Unknown tau prior '{settings.tau_prior}'"))
    if settings.chains < 1 or settings.draws_per_chain < 10:
        result.errors.append(ValidationError(code="too_few_draws", message="Use at least 1 chain and 10 draws per chain"))
    if result.errors:
        return result

    scratch = NMAResult(reference=model.reference, measure=model.spec.measure)
    blocks = build_study_blocks(model, scratch)
    result.warnings.extend(scratch.warnings)
    used = {block.baseline for block in blocks} | {name for block in blocks for name in block.treatments}
    treatments = [name for name in model.treatment_names if name in used]
    reference = model.reference
    result.treatments = treatments
    if reference not in used:
        result.errors.append(ValidationError(code="reference_excluded",
                                             message=f"The reference '{reference}' has no usable data after exclusions"))
        return result
    parameters = [name for name in treatments if name != reference]
    y, X, V, C = _assemble(blocks, parameters)
    if np.linalg.matrix_rank(X) < len(parameters):
        result.errors.append(ValidationError(code="not_estimable",
                                             message="Some treatment effects cannot be estimated (disconnected network)"))
        return result

    rng = np.random.default_rng(settings.seed)
    n_chains, n_draws = settings.chains, settings.draws_per_chain
    total = n_chains * n_draws
    random_effects = settings.effects == "random"
    if random_effects:
        grid, density = tau_grid_posterior(y, X, V, C, settings)
        tau_draws = _sample_tau(grid, density, total, rng)
    else:
        tau_draws = np.zeros(total)

    # d | tau, y ~ N(A^-1 b, A^-1), exactly, for every tau draw.
    spectral = _Spectral(y, X, V, C, settings.prior_sd)
    d_draws = np.empty((total, len(parameters)))
    if random_effects:
        z = rng.standard_normal((total, len(parameters)))
        for index in range(total):
            A, b, LA, _ = spectral.conditional(float(tau_draws[index]))
            d_draws[index] = np.linalg.solve(A, b) + np.linalg.solve(LA.T, z[index])
    else:
        A, b, LA, _ = spectral.conditional(0.0)
        mean = np.linalg.solve(A, b)
        z = rng.standard_normal((total, len(parameters)))
        d_draws = mean + np.linalg.solve(LA.T, z.T).T

    result.draws = d_draws
    result.tau_draws = tau_draws
    level = model.spec.ci_level
    chains_of = lambda values: values.reshape(n_chains, n_draws)  # noqa: E731
    for column, name in enumerate(parameters):
        result.diagnostics.append(_summary(f"d[{name}]", chains_of(d_draws[:, column]), level))
    if random_effects:
        result.tau = _summary("tau", chains_of(tau_draws), level)
        result.diagnostics.append(result.tau)

    result.fit = _posterior_fit(d_draws, tau_draws, treatments, reference, parameters,
                                model.spec.measure, level, random_effects)
    _rankings(result, d_draws, treatments, reference, parameters, model.spec.small_values)
    result.fit.p_scores = dict(result.sucra)
    _deviance(result, y, X, V, C, blocks, d_draws, tau_draws, random_effects, rng)
    return result


def _posterior_fit(d_draws, tau_draws, treatments, reference, parameters, measure, level, random_effects) -> NMAFit:
    ratio = is_ratio_measure(measure)
    transform = math.exp if ratio else (lambda value: value)
    alpha = (1.0 - level) / 2.0
    full = np.zeros((d_draws.shape[0], len(treatments)))
    index = {name: i for i, name in enumerate(treatments)}
    for column, name in enumerate(parameters):
        full[:, index[name]] = d_draws[:, column]

    def effect(a: str, b: str) -> RelativeEffect:
        values = full[:, index[a]] - full[:, index[b]]
        median = float(np.median(values))
        lower, upper = (float(q) for q in np.quantile(values, [alpha, 1.0 - alpha]))
        return RelativeEffect(
            treatment=a, versus=b, estimate=median, se=float(values.std(ddof=1)), lower=lower, upper=upper,
            natural=transform(median), natural_lower=transform(lower), natural_upper=transform(upper),
        )

    covariance = np.cov(d_draws, rowvar=False).reshape(len(parameters), len(parameters))
    fit = NMAFit(
        model="bayesian-random" if random_effects else "bayesian-fixed",
        tau2=float(np.median(tau_draws) ** 2) if random_effects else 0.0,
        parameter_names=list(parameters),
        covariance=covariance.tolist(),
    )
    fit.effects = [effect(name, reference) for name in parameters]
    fit.league = {(a, b): effect(a, b) for a in treatments for b in treatments if a != b}
    return fit


def _rankings(result: BayesResult, d_draws, treatments, reference, parameters, small_values: str) -> None:
    n = len(treatments)
    full = np.zeros((d_draws.shape[0], n))
    index = {name: i for i, name in enumerate(treatments)}
    for column, name in enumerate(parameters):
        full[:, index[name]] = d_draws[:, column]
    # rank 1 = best: smallest effect when small values are good.
    order = np.argsort(full if small_values == "good" else -full, axis=1)
    ranks = np.empty_like(order)
    rows = np.arange(full.shape[0])[:, None]
    ranks[rows, order] = np.arange(n)[None, :]
    for name in treatments:
        counts = np.bincount(ranks[:, index[name]], minlength=n) / full.shape[0]
        result.rank_probabilities[name] = counts.tolist()
        cumulative = np.cumsum(counts)[:-1]
        result.sucra[name] = float(cumulative.sum() / (n - 1)) if n > 1 else 1.0
        result.mean_rank[name] = float(np.sum((np.arange(n) + 1) * counts))


def _deviance(result: BayesResult, y, X, V, C, blocks, d_draws, tau_draws, random_effects, rng) -> None:
    """Posterior mean residual deviance, pD and DIC (TSD 2 definitions for a
    normal likelihood). Random effects: study-level deltas are drawn from
    their exact conditional given (d, tau, y), one draw per posterior draw."""

    n_draws = d_draws.shape[0]
    fitted_mean = X @ d_draws.T  # (n_obs, n_draws)
    theta = fitted_mean.copy()
    if random_effects:
        row = 0
        for block in blocks:
            k = len(block.treatments)
            rows = slice(row, row + k)
            Vb = V[rows, rows]
            Cb = C[rows, rows]
            Vb_inv = np.linalg.inv(Vb)
            Cb_inv = np.linalg.inv(Cb)
            tau2 = np.maximum(tau_draws, 1e-10) ** 2
            precision = Vb_inv[None, :, :] + Cb_inv[None, :, :] / tau2[:, None, None]  # (n, k, k)
            covariance = np.linalg.inv(precision)
            residual = (y[rows][:, None] - fitted_mean[rows]).T  # (n, k)
            mean = np.einsum("nij,jk,nk->ni", covariance, Vb_inv, residual)
            chol = np.linalg.cholesky(covariance)
            delta = mean + np.einsum("nij,nj->ni", chol, rng.standard_normal((n_draws, k)))
            theta[rows] += delta.T
            row += k
    W = np.linalg.inv(V)
    residuals = y[:, None] - theta  # (n_obs, n_draws)
    deviance_draws = np.einsum("in,ij,jn->n", residuals, W, residuals)
    theta_bar = theta.mean(axis=1)
    deviance_at_mean = float((y - theta_bar) @ W @ (y - theta_bar))
    result.residual_deviance = float(deviance_draws.mean())
    result.p_d = result.residual_deviance - deviance_at_mean
    result.dic = result.residual_deviance + result.p_d
    result.n_data_points = int(len(y))
