"""Bayesian component network meta-analysis (CNMA).

The additive component model (Rücker et al. 2020, Biom J 62:808-821, the
model netmeta::netcomb fits by GLS) says a treatment's effect versus the
reference is the SUM of the effects of the components it adds on top of the
reference:

    d_k - d_ref = (s_k - s_ref)' c,     s_k in {0,1}^m the component
    indicator vector of treatment k,  c the m component effects.

This is the Bayesian version: the same component design matrix B that
nma_components.run_component_nma builds is folded into the likelihood's
design matrix (Xc = X @ B), and the exact-conjugate Gibbs sampler of
nma_bayes is run on the m component effects instead of the p treatment
effects. Implied treatment effects are recovered per posterior draw as
d_draws = c_draws @ B.T, so every downstream NMA output (league table,
rank probabilities / SUCRA, EVPI/EVPPI over the network posterior) works
unchanged.

Interactions (a component pair whose combined effect is not the sum of the
two separate effects) are added as extra columns of B, each its own
parameter -- the "interaction" model of Welton et al. 2009 (Am J Epidemiol
169:1158-1165). The additive model is the special case with no interaction
columns.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .models import ValidationError
from .nma_bayes import (
    BayesSettings,
    _rankings,
    _sample_tau,
    _Spectral,
    _summary,
    tau_grid_posterior,
)
from .nma_components import parse_components
from .nma_freq import NMAResult, build_study_blocks, _assemble
from .nma_models import NetworkMetaAnalysis, NMAFit, RelativeEffect, is_ratio_measure
from .nma_validation import validate_nma

__all__ = ["BayesComponentSettings", "BayesComponentResult", "run_component_nma_bayes"]


@dataclass
class BayesComponentSettings:
    """MCMC settings for the component model, plus the interactions list."""

    bayes: BayesSettings = field(default_factory=BayesSettings)
    # interactions: list of component pairs [a, b] whose combined effect gets
    # its own parameter (the "interaction" columns of B). Empty = the pure
    # additive model.
    interactions: list[tuple[str, str]] = field(default_factory=list)


@dataclass
class BayesComponentResult:
    measure: str = "OR"
    reference: str | None = None
    treatments: list[str] = field(default_factory=list)
    components: list[str] = field(default_factory=list)
    membership: dict[str, list[str]] = field(default_factory=dict)
    interactions: list[tuple[str, str]] = field(default_factory=list)
    # Posterior draws: component_draws (n_draws, m) on component effects;
    # treatment_draws (n_draws, p) on implied treatment effects vs reference.
    component_draws: np.ndarray | None = None
    treatment_draws: np.ndarray | None = None
    tau_draws: np.ndarray | None = None
    component_effects: list[RelativeEffect] = field(default_factory=list)
    fit: NMAFit | None = None
    tau: object | None = None
    diagnostics: list = field(default_factory=list)
    rank_probabilities: dict[str, list[float]] = field(default_factory=dict)
    sucra: dict[str, float] = field(default_factory=dict)
    mean_rank: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list = field(default_factory=list)


def _component_design(
    model: NetworkMetaAnalysis,
    components: dict[str, list[str] | str] | None,
    interactions: list[tuple[str, str]],
    treatments: list[str],
    reference: str,
) -> tuple[np.ndarray, list[str], dict[str, list[str]]]:
    """Build the (p, m) component design matrix B.

    m = number of distinct components + one column per interaction pair.
    B[k, c] = (s_k - s_ref)_c for the treatment k's component indicator s_k.
    """

    parameters = [name for name in treatments if name != reference]
    membership: dict[str, list[str]] = {}
    given = components or {}
    for name in treatments:
        parts = parse_components(given.get(name))
        if not parts and name != reference:
            parts = [name]
        membership[name] = parts

    names: list[str] = []
    for name in treatments:
        for part in membership[name]:
            if part not in names:
                names.append(part)
    interaction_names = [f"{a}:{b}" for a, b in interactions]
    all_columns = names + interaction_names
    m = len(all_columns)
    index = {c: i for i, c in enumerate(all_columns)}

    def vector(treatment: str) -> np.ndarray:
        v = np.zeros(m)
        parts = set(membership[treatment])
        for part in parts:
            v[index[part]] = 1.0
        for (a, b), col in zip(interactions, interaction_names):
            if a in parts and b in parts:
                v[index[col]] = 1.0
        return v

    B = np.array([vector(t) - vector(reference) for t in parameters])
    return B, all_columns, membership


def run_component_nma_bayes(
    model: NetworkMetaAnalysis,
    components: dict[str, list[str] | str] | None = None,
    settings: BayesComponentSettings | None = None,
) -> BayesComponentResult:
    """Fit the additive (+ optional interaction) component model by MCMC.

    Same posterior machinery as run_nma_bayesian, run on the component
    design matrix: d | tau, y ~ Normal exact, tau by grid sampler. The
    returned treatment_draws / fit / rankings are over the IMPLIED treatment
    effects (d = B c), so every downstream NMA output is directly comparable
    to the standard Bayesian NMA's.
    """

    settings = settings or BayesComponentSettings()
    bayes = settings.bayes
    out = BayesComponentResult(measure=model.spec.measure, reference=model.reference)
    out.errors = validate_nma(model)
    if out.errors:
        return out

    scratch = NMAResult(reference=model.reference, measure=model.spec.measure)
    blocks = build_study_blocks(model, scratch)
    out.warnings.extend(scratch.warnings)
    used = {block.baseline for block in blocks} | {name for block in blocks for name in block.treatments}
    treatments = [name for name in model.treatment_names if name in used]
    reference = model.reference
    out.treatments = treatments
    if reference not in used:
        out.errors.append(ValidationError(
            code="reference_excluded",
            message=f"The reference '{reference}' has no usable data after exclusions"))
        return out

    parameters = [name for name in treatments if name != reference]
    y, X, V, C = _assemble(blocks, parameters)
    B, component_names, membership = _component_design(
        model, components, settings.interactions, treatments, reference)
    out.components = component_names
    out.membership = membership
    out.interactions = list(settings.interactions)

    Xc = X @ B
    m = len(component_names)
    if np.linalg.matrix_rank(Xc) < m:
        out.errors.append(ValidationError(
            code="components_not_identified",
            message="The component effects are not identifiable: some components always occur together "
                    "(add treatments with the components separately or in other combinations)."))
        return out

    rng = np.random.default_rng(bayes.seed)
    n_chains, n_draws = bayes.chains, bayes.draws_per_chain
    total = n_chains * n_draws
    random_effects = bayes.effects == "random"

    if random_effects:
        grid, density = tau_grid_posterior(y, Xc, V, C, bayes)
        tau_draws = _sample_tau(grid, density, total, rng)
    else:
        tau_draws = np.zeros(total)

    # c | tau, y ~ N(A^-1 b, A^-1), exactly, for every tau draw -- the same
    # conjugate conditional as the standard model, on the component design.
    spectral = _Spectral(y, Xc, V, C, bayes.prior_sd)
    c_draws = np.empty((total, m))
    if random_effects:
        z = rng.standard_normal((total, m))
        for index in range(total):
            A, b, LA, _ = spectral.conditional(float(tau_draws[index]))
            c_draws[index] = np.linalg.solve(A, b) + np.linalg.solve(LA.T, z[index])
    else:
        A, b, LA, _ = spectral.conditional(0.0)
        mean = np.linalg.solve(A, b)
        z = rng.standard_normal((total, m))
        c_draws = mean + np.linalg.solve(LA.T, z.T).T

    out.component_draws = c_draws
    out.tau_draws = tau_draws
    # Implied treatment effects vs reference: d_k = B_k' c, per draw.
    d_draws = c_draws @ B.T
    out.treatment_draws = d_draws

    level = model.spec.ci_level
    chains_of = lambda values: values.reshape(n_chains, n_draws)  # noqa: E731
    for column, name in enumerate(component_names):
        out.diagnostics.append(_summary(f"c[{name}]", chains_of(c_draws[:, column]), level))
    if random_effects:
        out.tau = _summary("tau", chains_of(tau_draws), level)
        out.diagnostics.append(out.tau)

    # Component effect summaries (posterior median + credible interval).
    ratio = is_ratio_measure(model.spec.measure)
    transform = math.exp if ratio else (lambda value: value)
    alpha = (1.0 - level) / 2.0
    out.component_effects = []
    for column, name in enumerate(component_names):
        values = c_draws[:, column]
        median = float(np.median(values))
        lower, upper = (float(q) for q in np.quantile(values, [alpha, 1.0 - alpha]))
        out.component_effects.append(RelativeEffect(
            treatment=name, versus="(component)", estimate=median,
            se=float(values.std(ddof=1)), lower=lower, upper=upper,
            natural=transform(median), natural_lower=transform(lower),
            natural_upper=transform(upper),
        ))

    # Reuse the standard posterior fit + rankings on the implied treatment
    # effects, so league tables / SUCRA compare directly to the standard NMA.
    from .nma_bayes import _posterior_fit  # local import to avoid a cycle

    out.fit = _posterior_fit(d_draws, tau_draws, treatments, reference, parameters,
                             model.spec.measure, level, random_effects)
    _rankings_bayes(out, d_draws, treatments, reference, parameters, model.spec.small_values)
    out.fit.p_scores = dict(out.sucra)
    return out


def _rankings_bayes(result: BayesComponentResult, d_draws, treatments, reference, parameters, small_values: str) -> None:
    """Rank probabilities / SUCRA over the implied treatment effects.

    Mirrors nma_bayes._rankings but writes into a BayesComponentResult.
    """

    n = len(treatments)
    full = np.zeros((d_draws.shape[0], n))
    index = {name: i for i, name in enumerate(treatments)}
    for column, name in enumerate(parameters):
        full[:, index[name]] = d_draws[:, column]
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
