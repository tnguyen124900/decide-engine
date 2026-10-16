"""Frequentist (contrast-based) network meta-analysis.

Model (consistency, multivariate normal within-study likelihood):

    y = X d + b + e,    e ~ N(0, V),    b ~ N(0, tau^2 C)

- y: study-level contrasts versus each study's baseline arm (analysis scale)
- X: design matrix; row "k vs b" has +1 for treatment k, -1 for b; the
  column of the network reference is dropped (d_reference = 0)
- V: block-diagonal within-study covariance; in a multi-arm study the
  contrasts share their baseline arm, so Cov(y_bk, y_bl) = var(baseline arm)
- C: between-study correlation structure - 1 on the diagonal, 0.5 between
  contrasts of the same multi-arm study (common heterogeneity tau^2)

Fixed effect: generalised least squares with W = V^-1.
Random effects: tau^2 by the method of moments generalised to NMA
(DerSimonian-Laird; Jackson, White & Riley 2012):

    tau^2 = max(0, (Q - df) / tr(P C)),   P = W - W X (X'WX)^-1 X'W

which reduces to the classic DerSimonian-Laird estimator for a pairwise
meta-analysis. GLS with the full covariance is invariant to which arm of a
multi-arm study is chosen as its baseline.

Only numpy is required (no scipy), so the engine runs wherever the app runs.
This engine is the fast check that runs before the Bayesian engine; its
results are also what the Bayesian engine must match approximately under
vague priors.
"""

from __future__ import annotations

import math

import numpy as np

from .models import ValidationError
from .nma_models import (
    ContrastObservation,
    Heterogeneity,
    NetworkMetaAnalysis,
    NMAFit,
    NMAResult,
    RelativeEffect,
    Study,
    is_ratio_measure,
)
from .nma_parser import _normal_quantile
from .nma_validation import validate_nma


# --------------------------------------------------------------------------
# Small numerical helpers (kept local to avoid a scipy dependency)
# --------------------------------------------------------------------------


def normal_cdf(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def chi2_sf(x: float, df: int) -> float:
    """Upper tail of the chi-square distribution: Q(df/2, x/2)."""

    if df <= 0:
        return float("nan")
    if x <= 0:
        return 1.0
    return _gamma_q(df / 2.0, x / 2.0)


def _gamma_q(a: float, x: float) -> float:
    """Regularised upper incomplete gamma function (Numerical Recipes 6.2)."""

    gln = math.lgamma(a)
    if x < a + 1.0:
        term = total = 1.0 / a
        ap = a
        for _ in range(1000):
            ap += 1.0
            term *= x / ap
            total += term
            if abs(term) < abs(total) * 1e-15:
                break
        return max(0.0, 1.0 - total * math.exp(-x + a * math.log(x) - gln))
    tiny = 1e-300
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return math.exp(-x + a * math.log(x) - gln) * h


def _beta_cf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function (NR 6.4)."""

    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 1000):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return h


def _beta_inc(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta I_x(a, b)."""

    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _beta_cf(a, b, x) / a
    return 1.0 - front * _beta_cf(b, a, 1.0 - x) / b


def t_cdf(x: float, df: float) -> float:
    tail = 0.5 * _beta_inc(df / 2.0, 0.5, df / (df + x * x))
    return 1.0 - tail if x >= 0 else tail


def t_quantile(p: float, df: float) -> float:
    """Inverse of t_cdf by bisection (monotone, so this is safe and exact
    to ~1e-12; speed is irrelevant at a few calls per run)."""

    if not 0 < p < 1:
        raise ValueError("p must be in (0, 1)")
    if p < 0.5:
        return -t_quantile(1 - p, df)
    lo, hi = 0.0, 1.0
    while t_cdf(hi, df) < p:
        hi *= 2.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-13:
            break
    return (lo + hi) / 2.0


# --------------------------------------------------------------------------
# Arm-level data -> study contrasts
# --------------------------------------------------------------------------


class _StudyBlock:
    """Contrasts of one study against its baseline arm + their covariance."""

    def __init__(self, study_id: str, baseline: str, treatments: list[str],
                 estimates: list[float], variances: list[float], baseline_variance: float):
        self.study_id = study_id
        self.baseline = baseline
        self.treatments = treatments  # non-baseline arms, one per contrast
        self.estimates = estimates
        self.variances = variances  # var(y_bk) = var_k + var_b
        self.baseline_variance = baseline_variance  # covariance between contrasts


def _choose_baseline(names: list[str], order: dict[str, int], reference: str | None) -> str:
    if reference in names:
        return reference  # type: ignore[return-value]
    return min(names, key=lambda name: order.get(name, len(order)))


def _arm_summaries(study: Study, outcome: str, zero_correction: float,
                   measure: str = "OR") -> tuple[dict[str, tuple[float, float]], bool]:
    """Per-arm (value on analysis scale, variance). Returns (summaries, corrected).

    Binary: log odds (OR), log risk (RR, var 1/a - 1/n) or risk (RD, var
    p(1-p)/n); zero cells get `zero_correction` added to events and
    non-events of every arm of the study. The contrast of two arms is the
    difference of their values, its variance the sum."""

    corrected = False
    summaries: dict[str, tuple[float, float]] = {}
    if outcome == "binary":
        zero = any(arm.events == 0 or arm.events == arm.n for arm in study.arms)
        inc = zero_correction if zero else 0.0
        corrected = zero and inc > 0
        for arm in study.arms:
            a = arm.events + inc  # type: ignore[operator]
            b = arm.n - arm.events + inc  # type: ignore[operator]
            if measure == "RR":
                summaries[arm.treatment] = (math.log(a / (a + b)), 1.0 / a - 1.0 / (a + b))
            elif measure == "RD":
                p = a / (a + b)
                summaries[arm.treatment] = (p, p * (1.0 - p) / (a + b))
            else:
                summaries[arm.treatment] = (math.log(a / b), 1.0 / a + 1.0 / b)
    elif outcome == "continuous":
        for arm in study.arms:
            summaries[arm.treatment] = (arm.mean, arm.sd ** 2 / arm.n)  # type: ignore[operator]
    elif outcome == "rate":
        zero = any(arm.events == 0 for arm in study.arms)
        inc = zero_correction if zero else 0.0
        corrected = zero and inc > 0
        for arm in study.arms:
            e = arm.events + inc  # type: ignore[operator]
            summaries[arm.treatment] = (math.log(e / arm.exposure), 1.0 / e)  # type: ignore[operator]
    else:  # pragma: no cover - guarded by validation
        raise ValueError(f"Arm-level data not supported for outcome '{outcome}'")
    return summaries, corrected


def _uninformative_binary(study: Study) -> bool:
    """All arms 0 events, or all arms all events: no information on the OR."""

    return all(arm.events == 0 for arm in study.arms) or all(arm.events == arm.n for arm in study.arms)


def _smd_block(study: Study, baseline: str) -> _StudyBlock:
    """Hedges' g of every arm vs the baseline with the SD pooled over all
    arms of the study (df = N - k) and small-sample factor J = 1 - 3/(4 df - 1).

    var(g_kb) = 1/n_k + 1/n_b + g_kb^2 / (2 N) (two arms: the usual
    (n1 + n2)/(n1 n2) + g^2 / (2 (n1 + n2))). Multi-arm covariance is taken as
    1/n_b, dropping the small g_k g_j / (2 N) term (Gleser & Olkin 2009)."""

    arms = {arm.treatment: arm for arm in study.arms}
    total = sum(arm.n for arm in study.arms)  # type: ignore[misc]
    df = total - len(study.arms)
    pooled = math.sqrt(sum((arm.n - 1) * arm.sd ** 2 for arm in study.arms) / df)  # type: ignore[operator]
    j = 1.0 - 3.0 / (4.0 * df - 1.0) if df > 1 else 1.0
    base = arms[baseline]
    others = [arm.treatment for arm in study.arms if arm.treatment != baseline]
    estimates, variances = [], []
    for name in others:
        g = j * (arms[name].mean - base.mean) / pooled  # type: ignore[operator]
        estimates.append(g)
        variances.append(1.0 / arms[name].n + 1.0 / base.n + g * g / (2.0 * total))  # type: ignore[operator]
    return _StudyBlock(study.id, baseline, others, estimates, variances, 1.0 / base.n)  # type: ignore[operator]


def build_study_blocks(model: NetworkMetaAnalysis, result: NMAResult) -> list[_StudyBlock]:
    """Study blocks with each study's evidence weight applied (variances / weight)."""

    blocks = _build_raw_blocks(model, result)
    weights = {study.id: float(getattr(study, "weight", 1.0)) for study in model.studies}
    for block in blocks:
        w = weights.get(block.study_id, 1.0)
        if w != 1.0 and w > 0:
            block.variances = [v / w for v in block.variances]
            block.baseline_variance = block.baseline_variance / w
    return blocks


def _build_raw_blocks(model: NetworkMetaAnalysis, result: NMAResult) -> list[_StudyBlock]:
    order = {name: index for index, name in enumerate(model.treatment_names)}
    reference = model.reference
    outcome = model.spec.outcome
    blocks: list[_StudyBlock] = []
    for study in model.studies:
        if study.is_contrast_level:
            baseline = study.contrasts[0].versus
            variances = [c.se ** 2 for c in study.contrasts]
            baseline_variance = (study.baseline_se ** 2) if len(study.contrasts) > 1 else 0.0  # type: ignore[operator]
            blocks.append(_StudyBlock(
                study.id, baseline,
                [c.treatment for c in study.contrasts],
                [c.estimate for c in study.contrasts],
                variances, baseline_variance,
            ))
            continue

        measure = model.spec.measure
        if outcome == "binary" and measure != "RD" and _uninformative_binary(study):
            result.excluded_studies.append(study.id)
            result.warnings.append(
                f"{study.id}: excluded - every arm has 0 events (or every patient had the event), so it carries no information on the {measure}"
            )
            continue
        if outcome == "continuous" and measure == "SMD":
            blocks.append(_smd_block(study, _choose_baseline([arm.treatment for arm in study.arms], order, reference)))
            continue
        summaries, corrected = _arm_summaries(study, outcome, model.spec.zero_correction, measure)
        if any(variance <= 0 for _, variance in summaries.values()):
            result.excluded_studies.append(study.id)
            result.warnings.append(
                f"{study.id}: excluded - an arm has zero variance (e.g. 0% or 100% risk with no zero-cell correction)")
            continue
        if corrected:
            result.warnings.append(
                f"{study.id}: zero cell - {model.spec.zero_correction:g} added to every cell of this study"
            )
        names = [arm.treatment for arm in study.arms]
        baseline = _choose_baseline(names, order, reference)
        base_value, base_var = summaries[baseline]
        others = [name for name in names if name != baseline]
        blocks.append(_StudyBlock(
            study.id, baseline, others,
            [summaries[name][0] - base_value for name in others],
            [summaries[name][1] + base_var for name in others],
            base_var,
        ))
    return blocks


# --------------------------------------------------------------------------
# Matrices and fitting
# --------------------------------------------------------------------------


def _assemble(blocks: list[_StudyBlock], parameters: list[str]):
    index = {name: i for i, name in enumerate(parameters)}
    n_obs = sum(len(block.treatments) for block in blocks)
    y = np.zeros(n_obs)
    X = np.zeros((n_obs, len(parameters)))
    V = np.zeros((n_obs, n_obs))
    C = np.zeros((n_obs, n_obs))
    row = 0
    for block in blocks:
        k = len(block.treatments)
        rows = slice(row, row + k)
        y[rows] = block.estimates
        for offset, treatment in enumerate(block.treatments):
            if treatment in index:
                X[row + offset, index[treatment]] += 1.0
            if block.baseline in index:
                X[row + offset, index[block.baseline]] -= 1.0
        V[rows, rows] = block.baseline_variance
        C[rows, rows] = 0.5
        for offset in range(k):
            V[row + offset, row + offset] = block.variances[offset]
            C[row + offset, row + offset] = 1.0
        row += k
    return y, X, V, C


def _gls(y: np.ndarray, X: np.ndarray, V: np.ndarray):
    W = np.linalg.inv(V)
    XtW = X.T @ W
    information = XtW @ X
    covariance = np.linalg.inv(information)
    d = covariance @ (XtW @ y)
    return d, covariance, W


def _relative(treatment: str, versus: str, estimate: float, se: float, z: float, ratio: bool) -> RelativeEffect:
    lower, upper = estimate - z * se, estimate + z * se
    transform = math.exp if ratio else (lambda value: value)
    return RelativeEffect(
        treatment=treatment, versus=versus, estimate=estimate, se=se,
        lower=lower, upper=upper,
        natural=transform(estimate), natural_lower=transform(lower), natural_upper=transform(upper),
    )


def _fit_summary(model_name: str, tau2: float, treatments: list[str], reference: str,
                 parameters: list[str], d: np.ndarray, covariance: np.ndarray,
                 ratio: bool, z: float, small_values: str) -> NMAFit:
    # Full effect vector and covariance including the reference (fixed at 0).
    full_index = {name: i for i, name in enumerate(treatments)}
    n = len(treatments)
    effects = np.zeros(n)
    full_cov = np.zeros((n, n))
    param_positions = [full_index[name] for name in parameters]
    effects[param_positions] = d
    full_cov[np.ix_(param_positions, param_positions)] = covariance

    fit = NMAFit(
        model=model_name,
        tau2=tau2,
        parameter_names=list(parameters),
        covariance=covariance.tolist(),
    )
    for name in parameters:
        i = full_index[name]
        fit.effects.append(_relative(name, reference, float(effects[i]), math.sqrt(full_cov[i, i]), z, ratio))

    for a in treatments:
        for b in treatments:
            if a == b:
                continue
            i, j = full_index[a], full_index[b]
            variance = full_cov[i, i] + full_cov[j, j] - 2 * full_cov[i, j]
            fit.league[(a, b)] = _relative(a, b, float(effects[i] - effects[j]), math.sqrt(max(variance, 0.0)), z, ratio)

    # P-scores: mean over competitors of P(a is better than b).
    for a in treatments:
        if n < 2:
            fit.p_scores[a] = 1.0
            continue
        total = 0.0
        for b in treatments:
            if a == b:
                continue
            effect = fit.league[(a, b)]
            if effect.se == 0:
                better = 0.5
            elif small_values == "good":
                better = normal_cdf(-effect.estimate / effect.se)
            else:
                better = normal_cdf(effect.estimate / effect.se)
            total += better
        fit.p_scores[a] = total / (n - 1)
    return fit


def run_nma_frequentist(model: NetworkMetaAnalysis) -> NMAResult:
    """Validate and fit fixed- and random-effects NMA."""

    ratio = is_ratio_measure(model.spec.measure)
    result = NMAResult(reference=model.reference, measure=model.spec.measure)
    result.errors = validate_nma(model)
    if result.errors:
        return result

    blocks = build_study_blocks(model, result)
    used = {block.baseline for block in blocks} | {name for block in blocks for name in block.treatments}
    treatments = [name for name in model.treatment_names if name in used]
    reference = model.reference
    result.treatments = treatments
    if reference not in used:
        result.errors.append(ValidationError(
            code="reference_excluded",
            message=f"The reference '{reference}' has no usable data after exclusions",
        ))
        return result
    parameters = [name for name in treatments if name != reference]

    for block in blocks:
        for treatment, estimate, variance in zip(block.treatments, block.estimates, block.variances):
            result.observations.append(ContrastObservation(block.study_id, treatment, block.baseline, estimate, variance))

    y, X, V, C = _assemble(blocks, parameters)
    if np.linalg.matrix_rank(X) < len(parameters):
        result.errors.append(ValidationError(
            code="not_estimable",
            message="Some treatment effects cannot be estimated (the network is disconnected after exclusions)",
        ))
        return result

    z = _normal_quantile(0.5 + model.spec.ci_level / 2)

    d_fe, cov_fe, W = _gls(y, X, V)
    residual = y - X @ d_fe
    q = float(residual @ W @ residual)
    df = len(y) - len(parameters)
    P = W - W @ X @ cov_fe @ X.T @ W
    trace = float(np.trace(P @ C))
    tau2 = max(0.0, (q - df) / trace) if df > 0 and trace > 0 else 0.0
    if df == 0:
        result.warnings.append("No degrees of freedom for heterogeneity (as many contrasts as treatment effects): tau^2 set to 0; random-effects = fixed-effect")

    d_re, cov_re, _ = _gls(y, X, V + tau2 * C)

    result.heterogeneity = Heterogeneity(
        q=q,
        df=df,
        p_value=chi2_sf(q, df) if df > 0 else None,
        i2=max(0.0, (q - df) / q) if q > 0 and df > 0 else (0.0 if df > 0 else None),
        tau2=tau2,
        tau=math.sqrt(tau2),
    )
    result.fixed = _fit_summary("fixed", 0.0, treatments, reference, parameters, d_fe, cov_fe, ratio, z, model.spec.small_values)
    result.random = _fit_summary("random", tau2, treatments, reference, parameters, d_re, cov_re, ratio, z, model.spec.small_values)
    # t degrees of freedom for the prediction interval: the heterogeneity df
    # (contrasts - treatment effects), as in netmeta.
    prediction_df = df
    if prediction_df >= 1:
        result.heterogeneity.prediction_df = prediction_df
        add_prediction_intervals(result.random, tau2, prediction_df, model.spec.ci_level, ratio)
    return result


def add_prediction_intervals(fit: NMAFit, tau2: float, df: int, level: float, ratio: bool) -> None:
    """estimate +/- t_{df} * sqrt(se^2 + tau^2) for every effect and league cell.

    df = the heterogeneity degrees of freedom (number of study contrasts
    minus number of treatment effects), as netmeta does (k - 1 for a
    pairwise meta-analysis of k studies; Riley, Higgins & Deeks 2011 use
    k - 2, one fewer).
    """

    t = t_quantile(0.5 + level / 2, df)
    transform = math.exp if ratio else (lambda value: value)
    for effect in [*fit.effects, *fit.league.values()]:
        half = t * math.sqrt(effect.se ** 2 + tau2)
        effect.pred_lower = effect.estimate - half
        effect.pred_upper = effect.estimate + half
        effect.natural_pred_lower = transform(effect.pred_lower)
        effect.natural_pred_upper = transform(effect.pred_upper)

