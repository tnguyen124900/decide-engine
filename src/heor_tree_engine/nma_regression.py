"""Network meta-regression (frequentist, contrast-based).

Adds a study-level covariate x (centred at its mean over the analysed
studies) to the NMA model:

    y_i,kb = (d_k - d_b) + (beta_k - beta_b) * (x_i - x_bar) + delta_i,kb + e_i,kb,   beta_ref = 0

- "shared" interaction (NICE DSU TSD 3's exchangeable-to-one common
  coefficient, the usual default): beta_k = beta for every treatment other
  than the reference, so beta is how the effect of any active treatment vs
  the reference changes per unit of x (contrasts between two active
  treatments are unaffected);
- "specific": one beta_k per non-reference treatment (needs covariate
  variation among each treatment's studies).

The covariate is a study characteristic (year, mean age, % female,
follow-up...) or the baseline risk, computed from each study's reference
arm (log-odds / log-rate / mean, as in the analysis scale). Studies without
a covariate value are left out (reported). Fitting mirrors nma_freq: GLS
with the multi-arm covariance, tau^2 by the method of moments on the
extended design.

Caveats shown in the UI: study-level covariates describe trial averages
(ecological bias), and baseline risk regression is affected by regression
to the mean because baseline risk is measured with error.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .models import ValidationError
from .nma_freq import _StudyBlock, _assemble, _gls, build_study_blocks, chi2_sf, normal_cdf
from .nma_models import NetworkMetaAnalysis, NMAResult, is_ratio_measure
from .nma_parser import _normal_quantile
from .nma_validation import validate_nma

BASELINE_RISK = "__baseline_risk__"
INTERACTIONS = ("shared", "specific")


@dataclass
class Coefficient:
    treatment: str  # "all" for the shared coefficient
    estimate: float
    se: float
    lower: float
    upper: float
    p_value: float


@dataclass
class AdjustedEffect:
    treatment: str
    versus: str
    at: float  # covariate value the effect is evaluated at
    estimate: float
    se: float
    lower: float
    upper: float
    natural: float
    natural_lower: float
    natural_upper: float


@dataclass
class RegressionResult:
    covariate: str
    interaction: str
    model: str  # "fixed" | "random"
    measure: str = "OR"
    reference: str | None = None
    treatments: list[str] = field(default_factory=list)
    centre: float = 0.0
    covariate_range: tuple[float, float] = (0.0, 0.0)
    study_values: dict[str, float] = field(default_factory=dict)
    excluded_studies: list[str] = field(default_factory=list)
    coefficients: list[Coefficient] = field(default_factory=list)
    tau2: float = 0.0
    tau2_without: float = 0.0
    q_residual: float = 0.0
    df_residual: int = 0
    p_residual: float | None = None
    # parameter vector [d..., beta...] and covariance (analysis scale)
    parameter_names: list[str] = field(default_factory=list)
    estimates: list[float] = field(default_factory=list)
    covariance: list[list[float]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[ValidationError] = field(default_factory=list)

    @property
    def tau2_explained(self) -> float | None:
        if self.model != "random" or self.tau2_without <= 0:
            return None
        return max(0.0, (self.tau2_without - self.tau2) / self.tau2_without)


def baseline_risk_values(model: NetworkMetaAnalysis) -> dict[str, float]:
    """Reference-arm value per study on the analysis scale (log-odds for
    OR, log-risk for RR, risk for RD, log-rate for counts, mean for
    continuous; not defined for SMD). Studies without a
    reference arm (or contrast-level studies) have none."""

    reference = model.reference
    outcome = model.spec.outcome
    values: dict[str, float] = {}
    if model.spec.measure == "SMD":
        return values
    for study in model.studies:
        arm = next((arm for arm in study.arms if arm.treatment == reference), None)
        if arm is None:
            continue
        try:
            if outcome == "binary":
                zero = any(a.events == 0 or a.events == a.n for a in study.arms)
                inc = model.spec.zero_correction if zero else 0.0
                a, b = arm.events + inc, arm.n - arm.events + inc
                measure = model.spec.measure
                values[study.id] = (math.log(a / (a + b)) if measure == "RR"
                                    else a / (a + b) if measure == "RD" else math.log(a / b))
            elif outcome == "rate":
                zero = any(a.events == 0 for a in study.arms)
                inc = model.spec.zero_correction if zero else 0.0
                values[study.id] = math.log((arm.events + inc) / arm.exposure)
            elif outcome == "continuous":
                values[study.id] = float(arm.mean)
        except (TypeError, ValueError, ZeroDivisionError):
            continue
    return values


def _covariate_column(blocks: list[_StudyBlock], parameters: list[str], reference: str,
                      centred: dict[str, float], interaction: str) -> tuple[np.ndarray, list[str]]:
    names = ["all"] if interaction == "shared" else list(parameters)
    index = {name: i for i, name in enumerate(names)}
    n_obs = sum(len(block.treatments) for block in blocks)
    Z = np.zeros((n_obs, len(names)))
    row = 0
    for block in blocks:
        x = centred[block.study_id]
        for offset, treatment in enumerate(block.treatments):
            for name, sign in ((treatment, 1.0), (block.baseline, -1.0)):
                if name == reference:
                    continue
                column = index["all"] if interaction == "shared" else index[name]
                Z[row + offset, column] += sign * x
        row += len(block.treatments)
    return Z, names


def _moment_tau2(y, X, V, C) -> tuple[float, float, int]:
    d, cov, W = _gls(y, X, V)
    r = y - X @ d
    q = float(r @ W @ r)
    df = len(y) - X.shape[1]
    P = W - W @ X @ cov @ X.T @ W
    trace = float(np.trace(P @ C))
    tau2 = max(0.0, (q - df) / trace) if df > 0 and trace > 0 else 0.0
    return tau2, q, df


def run_meta_regression(model: NetworkMetaAnalysis, covariate: str, study_values: dict[str, float] | None = None,
                        interaction: str = "shared", effects: str = "random") -> RegressionResult:
    """`covariate` is a label; values come from `study_values` (study id ->
    number) or, for covariate == BASELINE_RISK, are computed from the
    reference arms."""

    result = RegressionResult(covariate=covariate, interaction=interaction, model=effects,
                              measure=model.spec.measure, reference=model.reference)
    result.errors = validate_nma(model)
    if interaction not in INTERACTIONS:
        result.errors.append(ValidationError(code="unknown_interaction", message=f"Unknown interaction '{interaction}'"))
    if result.errors:
        return result

    values = baseline_risk_values(model) if covariate == BASELINE_RISK else dict(study_values or {})
    scratch = NMAResult()
    blocks = build_study_blocks(model, scratch)
    result.warnings.extend(scratch.warnings)
    kept = [block for block in blocks if isinstance(values.get(block.study_id), (int, float))
            and not math.isnan(float(values[block.study_id]))]
    result.excluded_studies = [block.study_id for block in blocks if block not in kept]
    if len(kept) < 3:
        result.errors.append(ValidationError(code="too_few_studies",
                                             message="Meta-regression needs a covariate value for at least 3 studies"))
        return result

    used = {name for block in kept for name in (block.baseline, *block.treatments)}
    treatments = [name for name in model.treatment_names if name in used]
    reference = model.reference
    if reference not in used:
        result.errors.append(ValidationError(code="reference_excluded",
                                             message="No study with a covariate value includes the reference treatment"))
        return result
    parameters = [name for name in treatments if name != reference]
    result.treatments = treatments

    raw = {block.study_id: float(values[block.study_id]) for block in kept}
    centre = float(np.mean(list(raw.values())))
    centred = {study_id: value - centre for study_id, value in raw.items()}
    result.centre = centre
    result.covariate_range = (min(raw.values()), max(raw.values()))
    result.study_values = raw

    y, X, V, C = _assemble(kept, parameters)
    Z, beta_names = _covariate_column(kept, parameters, reference, centred, interaction)
    XZ = np.hstack([X, Z])
    if np.linalg.matrix_rank(X) < X.shape[1]:
        result.errors.append(ValidationError(code="not_estimable",
                                             message="The network is disconnected once studies without the covariate are dropped"))
        return result
    if np.linalg.matrix_rank(XZ) < XZ.shape[1]:
        result.errors.append(ValidationError(
            code="beta_not_estimable",
            message=("The covariate coefficient(s) cannot be estimated - the covariate does not vary enough "
                     + ("across studies." if interaction == "shared"
                        else "within each treatment's studies; try the shared interaction."))))
        return result

    tau2_without, _, _ = _moment_tau2(y, X, V, C)
    tau2, q, df = _moment_tau2(y, XZ, V, C)
    if effects != "random":
        tau2 = 0.0
    result.tau2 = tau2
    result.tau2_without = tau2_without if effects == "random" else 0.0
    result.q_residual = q
    result.df_residual = df
    result.p_residual = chi2_sf(q, df) if df > 0 else None
    if df <= 0:
        result.warnings.append("No residual degrees of freedom: the regression fits the data exactly.")

    theta, cov, _ = _gls(y, XZ, V + tau2 * C)
    z = _normal_quantile(0.5 + model.spec.ci_level / 2)
    for offset, name in enumerate(beta_names):
        i = len(parameters) + offset
        estimate, se = float(theta[i]), math.sqrt(max(float(cov[i, i]), 0.0))
        p_value = 2.0 * (1.0 - normal_cdf(abs(estimate / se))) if se > 0 else float("nan")
        result.coefficients.append(Coefficient(treatment=name, estimate=estimate, se=se,
                                               lower=estimate - z * se, upper=estimate + z * se, p_value=p_value))
    result.parameter_names = [f"d[{name}]" for name in parameters] + [f"beta[{name}]" for name in beta_names]
    result.estimates = theta.tolist()
    result.covariance = cov.tolist()
    return result


def effects_at(result: RegressionResult, value: float, ci_level: float = 0.95) -> list[AdjustedEffect]:
    """Effect of every treatment vs the reference at covariate = value."""

    if result.errors or not result.estimates:
        return []
    theta = np.array(result.estimates)
    cov = np.array(result.covariance)
    parameters = [name for name in result.treatments if name != result.reference]
    n_d = len(parameters)
    shift = value - result.centre
    z = _normal_quantile(0.5 + ci_level / 2)
    transform = math.exp if is_ratio_measure(result.measure) else (lambda v: v)
    out = []
    for i, name in enumerate(parameters):
        vector = np.zeros(len(theta))
        vector[i] = 1.0
        beta_index = n_d if result.interaction == "shared" else n_d + i
        vector[beta_index] = shift
        estimate = float(vector @ theta)
        se = math.sqrt(max(float(vector @ cov @ vector), 0.0))
        lower, upper = estimate - z * se, estimate + z * se
        out.append(AdjustedEffect(treatment=name, versus=result.reference, at=value, estimate=estimate, se=se,
                                  lower=lower, upper=upper, natural=transform(estimate),
                                  natural_lower=transform(lower), natural_upper=transform(upper)))
    return out


def effect_curve(result: RegressionResult, treatment: str, points: int = 41, ci_level: float = 0.95) -> list[AdjustedEffect]:
    """Effect of `treatment` vs the reference across the observed covariate range."""

    low, high = result.covariate_range
    if high <= low:
        return []
    curve = []
    for value in np.linspace(low, high, points):
        curve.extend(effect for effect in effects_at(result, float(value), ci_level) if effect.treatment == treatment)
    return curve
