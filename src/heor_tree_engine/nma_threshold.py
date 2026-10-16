"""Threshold analysis for network meta-analysis (Phillippo et al. 2018, JRSS A
181:819-845; Phillippo et al. 2019, Med Decis Making 39:816-826).

Question: how much could each piece of evidence change before the treatment
recommendation changes? With the GLS (frequentist) NMA, the network estimates
are linear in the study contrasts,
    d_hat = H y,     H = (X' W X)^-1 X' W,     W = (V + tau^2 C)^-1
(tau^2 held at its estimate; multi-arm studies enter through their
within-study covariance). The decision is the treatment with the largest
"benefit" b_k(d) = a_k + g_k' d, which is linear in the treatment effects d:
  * effect basis: b_k = +d_k (or -d_k when small values are good), b_ref = 0;
  * net-monetary-benefit basis: NMB_k = lambda E_k - C_k at the NMA estimate,
    and a first-order response to d_k of w_k = lambda dE_k/dd_k - dC_k/dd_k.
A shift delta of one contrast y_i, with all other data held fixed, moves the
estimates by h_i delta (h_i = column i of H), so competitor j overtakes the
current best k* exactly when delta = (b_k* - b_j) / ((g_j - g_k*)' h_i): a
closed form for the lower and upper limits of the invariant interval. The
treatment that takes over at each limit is the competitor reached first.

The same is reported for each network estimate d_k (shifting it alone). At a
threshold the two treatments have equal NMB, so the ICER of the new best
against the old best equals lambda there; `icer` lists cost, effect and ICER
against the reference for the decision inputs as context.

Limits: data points are shifted one at a time; the linear response used
for NMB is only a local approximation of the economic model; a Bayesian
analysis uses the same frequentist matrices (posterior ~ normal).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .models import ValidationError
from .nma_freq import _assemble, _gls, build_study_blocks, run_nma_frequentist
from .nma_models import NetworkMetaAnalysis, NMAResult, is_ratio_measure
from .nma_validation import validate_nma


@dataclass
class DecisionInputs:
    """Decision basis. `basis` = "effect" or "nmb". For "nmb": `wtp` is the willingness to pay per unit of
    effect (lambda) and `values[treatment]` holds `cost`, `effect` (absolute, at the NMA estimate) and the
    sensitivities `d_effect` and `d_cost` (change of the treatment's effect / cost per unit of its NMA effect d_k
    on the analysis scale). Missing sensitivities are 0 and missing cost / effect are 0."""

    basis: str = "effect"
    wtp: float = 30000.0
    values: dict[str, dict[str, float]] = field(default_factory=dict)


@dataclass
class ThresholdRow:
    study: str  # study id, or "NMA" for a network estimate
    comparison: str  # "B vs A"
    estimate: float  # observed contrast (or NMA estimate), analysis scale
    se: float
    lower_delta: float | None  # shift that changes the decision (None: never in that direction)
    upper_delta: float | None
    lower_new_best: str | None
    upper_new_best: str | None
    lower_value: float | None = None  # estimate + delta
    upper_value: float | None = None
    lower_se_units: float | None = None  # |delta| / se
    upper_se_units: float | None = None
    fragile: bool = False  # a limit lies within the 95% interval of the estimate itself
    natural: float = float("nan")
    natural_lower: float | None = None
    natural_upper: float | None = None


@dataclass
class ThresholdResult:
    basis: str = "effect"
    measure: str = "OR"
    reference: str | None = None
    effects: str = "random"
    tau2: float = 0.0
    treatments: list[str] = field(default_factory=list)
    benefits: dict[str, float] = field(default_factory=dict)
    order: list[str] = field(default_factory=list)  # treatments, best first
    best: str = ""
    rows: list[ThresholdRow] = field(default_factory=list)  # one per study contrast
    network_rows: list[ThresholdRow] = field(default_factory=list)  # one per NMA estimate
    icer: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[ValidationError] = field(default_factory=list)


def _limits(gap: np.ndarray, slope: np.ndarray, names: list[str], best: int):
    """Lower / upper shift and the treatment reached first. gap[j] = b_best - b_j >= 0, slope[j] = change of
    (b_j - b_best) per unit shift."""

    upper, upper_name, lower, lower_name = None, None, None, None
    for j, name in enumerate(names):
        if j == best or abs(slope[j]) < 1e-12:
            continue
        delta = gap[j] / slope[j]  # b_j - b_best = -gap + slope * delta = 0
        if slope[j] > 0 and (upper is None or delta < upper):
            upper, upper_name = float(delta), name
        elif slope[j] < 0 and (lower is None or delta > lower):
            lower, lower_name = float(delta), name
    return lower, lower_name, upper, upper_name


def run_threshold_analysis(model: NetworkMetaAnalysis, effects: str = "random",
                           decision: DecisionInputs | None = None,
                           result: NMAResult | None = None) -> ThresholdResult:
    decision = decision or DecisionInputs()
    out = ThresholdResult(basis=decision.basis, measure=model.spec.measure, reference=model.reference, effects=effects)
    out.errors = validate_nma(model)
    if decision.basis not in ("effect", "nmb"):
        out.errors.append(ValidationError(code="unknown_basis", message=f"Unknown decision basis '{decision.basis}'"))
    if out.errors:
        return out
    result = result if result is not None else run_nma_frequentist(model)
    if result.errors:
        out.errors = list(result.errors)
        return out
    blocks = build_study_blocks(model, NMAResult())
    reference = model.reference
    used = {block.baseline for block in blocks} | {name for block in blocks for name in block.treatments}
    treatments = [name for name in model.treatment_names if name in used]
    parameters = [name for name in treatments if name != reference]
    y, X, V, C = _assemble(blocks, parameters)
    tau2 = float(result.heterogeneity.tau2) if effects == "random" and result.heterogeneity is not None else 0.0
    d_hat, covariance, W = _gls(y, X, V + tau2 * C)
    H = covariance @ X.T @ W  # (p, n_obs)
    out.treatments, out.tau2 = treatments, tau2
    column = {name: i for i, name in enumerate(parameters)}
    K, p = len(treatments), len(parameters)

    a = np.zeros(K)
    G = np.zeros((K, p))
    if decision.basis == "effect":
        sign = -1.0 if model.spec.small_values == "good" else 1.0
        for k, name in enumerate(treatments):
            if name in column:
                G[k, column[name]] = sign
    else:
        for k, name in enumerate(treatments):
            v = decision.values.get(name, {})
            nmb = decision.wtp * float(v.get("effect", 0.0)) - float(v.get("cost", 0.0))
            slope = decision.wtp * float(v.get("d_effect", 0.0)) - float(v.get("d_cost", 0.0))
            if name in column:
                G[k, column[name]] = slope
                a[k] = nmb - slope * d_hat[column[name]]
            else:
                a[k] = nmb  # reference
    benefit = a + G @ d_hat
    best = int(np.argmax(benefit))
    out.benefits = {name: float(benefit[k]) for k, name in enumerate(treatments)}
    out.order = [treatments[k] for k in np.argsort(-benefit, kind="stable")]
    out.best = treatments[best]
    gap = benefit[best] - benefit
    if np.sum(gap < 1e-12) > 1:
        out.warnings.append("Two treatments have the same benefit: the decision is on a threshold already.")

    ratio = is_ratio_measure(model.spec.measure)
    z = 1.959963984540054

    def build_row(study: str, comparison: str, estimate: float, se: float, h: np.ndarray) -> ThresholdRow:
        slope = (G - G[best][None, :]) @ h
        lower, lower_name, upper, upper_name = _limits(gap, slope, treatments, best)
        row = ThresholdRow(study=study, comparison=comparison, estimate=float(estimate), se=float(se),
                           lower_delta=lower, upper_delta=upper, lower_new_best=lower_name, upper_new_best=upper_name)
        if lower is not None:
            row.lower_value, row.lower_se_units = estimate + lower, abs(lower) / se if se > 0 else None
        if upper is not None:
            row.upper_value, row.upper_se_units = estimate + upper, upper / se if se > 0 else None
        row.fragile = bool((lower is not None and se > 0 and abs(lower) < z * se)
                           or (upper is not None and se > 0 and upper < z * se))
        if ratio:
            row.natural = math.exp(estimate)
            row.natural_lower = None if row.lower_value is None else math.exp(row.lower_value)
            row.natural_upper = None if row.upper_value is None else math.exp(row.upper_value)
        else:
            row.natural = float(estimate)
            row.natural_lower, row.natural_upper = row.lower_value, row.upper_value
        return row

    row_index = 0
    for block in blocks:
        for treatment, estimate, variance in zip(block.treatments, block.estimates, block.variances):
            out.rows.append(build_row(block.study_id, f"{treatment} vs {block.baseline}", estimate,
                                      math.sqrt(variance), H[:, row_index]))
            row_index += 1
    for name in parameters:
        unit = np.zeros(p)
        unit[column[name]] = 1.0
        out.network_rows.append(build_row("NMA", f"{name} vs {reference}", d_hat[column[name]],
                                          math.sqrt(covariance[column[name], column[name]]), unit))

    if decision.basis == "nmb":
        ref_v = decision.values.get(reference, {})
        for name in treatments:
            v = decision.values.get(name, {})
            d_cost = float(v.get("cost", 0.0)) - float(ref_v.get("cost", 0.0))
            d_eff = float(v.get("effect", 0.0)) - float(ref_v.get("effect", 0.0))
            out.icer.append({"treatment": name, "cost": float(v.get("cost", 0.0)), "effect": float(v.get("effect", 0.0)),
                             "incremental_cost": d_cost, "incremental_effect": d_eff,
                             "icer": (d_cost / d_eff) if abs(d_eff) > 1e-12 else None,
                             "nmb": out.benefits[name]})
    return out
