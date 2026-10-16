"""Component network meta-analysis, additive model (Rücker et al. 2020, Biom J 62:808-821, as netmeta::netcomb).

A treatment is a set of components (e.g. "drug + counselling"). The additive model says the effect of a
treatment versus the reference is the sum of the effects of the components in which the two differ:
    d_k - d_ref = (s_k - s_ref)' c,     s_k in {0,1}^m the component indicator vector of treatment k.
The model is the contrast-based GLS of nma_freq with the design matrix X replaced by X B (B = rows s_k - s_ref
of the non-reference treatments). It is estimable only if X B has full column rank m. Additivity is checked
against the standard NMA: Q_diff = Q_additive - Q_standard on p - m degrees of freedom (fixed-effect Q values).
A treatment with no components entered is treated as one component of its own name; the reference with none is
the empty set (a placebo / usual care).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .models import ValidationError
from .nma_freq import (_assemble, _fit_summary, _gls, _normal_quantile, _relative, build_study_blocks, chi2_sf,
                       run_nma_frequentist)
from .nma_models import Heterogeneity, NetworkMetaAnalysis, NMAFit, NMAResult, RelativeEffect, is_ratio_measure
from .nma_validation import validate_nma


def parse_components(text) -> list[str]:
    """"A + B" / "A, B" / ["A", "B"] -> ["A", "B"] (unique, in order)."""

    if text is None:
        return []
    parts = text if isinstance(text, (list, tuple)) else str(text).replace(",", "+").replace(";", "+").split("+")
    seen: list[str] = []
    for part in parts:
        name = str(part).strip()
        if name and name not in seen:
            seen.append(name)
    return seen


@dataclass
class ComponentResult:
    measure: str = "OR"
    reference: str | None = None
    effects: str = "random"
    components: list[str] = field(default_factory=list)
    component_effects: list[RelativeEffect] = field(default_factory=list)  # effect of adding each component
    fit: NMAFit | None = None  # treatment effects implied by the additive model
    tau2: float = 0.0
    q_additive: float = 0.0
    q_standard: float = 0.0
    q_diff: float = 0.0
    df_diff: int = 0
    p_additivity: float | None = None
    membership: dict[str, list[str]] = field(default_factory=dict)
    standard: NMAResult | None = None
    warnings: list[str] = field(default_factory=list)
    errors: list[ValidationError] = field(default_factory=list)


def run_component_nma(model: NetworkMetaAnalysis, components: dict[str, list[str] | str] | None = None,
                      effects: str = "random") -> ComponentResult:
    out = ComponentResult(measure=model.spec.measure, reference=model.reference, effects=effects)
    out.errors = validate_nma(model)
    if out.errors:
        return out
    standard = run_nma_frequentist(model)
    out.standard = standard
    if standard.errors:
        out.errors = list(standard.errors)
        return out
    blocks = build_study_blocks(model, NMAResult())
    reference = model.reference
    used = {block.baseline for block in blocks} | {name for block in blocks for name in block.treatments}
    treatments = [name for name in model.treatment_names if name in used]
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
    if not names:
        out.errors.append(ValidationError(code="no_components", message="No components defined"))
        return out
    m, p = len(names), len(parameters)
    index = {c: i for i, c in enumerate(names)}

    def vector(treatment: str) -> np.ndarray:
        v = np.zeros(m)
        for part in membership[treatment]:
            v[index[part]] = 1.0
        return v

    B = np.array([vector(t) - vector(reference) for t in parameters])  # (p, m)
    y, X, V, C = _assemble(blocks, parameters)
    Xc = X @ B
    if np.linalg.matrix_rank(Xc) < m:
        out.errors.append(ValidationError(
            code="components_not_identified",
            message="The component effects are not identifiable: some components always occur together "
                    "(add treatments with the components separately or in other combinations)."))
        return out
    z = _normal_quantile(0.5 + model.spec.ci_level / 2)
    ratio = is_ratio_measure(model.spec.measure)

    c_fe, cov_fe, W = _gls(y, Xc, V)
    res = y - Xc @ c_fe
    q_add = float(res @ W @ res)
    Xs = X
    d_std, cov_std, Ws = _gls(y, Xs, V)
    rs = y - Xs @ d_std
    q_std = float(rs @ Ws @ rs)
    out.q_additive, out.q_standard = q_add, q_std
    out.q_diff = max(0.0, q_add - q_std)
    out.df_diff = p - m
    out.p_additivity = chi2_sf(out.q_diff, out.df_diff) if out.df_diff > 0 else None
    df = len(y) - m
    tau2 = 0.0
    if effects == "random" and df > 0:
        P = W - W @ Xc @ cov_fe @ Xc.T @ W
        trace = float(np.trace(P @ C))
        tau2 = max(0.0, (q_add - df) / trace) if trace > 0 else 0.0
    out.tau2 = tau2
    c_hat, cov_c, _ = _gls(y, Xc, V + tau2 * C)
    d_hat = B @ c_hat
    cov_d = B @ cov_c @ B.T
    out.components = names
    out.membership = membership
    out.component_effects = [_relative(name, "none", float(c_hat[i]), math.sqrt(cov_c[i, i]), z, ratio)
                             for i, name in enumerate(names)]
    out.fit = _fit_summary("random" if effects == "random" else "fixed", tau2, treatments, reference, parameters,
                           d_hat, cov_d, ratio, z, model.spec.small_values)
    if out.df_diff <= 0:
        out.warnings.append("The additive model has as many parameters as the standard NMA: additivity cannot be tested.")
    elif out.p_additivity is not None and out.p_additivity < 0.05:
        out.warnings.append(f"Additivity is questionable (Q_diff = {out.q_diff:.2f} on {out.df_diff} df, p = {out.p_additivity:.3f}).")
    return out
