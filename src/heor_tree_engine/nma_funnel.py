"""Comparison-adjusted funnel plot for NMA (Chaimani & Salanti 2012;
Chaimani et al. 2013, PLoS One 8:e76654).

Each study contributes one point per pair of its arms: the observed
contrast y_iXY minus the comparison-specific summary mu_XY (fixed-effect
pairwise meta-analysis of the direct studies of X vs Y), plotted against
its standard error. Every contrast is oriented the same way - "newer vs
older" by a user-given treatment order (earlier = older / control) - so if
small studies systematically favour newer treatments, the points lean to
one side. Without small-study effects the funnel is symmetric around 0.

Asymmetry test: Egger's regression on the centred points,
    (y - mu) / se = a + b * (1 / se) + error,
with H0: a = 0 (t-test, n - 2 df). Low power with fewer than ~10 points.
Multi-arm studies contribute every pair of arms (correlated points), as in
netmeta's funnel().
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .models import ValidationError
from .nma_freq import _StudyBlock, build_study_blocks, t_cdf
from .nma_models import NetworkMetaAnalysis, NMAResult
from .nma_validation import validate_nma


@dataclass
class FunnelPoint:
    study_id: str
    newer: str
    older: str
    estimate: float  # observed contrast, newer vs older (analysis scale)
    se: float
    centred: float  # estimate - comparison summary


@dataclass
class FunnelResult:
    measure: str = "OR"
    order: list[str] = field(default_factory=list)
    points: list[FunnelPoint] = field(default_factory=list)
    comparison_means: dict[tuple[str, str], float] = field(default_factory=dict)
    egger_intercept: float | None = None
    egger_se: float | None = None
    egger_p: float | None = None
    warnings: list[str] = field(default_factory=list)
    errors: list[ValidationError] = field(default_factory=list)


def _pair_contrasts(block: _StudyBlock):
    """Every pair of arms of a study: (a, b, y_b - y_a, var)."""

    values = {block.baseline: 0.0, **dict(zip(block.treatments, block.estimates))}
    arm_var = {block.baseline: block.baseline_variance}
    for name, variance in zip(block.treatments, block.variances):
        arm_var[name] = variance - block.baseline_variance
    names = [block.baseline, *block.treatments]
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            yield a, b, values[b] - values[a], arm_var[a] + arm_var[b]


def comparison_adjusted_funnel(model: NetworkMetaAnalysis, order: list[str] | None = None) -> FunnelResult:
    result = FunnelResult(measure=model.spec.measure)
    result.errors = validate_nma(model)
    if result.errors:
        return result
    order = [name for name in (order or model.treatment_names) if name in model.treatment_names]
    order += [name for name in model.treatment_names if name not in order]
    result.order = order
    rank = {name: i for i, name in enumerate(order)}

    scratch = NMAResult()
    blocks = build_study_blocks(model, scratch)
    result.warnings.extend(scratch.warnings)

    raw: list[tuple[str, str, str, float, float]] = []  # study, newer, older, y, var
    for block in blocks:
        for a, b, y, variance in _pair_contrasts(block):
            if variance <= 0:
                continue
            if rank[a] > rank[b]:  # a is newer: flip to newer vs older
                raw.append((block.study_id, a, b, -y, variance))
            else:
                raw.append((block.study_id, b, a, y, variance))

    by_comparison: dict[tuple[str, str], list[tuple[float, float]]] = {}
    for _, newer, older, y, variance in raw:
        by_comparison.setdefault((newer, older), []).append((y, variance))
    for key, rows in by_comparison.items():
        weights = [1.0 / v for _, v in rows]
        result.comparison_means[key] = sum(w * y for w, (y, _) in zip(weights, rows)) / sum(weights)

    for study_id, newer, older, y, variance in raw:
        mean = result.comparison_means[(newer, older)]
        result.points.append(FunnelPoint(study_id=study_id, newer=newer, older=older, estimate=y,
                                         se=math.sqrt(variance), centred=y - mean))

    n = len(result.points)
    if n >= 3:
        se = np.array([point.se for point in result.points])
        z = np.array([point.centred for point in result.points]) / se
        precision = 1.0 / se
        design = np.column_stack([np.ones(n), precision])
        coef, *_ = np.linalg.lstsq(design, z, rcond=None)
        residual = z - design @ coef
        sigma2 = float(residual @ residual) / (n - 2) if n > 2 else float("nan")
        cov = sigma2 * np.linalg.inv(design.T @ design)
        intercept, intercept_se = float(coef[0]), math.sqrt(max(float(cov[0, 0]), 0.0))
        result.egger_intercept = intercept
        result.egger_se = intercept_se
        if intercept_se > 0 and n > 2:
            t = intercept / intercept_se
            result.egger_p = 2.0 * (1.0 - t_cdf(abs(t), n - 2))
    if n < 10:
        result.warnings.append(f"Only {n} points: the funnel plot and Egger's test have little power below about 10.")
    return result
