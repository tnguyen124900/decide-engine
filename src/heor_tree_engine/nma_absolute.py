"""Absolute effects and number needed to treat from NMA relative effects
(GRADE Summary of Findings; Cochrane Handbook 15.4-15.5).

Given the risk p0 with the reference treatment (over the same time frame
as the outcome), the risk with treatment k is
    OR:               p1 = OR p0 / (1 - p0 + OR p0)
    RR:               p1 = min(1, RR p0)
    HR / rate ratio:  p1 = 1 - (1 - p0) ** HR   (proportional hazards)
    RD:               p1 = p0 + RD
The CI limits of the relative effect map to limits of p1 (the maps are
monotone). NNT = 1 / |p1 - p0|: NNTB when treatment reduces a harmful event
(or increases a good one), NNTH otherwise. When the CI of the risk
difference includes 0 the NNT interval passes through infinity (Altman
1998), reported as "NNTB a to ∞ to NNTH b".
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .nma_models import NetworkMetaAnalysis

ABSOLUTE_MEASURES = ("OR", "RR", "HR", "RATE_RATIO", "RD")


@dataclass
class AbsoluteEffect:
    treatment: str
    versus: str
    baseline_risk: float
    risk: float
    risk_lower: float
    risk_upper: float
    difference: float  # p1 - p0
    difference_lower: float
    difference_upper: float
    nnt: float | None  # 1 / |difference|, None when difference == 0
    nnt_kind: str  # "NNTB" | "NNTH" | ""
    nnt_text: str  # with the CI, Altman style


def risk_with_treatment(p0: float, effect: float, measure: str) -> float:
    """Risk with treatment for a relative effect on the natural scale."""

    if measure == "OR":
        return effect * p0 / (1.0 - p0 + effect * p0)
    if measure == "RR":
        return min(1.0, effect * p0)
    if measure in ("HR", "RATE_RATIO"):
        return 1.0 - (1.0 - p0) ** effect
    if measure == "RD":
        return min(1.0, max(0.0, p0 + effect))
    raise ValueError(f"No absolute effect for measure '{measure}'")


def observed_baseline_risk(model: NetworkMetaAnalysis) -> float | None:
    """Pooled risk in the reference arms of arm-level binary studies
    (total events / total patients); None when there is none."""

    if model.spec.outcome != "binary":
        return None
    events = total = 0.0
    for study in model.studies:
        for arm in study.arms:
            if arm.treatment == model.reference and arm.events is not None and arm.n:
                events += arm.events
                total += arm.n
    return events / total if total > 0 else None


def _benefit(difference: float, small_values: str) -> bool:
    """True when the change in risk is good for patients."""

    return difference < 0 if small_values == "good" else difference > 0


def _nnt(difference: float) -> str:
    return "∞" if difference == 0 else str(math.ceil(1.0 / abs(difference) - 1e-9))


def _nnt_text(point: float, low: float, high: float, small_values: str) -> str:
    """NNT (rounded up) with its CI; Altman's form when the CI includes 0."""

    def kind(difference: float) -> str:
        return "NNTB" if _benefit(difference, small_values) else "NNTH"

    if low < 0 < high:
        first, second = (low, high) if _benefit(low, small_values) else (high, low)
        return f"{kind(point)} {_nnt(point)} ({kind(first)} {_nnt(first)} to ∞ to {kind(second)} {_nnt(second)})"
    near, far = (low, high) if abs(low) > abs(high) else (high, low)
    return f"{kind(point)} {_nnt(point)} ({_nnt(near)} to {_nnt(far)})"


def absolute_effects(fit, measure: str, baseline_risk: float, small_values: str = "good") -> list[AbsoluteEffect]:
    if measure not in ABSOLUTE_MEASURES:
        return []
    if not 0.0 < baseline_risk < 1.0:
        raise ValueError("The baseline risk must be between 0 and 1")
    p0 = baseline_risk
    out = []
    for effect in fit.effects:
        values = [risk_with_treatment(p0, value, measure)
                  for value in (effect.natural, effect.natural_lower, effect.natural_upper)]
        risk, low, high = values[0], min(values[1:]), max(values[1:])
        difference = risk - p0
        nnt = 1.0 / abs(difference) if difference else None
        kind = "" if not difference else ("NNTB" if _benefit(difference, small_values) else "NNTH")
        out.append(AbsoluteEffect(
            treatment=effect.treatment, versus=effect.versus, baseline_risk=p0, risk=risk,
            risk_lower=low, risk_upper=high, difference=difference,
            difference_lower=low - p0, difference_upper=high - p0, nnt=nnt, nnt_kind=kind,
            nnt_text=(_nnt_text(difference, low - p0, high - p0, small_values) if difference else "–"),
        ))
    return out
