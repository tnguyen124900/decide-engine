"""Non-randomised evidence in an NMA (B4): sensitivity of the network estimates to how non-randomised studies
(NRS) are handled. Scenarios: RCT only (NRS excluded), all studies at full weight, and NRS down-weighted by a
factor alpha in (0, 1] (a power / discount weight: the NRS variances are divided by alpha, so alpha = 0.25 gives a
NRS the information of a quarter of its size). Frequentist model (GLS). This is a sensitivity analysis; the main
analysis is unchanged. Reference: Schmitz et al. 2013 (Value Health), Verde & Ohmann 2015 for the general idea of
discounting non-randomised evidence; NICE DSU TSD 3 discusses bias adjustment."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

from .models import ValidationError
from .nma_freq import run_nma_frequentist
from .nma_models import NetworkMetaAnalysis, is_ratio_measure
from .nma_sensitivity import exclude_studies


@dataclass
class DesignScenario:
    name: str
    alpha: float | None
    n_studies: int
    tau2: float = 0.0
    effects: dict[str, tuple[float, float, float]] = field(default_factory=dict)  # natural scale: est, lower, upper
    error: str = ""


@dataclass
class DesignResult:
    scenarios: list[DesignScenario] = field(default_factory=list)
    nrs: list[str] = field(default_factory=list)
    reference: str | None = None
    measure: str = "OR"
    max_shift: dict[str, float] = field(default_factory=dict)  # treatment -> |all - RCT only| on the analysis scale
    warnings: list[str] = field(default_factory=list)
    errors: list[ValidationError] = field(default_factory=list)


def _row(name: str, alpha, model: NetworkMetaAnalysis, result, effects: str) -> DesignScenario:
    row = DesignScenario(name=name, alpha=alpha, n_studies=len(model.studies))
    if result.errors:
        row.error = "; ".join(e.message for e in result.errors[:2])
        return row
    fit = result.random if effects == "random" else result.fixed
    row.tau2 = fit.tau2
    for e in fit.effects:
        row.effects[e.treatment] = (e.natural, e.natural_lower, e.natural_upper)
    return row


def run_design_sensitivity(model: NetworkMetaAnalysis, nrs_ids, alphas=(0.5, 0.25, 0.1),
                           effects: str = "random") -> DesignResult:
    ids = [s.id for s in model.studies if s.id in set(nrs_ids)]
    out = DesignResult(nrs=ids, reference=model.reference, measure=model.spec.measure)
    if not ids:
        out.errors.append(ValidationError(code="no_nrs", message="No non-randomised studies are marked"))
        return out
    if len(ids) == len(model.studies):
        out.errors.append(ValidationError(code="all_nrs", message="Every study is marked non-randomised"))
        return out
    out.scenarios.append(_row("RCT only", None, model, exclude_studies(model, ids), effects))
    out.scenarios.append(_row("All studies, full weight", 1.0, model, run_nma_frequentist(model), effects))
    for alpha in alphas:
        weighted = replace(model, studies=[replace(s, weight=float(alpha)) if s.id in ids else s for s in model.studies])
        out.scenarios.append(_row(f"NRS weight {alpha:g}", float(alpha), model, run_nma_frequentist(weighted), effects))
    rct, full = out.scenarios[0], out.scenarios[1]
    for name, (est, _, _) in full.effects.items():
        if name in rct.effects:
            a, b = rct.effects[name][0], est
            out.max_shift[name] = abs(math.log(b) - math.log(a)) if is_ratio_measure(model.spec.measure) else abs(b - a)
    if rct.error:
        out.warnings.append("RCT-only fit failed: " + rct.error)
    return out
