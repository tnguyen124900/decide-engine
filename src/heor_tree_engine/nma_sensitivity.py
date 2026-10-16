"""Sensitivity analyses for a frequentist NMA.

Leave-one-study-out: refit the network once per study with that study
removed, to show how much any single trial drives the results (and whether
removing it would change a conclusion, i.e. move a CI across the null).
A study whose removal disconnects the network, or leaves a treatment with
no data, is reported as not estimable for the affected treatments.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

from .nma_freq import run_nma_frequentist
from .nma_models import NetworkMetaAnalysis, NMAResult, is_ratio_measure


@dataclass
class LeaveOneOutRow:
    study_id: str
    tau2: float | None
    # treatment -> (natural estimate, natural lower, natural upper); missing
    # when not estimable without this study.
    effects: dict[str, tuple[float, float, float]] = field(default_factory=dict)
    # treatments whose "CI excludes the null" status differs from the full model
    changed: list[str] = field(default_factory=list)
    error: str | None = None


def _significant(lower: float, upper: float, null: float) -> bool:
    return lower > null or upper < null


def leave_one_out(model: NetworkMetaAnalysis, full: NMAResult | None = None,
                  effects_model: str = "random") -> list[LeaveOneOutRow]:
    full = full or run_nma_frequentist(model)
    if full.errors:
        return []
    fit_full = full.random if effects_model == "random" else full.fixed
    null = 1.0 if is_ratio_measure(model.spec.measure) else 0.0
    baseline = {
        effect.treatment: _significant(effect.natural_lower, effect.natural_upper, null)
        for effect in fit_full.effects
    }

    rows: list[LeaveOneOutRow] = []
    for study in model.studies:
        remaining = [other for other in model.studies if other is not study]
        used = {name for other in remaining for name in other.treatments}
        treatments = [treatment for treatment in model.treatments if treatment.name in used]
        reduced = replace(model, studies=remaining, treatments=treatments)
        row = LeaveOneOutRow(study_id=study.id, tau2=None)
        if model.reference not in used:
            row.error = f"removes every study of the reference '{model.reference}'"
            rows.append(row)
            continue
        result = run_nma_frequentist(reduced)
        if result.errors:
            row.error = "; ".join(error.message for error in result.errors[:2])
            rows.append(row)
            continue
        fit = result.random if effects_model == "random" else result.fixed
        row.tau2 = fit.tau2
        for effect in fit.effects:
            row.effects[effect.treatment] = (effect.natural, effect.natural_lower, effect.natural_upper)
            if effect.treatment in baseline and \
                    _significant(effect.natural_lower, effect.natural_upper, null) != baseline[effect.treatment]:
                row.changed.append(effect.treatment)
        rows.append(row)
    return rows


def influence_summary(rows: list[LeaveOneOutRow], full: NMAResult, treatment: str,
                      effects_model: str = "random") -> dict[str, float] | None:
    """Range of the treatment's estimate across leave-one-out fits, and the
    largest absolute shift on the analysis scale (log for ratios)."""

    fit = full.random if effects_model == "random" else full.fixed
    reference = {effect.treatment: effect for effect in fit.effects}.get(treatment)
    values = [row.effects[treatment][0] for row in rows if treatment in row.effects]
    if reference is None or not values:
        return None
    to_analysis = math.log if is_ratio_measure(full.measure) else (lambda value: value)
    shifts = [abs(to_analysis(value) - reference.estimate) for value in values]
    return {"min": min(values), "max": max(values), "max_shift": max(shifts)}


def exclude_studies(model: NetworkMetaAnalysis, study_ids: set[str] | list[str]) -> NMAResult:
    """Refit without the given studies (e.g. those at high risk of bias).
    Treatments left without data are dropped; if the reference loses all its
    studies or the network disconnects, the result carries errors."""

    excluded = set(study_ids)
    remaining = [study for study in model.studies if study.id not in excluded]
    used = {name for study in remaining for name in study.treatments}
    treatments = [treatment for treatment in model.treatments if treatment.name in used]
    reduced = replace(model, studies=remaining, treatments=treatments)
    if model.reference not in used:
        result = NMAResult(reference=model.reference, measure=model.spec.measure)
        from .models import ValidationError

        result.errors.append(ValidationError(
            code="reference_excluded",
            message=f"Excluding these studies removes every study of the reference '{model.reference}'",
        ))
        return result
    return run_nma_frequentist(reduced)
