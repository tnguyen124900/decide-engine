"""Validation rules for network meta-analysis documents.

Errors block the analysis; each carries a `code`, a readable `message`, the
study id in `node_id` (when it concerns one study) and the offending field.
Soft issues that do not block a run (zero cells, studies excluded for having
no events) are reported by the engine as warnings instead.
"""

from __future__ import annotations

import difflib

from .models import ValidationError
from .nma_models import (
    ARM_LEVEL_MEASURES,
    MEASURES,
    OUTCOME_TYPES,
    SMALL_VALUES,
    Arm,
    NetworkMetaAnalysis,
    Study,
)
from .nma_network import summarize_network


def _suggest(name: str, known: list[str]) -> str:
    match = difflib.get_close_matches(name, known, n=1, cutoff=0.6)
    return f" (did you mean '{match[0]}'?)" if match else ""


def _validate_spec(model: NetworkMetaAnalysis, errors: list[ValidationError]) -> None:
    spec = model.spec
    if spec.outcome not in OUTCOME_TYPES:
        errors.append(ValidationError(
            code="unknown_outcome",
            message=f"Unknown outcome type '{spec.outcome}' (expected one of {', '.join(OUTCOME_TYPES)})",
            field="spec.outcome",
        ))
        return
    if spec.measure not in MEASURES:
        errors.append(ValidationError(
            code="unknown_measure",
            message=f"Unknown effect measure '{spec.measure}' (expected one of {', '.join(MEASURES)})",
            field="spec.measure",
        ))
    elif spec.outcome in ARM_LEVEL_MEASURES and spec.measure not in ARM_LEVEL_MEASURES[spec.outcome]:
        allowed = ", ".join(ARM_LEVEL_MEASURES[spec.outcome])
        errors.append(ValidationError(
            code="unsupported_measure",
            message=f"Outcome '{spec.outcome}' supports measure {allowed} in this version, not '{spec.measure}'",
            field="spec.measure",
        ))
    if spec.small_values not in SMALL_VALUES:
        errors.append(ValidationError(
            code="invalid_small_values",
            message="'small_values' must be 'good' or 'bad'",
            field="spec.small_values",
        ))
    if spec.zero_correction < 0:
        errors.append(ValidationError(
            code="invalid_zero_correction",
            message="Zero-cell correction must be >= 0",
            field="spec.zero_correction",
        ))
    if not 0 < spec.ci_level < 1:
        errors.append(ValidationError(
            code="invalid_ci_level",
            message="CI level must be between 0 and 1 (e.g. 0.95)",
            field="spec.ci_level",
        ))


def _arm_error(study: Study, index: int, code: str, message: str, field: str) -> ValidationError:
    return ValidationError(code=code, message=f"{study.id}, arm {index + 1}: {message}", node_id=study.id, field=field)


def _validate_arm(study: Study, index: int, arm: Arm, outcome: str, errors: list[ValidationError]) -> None:
    def require(value: float | None, name: str) -> bool:
        if value is None:
            errors.append(_arm_error(study, index, "missing_arm_value", f"'{name}' is required for {outcome} data", name))
            return False
        return True

    if outcome == "binary":
        ok_n = require(arm.n, "n")
        ok_r = require(arm.events, "events")
        if ok_n and arm.n <= 0:  # type: ignore[operator]
            errors.append(_arm_error(study, index, "invalid_n", "'n' must be > 0", "n"))
        if ok_r and arm.events < 0:  # type: ignore[operator]
            errors.append(_arm_error(study, index, "invalid_events", "'events' must be >= 0", "events"))
        if ok_n and ok_r and arm.events > arm.n:  # type: ignore[operator]
            errors.append(_arm_error(study, index, "events_exceed_n", "'events' cannot exceed 'n'", "events"))
    elif outcome == "continuous":
        require(arm.mean, "mean")
        if require(arm.sd, "sd") and arm.sd <= 0:  # type: ignore[operator]
            errors.append(_arm_error(study, index, "invalid_sd", "'sd' must be > 0", "sd"))
        if require(arm.n, "n") and arm.n <= 0:  # type: ignore[operator]
            errors.append(_arm_error(study, index, "invalid_n", "'n' must be > 0", "n"))
    elif outcome == "rate":
        if require(arm.events, "events") and arm.events < 0:  # type: ignore[operator]
            errors.append(_arm_error(study, index, "invalid_events", "'events' must be >= 0", "events"))
        if require(arm.exposure, "exposure") and arm.exposure <= 0:  # type: ignore[operator]
            errors.append(_arm_error(study, index, "invalid_exposure", "'exposure' (person-time) must be > 0", "exposure"))


def _validate_study(study: Study, model: NetworkMetaAnalysis, known: list[str], errors: list[ValidationError]) -> None:
    outcome = model.spec.outcome
    if study.arms and study.contrasts:
        errors.append(ValidationError(
            code="mixed_study_data",
            message=f"{study.id}: give either arm-level data or contrasts, not both",
            node_id=study.id,
        ))
        return
    if not study.arms and not study.contrasts:
        errors.append(ValidationError(code="empty_study", message=f"{study.id}: no data", node_id=study.id))
        return

    names = study.treatments
    if len(set(names)) < 2:
        errors.append(ValidationError(
            code="too_few_arms",
            message=f"{study.id}: a study must compare at least 2 different treatments",
            node_id=study.id,
        ))
    duplicates = sorted({name for name in names if names.count(name) > 1})
    for name in duplicates:
        errors.append(ValidationError(
            code="duplicate_arm",
            message=f"{study.id}: treatment '{name}' appears in more than one arm",
            node_id=study.id,
            field="treatment",
        ))
    for name in dict.fromkeys(names):
        if name not in known:
            errors.append(ValidationError(
                code="unknown_treatment",
                message=f"{study.id}: unknown treatment '{name}'{_suggest(name, known)}",
                node_id=study.id,
                field="treatment",
            ))

    if study.arms:
        if outcome == "contrast":
            errors.append(ValidationError(
                code="arm_data_for_contrast_outcome",
                message=f"{study.id}: outcome type 'contrast' expects contrasts, not arm data",
                node_id=study.id,
            ))
            return
        for index, arm in enumerate(study.arms):
            _validate_arm(study, index, arm, outcome, errors)
        return

    # Contrast-level study.
    for index, contrast in enumerate(study.contrasts):
        if contrast.se <= 0:
            errors.append(ValidationError(
                code="invalid_se",
                message=f"{study.id}, contrast {index + 1}: 'se' must be > 0",
                node_id=study.id,
                field="se",
            ))
        if contrast.treatment == contrast.versus:
            errors.append(ValidationError(
                code="self_contrast",
                message=f"{study.id}, contrast {index + 1}: treatment and comparator are the same",
                node_id=study.id,
            ))
    if len(study.contrasts) > 1:
        comparators = {contrast.versus for contrast in study.contrasts}
        if len(comparators) > 1:
            errors.append(ValidationError(
                code="multi_arm_comparator",
                message=f"{study.id}: all contrasts of a multi-arm study must share one comparator arm",
                node_id=study.id,
                field="versus",
            ))
        if study.baseline_se is None:
            errors.append(ValidationError(
                code="missing_baseline_se",
                message=f"{study.id}: a multi-arm contrast study needs 'baseline_se' (SE of the outcome in the comparator arm)",
                node_id=study.id,
                field="baseline_se",
            ))
        elif study.baseline_se <= 0:
            errors.append(ValidationError(
                code="invalid_baseline_se",
                message=f"{study.id}: 'baseline_se' must be > 0",
                node_id=study.id,
                field="baseline_se",
            ))


def validate_nma(model: NetworkMetaAnalysis) -> list[ValidationError]:
    errors: list[ValidationError] = []
    _validate_spec(model, errors)

    if not model.studies:
        errors.append(ValidationError(code="no_studies", message="Add at least one study"))
        return errors

    known = model.treatment_names
    seen_names: set[str] = set()
    for name in known:
        if name in seen_names:
            errors.append(ValidationError(
                code="duplicate_treatment",
                message=f"Treatment '{name}' is defined twice",
                field="treatments",
            ))
        seen_names.add(name)

    reference = model.reference
    if reference is None or reference not in known:
        errors.append(ValidationError(
            code="unknown_reference",
            message=f"Reference treatment '{reference}' is not in the treatment list{_suggest(str(reference), known)}",
            field="spec.reference",
        ))

    seen_ids: set[str] = set()
    for study in model.studies:
        if study.id in seen_ids:
            errors.append(ValidationError(
                code="duplicate_study",
                message=f"Study id '{study.id}' is used more than once",
                node_id=study.id,
            ))
        seen_ids.add(study.id)
        _validate_study(study, model, known, errors)

    summary = summarize_network(model)
    unused = [name for name in known if summary.n_studies.get(name, 0) == 0]
    for name in unused:
        errors.append(ValidationError(
            code="unused_treatment",
            message=f"Treatment '{name}' is not in any study",
            field="treatments",
        ))
    used_components = [c for c in summary.components if any(summary.n_studies.get(n, 0) for n in c)]
    if len(used_components) > 1:
        groups = "; ".join(" + ".join(component) for component in used_components)
        errors.append(ValidationError(
            code="disconnected_network",
            message=f"The network is not connected - {len(used_components)} separate groups: {groups}",
        ))
    return errors
