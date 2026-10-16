"""Validation rules for parametric partitioned survival models."""

from __future__ import annotations

from collections.abc import Mapping

from .models import NumericRef, Param, ValidationError
from .psm_models import PSM_STATE_ROLES, PartitionedSurvivalModel, SurvivalCurve

CURVE_PARAMS = {
    "exponential": ("rate",),
    "weibull": ("shape", "scale"),
    "log_normal": ("meanlog", "sdlog"),
    "log_logistic": ("shape", "scale"),
    "gompertz": ("shape", "rate"),
    "generalized_gamma": ("mu", "sigma", "Q"),
}
CURVE_PARAMS_NONZERO = {
    ("gompertz", "shape"),
}
CURVE_PARAMS_ANY_REAL = {
    ("log_normal", "meanlog"),
    ("generalized_gamma", "mu"),
    ("generalized_gamma", "Q"),
}
LEGACY_GENGAMMA_PARAMS = ("a", "c", "scale")


def curve_param_names(curve: SurvivalCurve) -> tuple[str, ...]:
    if curve.distribution == "generalized_gamma" and "a" in curve.params:
        return LEGACY_GENGAMMA_PARAMS
    return CURVE_PARAMS.get(curve.distribution, ())
CROSSING_TOLERANCE = 1e-6
CROSSING_SAMPLE_POINTS = 20


def _reference_value(
    value: NumericRef,
    parameters: Mapping[str, Param],
    field: str,
    errors: list[ValidationError],
) -> float | None:
    if isinstance(value, str):
        parameter = parameters.get(value)
        if parameter is None:
            errors.append(
                ValidationError(code="missing_parameter", message=f"Parameter '{value}' does not exist", field=field)
            )
            return None
        return parameter.base
    return float(value)


def _validate_curve(
    label: str,
    curve: SurvivalCurve,
    parameters: Mapping[str, Param],
    errors: list[ValidationError],
) -> None:
    if curve.distribution not in CURVE_PARAMS:
        errors.append(
            ValidationError(
                code="unknown_distribution",
                message=f"{label}: unknown distribution '{curve.distribution}' (expected exponential or weibull)",
                field=label,
            )
        )
        return
    legacy_gengamma = curve.distribution == "generalized_gamma" and "a" in curve.params
    for param_name in curve_param_names(curve):
        if param_name not in curve.params:
            errors.append(
                ValidationError(
                    code="missing_curve_param",
                    message=f"{label}: '{curve.distribution}' requires a '{param_name}' parameter",
                    field=f"{label}.{param_name}",
                )
            )
            continue
        value = _reference_value(curve.params[param_name], parameters, f"{label}.{param_name}", errors)
        if (curve.distribution, param_name) in CURVE_PARAMS_ANY_REAL:
            continue
        if (curve.distribution, param_name) in CURVE_PARAMS_NONZERO or (legacy_gengamma and param_name == "c"):
            if value is not None and value == 0:
                errors.append(
                    ValidationError(
                        code="non_positive_curve_param",
                        message=f"{label}: '{param_name}' must not be 0",
                        field=f"{label}.{param_name}",
                    )
                )
        elif (curve.distribution, param_name) != ("log_normal", "meanlog") and value is not None and value <= 0:
            errors.append(
                ValidationError(
                    code="non_positive_curve_param",
                    message=f"{label}: '{param_name}' must be greater than 0",
                    field=f"{label}.{param_name}",
                )
            )


def _survival_unchecked(curve: SurvivalCurve, _cycle: float, parameters: Mapping[str, Param]) -> float | None:
    """Evaluate survival at _cycle, or return None if a parameter can't be resolved."""

    from .des_sampling import survival

    try:
        if curve.distribution not in CURVE_PARAMS:
            return None
        resolved_params = {
            name: _reference_value(curve.params[name], parameters, "", [])
            for name in curve_param_names(curve)
        }
        if any(value is None for value in resolved_params.values()):
            return None
        return survival(curve.distribution, resolved_params, _cycle)
    except (KeyError, ValueError, OverflowError, ZeroDivisionError):
        return None


def validate_psm(
    model: PartitionedSurvivalModel,
    parameters: Mapping[str, Param] | None = None,
    allow_negative_cost: bool = False,
) -> list[ValidationError]:
    """Return all validation errors found in a partitioned survival model."""

    resolved_parameters = parameters or {}
    errors: list[ValidationError] = []

    if model.cycles < 1:
        errors.append(ValidationError(code="invalid_cycles", message="Number of cycles must be at least 1"))

    _validate_curve("pfs_curve", model.pfs_curve, resolved_parameters, errors)
    _validate_curve("os_curve", model.os_curve, resolved_parameters, errors)

    roles = [state.role for state in model.states]
    missing_roles = set(PSM_STATE_ROLES) - set(roles)
    for role in missing_roles:
        errors.append(ValidationError(code="missing_state", message=f"Missing required state role '{role}'", node_id=role))
    seen_roles: set[str] = set()
    for state in model.states:
        if state.role not in PSM_STATE_ROLES:
            errors.append(
                ValidationError(code="unknown_state_role", message=f"Unknown state role '{state.role}'", node_id=state.role)
            )
            continue
        if state.role in seen_roles:
            errors.append(
                ValidationError(code="duplicate_state", message=f"Duplicate state role '{state.role}'", node_id=state.role)
            )
        seen_roles.add(state.role)
        cost = _reference_value(state.cost, resolved_parameters, f"{state.role}.cost", errors)
        if cost is not None and cost < 0 and not allow_negative_cost:
            errors.append(
                ValidationError(
                    code="negative_cost",
                    message="Negative state cost requires explicit permission",
                    node_id=state.role,
                    field="cost",
                )
            )
        _reference_value(state.effect, resolved_parameters, f"{state.role}.effect", errors)

    # Only check curve crossing once both curves are individually valid.
    curve_errors = [
        error
        for error in errors
        if error.code in ("unknown_distribution", "missing_curve_param", "non_positive_curve_param", "missing_parameter")
    ]
    if not curve_errors and model.cycles >= 1:
        for step in range(CROSSING_SAMPLE_POINTS + 1):
            _cycle = (model.cycles * step) / CROSSING_SAMPLE_POINTS
            s_pfs = _survival_unchecked(model.pfs_curve, _cycle, resolved_parameters)
            s_os = _survival_unchecked(model.os_curve, _cycle, resolved_parameters)
            if s_pfs is None or s_os is None:
                continue
            if s_pfs - s_os > CROSSING_TOLERANCE:
                errors.append(
                    ValidationError(
                        code="pfs_exceeds_os",
                        message=(
                            f"PFS survival exceeds OS survival at _cycle={_cycle:.2f} "
                            f"(PFS={s_pfs:.4f} > OS={s_os:.4f}) - progression-free population "
                            "can't be larger than the population still alive"
                        ),
                    )
                )
                break

    return errors
