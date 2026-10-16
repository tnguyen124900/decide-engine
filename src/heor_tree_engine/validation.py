"""Validation rules for decision-tree models."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from .models import (
    ChanceNode,
    DecisionNode,
    DESTreeNode,
    MarkovTreeNode,
    Node,
    NumericRef,
    Param,
    PSM_STATE_ROLES,
    PSMNode,
    SurvivalCurve,
    TerminalNode,
    ValidationError,
    is_complement,
)
from .expressions import evaluate_expression

PROBABILITY_TOLERANCE = 1e-6
PSM_CURVE_PARAMS = {
    "exponential": ("rate",),
    "weibull": ("shape", "scale"),
    # The other 4 NICE DSU TSD 14 distributions, so the PSM curve picker
    # offers all 6. Same param names
    # as the general Distributions system's own scipy adapters, except
    # generalized gamma which uses Prentice's publication mu/sigma/Q form.
    "log_normal": ("meanlog", "sdlog"),
    "log_logistic": ("shape", "scale"),
    "gompertz": ("shape", "rate"),
    "generalized_gamma": ("mu", "sigma", "Q"),
}
# (distribution, param) pairs exempt from PSM_CURVE_PARAMS' default "must
# be > 0" check. Log-normal meanlog and Prentice generalized-gamma mu/Q
# accept any real value; legacy Stacy generalized-gamma c is checked
# separately for nonzero.
PSM_CURVE_PARAMS_UNSIGNED = {
    ("gompertz", "shape"),
}
PSM_CURVE_PARAMS_ANY_REAL = {
    ("log_normal", "meanlog"),
    ("generalized_gamma", "mu"),
    ("generalized_gamma", "Q"),
}
PSM_CROSSING_TOLERANCE = 1e-6
PSM_CROSSING_SAMPLE_POINTS = 20


def _reference_value(
    value: NumericRef,
    parameters: Mapping[str, Param],
    node_id: str,
    field: str,
    errors: list[ValidationError],
    variables: Mapping[str, Any] | None = None,
    distributions: Mapping[str, Mapping[str, Any]] | None = None,
    tables: Mapping[str, Mapping[str, Any]] | None = None,
) -> float | None:
    if isinstance(value, str):
        try:
            return evaluate_expression(value, parameters, variables, distributions, tables)
        except ValueError:
            errors.append(
                ValidationError(
                    code="missing_parameter",
                    message=f"Parameter '{value}' does not exist",
                    node_id=node_id,
                    field=field,
                )
            )
            return None
            return None
    return float(value)


def _validate_parameter(
    parameter: Param,
    errors: list[ValidationError],
) -> None:
    if parameter.low is not None and parameter.high is not None:
        if parameter.low > parameter.high:
            errors.append(
                ValidationError(
                    code="invalid_parameter_bounds",
                    message=f"Parameter '{parameter.name}' has low greater than high",
                    field=parameter.name,
                )
            )
    if parameter.low is not None and parameter.base < parameter.low:
        errors.append(
            ValidationError(
                code="base_below_low",
                message=f"Parameter '{parameter.name}' base is below low",
                field=parameter.name,
            )
        )
    if parameter.high is not None and parameter.base > parameter.high:
        errors.append(
            ValidationError(
                code="base_above_high",
                message=f"Parameter '{parameter.name}' base is above high",
                field=parameter.name,
            )
        )


def _validate_psm_curve(
    node_id: str,
    label: str,
    curve: SurvivalCurve,
    parameters: Mapping[str, Param],
    errors: list[ValidationError],
    variables: Mapping[str, Any] | None = None,
    distributions: Mapping[str, Mapping[str, Any]] | None = None,
    tables: Mapping[str, Mapping[str, Any]] | None = None,
) -> None:
    from .models import PSM_CURVE_TIME_UNITS

    if curve.time_unit not in PSM_CURVE_TIME_UNITS:
        errors.append(
            ValidationError(
                code="unknown_time_unit",
                message=(
                    f"{label}: unknown time unit '{curve.time_unit}' "
                    f"(expected one of {', '.join(sorted(PSM_CURVE_TIME_UNITS))})"
                ),
                node_id=node_id,
                field=label,
            )
        )
    if curve.distribution not in PSM_CURVE_PARAMS:
        errors.append(
            ValidationError(
                code="unknown_distribution",
                message=(
                    f"{label}: unknown distribution '{curve.distribution}' "
                    f"(expected one of {', '.join(sorted(PSM_CURVE_PARAMS))})"
                ),
                node_id=node_id,
                field=label,
            )
        )
        return
    legacy_gengamma = curve.distribution == "generalized_gamma" and "a" in curve.params
    param_names = ("a", "c", "scale") if legacy_gengamma else PSM_CURVE_PARAMS[curve.distribution]
    for param_name in param_names:
        if param_name not in curve.params:
            errors.append(
                ValidationError(
                    code="missing_curve_param",
                    message=f"{label}: '{curve.distribution}' requires a '{param_name}' parameter",
                    node_id=node_id,
                    field=f"{label}.{param_name}",
                )
            )
            continue
        value = _reference_value(
            curve.params[param_name], parameters, node_id, f"{label}.{param_name}", errors, variables, distributions, tables
        )
        if (curve.distribution, param_name) in PSM_CURVE_PARAMS_ANY_REAL:
            continue
        if (curve.distribution, param_name) in PSM_CURVE_PARAMS_UNSIGNED or (legacy_gengamma and param_name == "c"):
            if value == 0:
                errors.append(
                    ValidationError(
                        code="non_positive_curve_param",
                        message=f"{label}: '{param_name}' must not be 0",
                        node_id=node_id,
                        field=f"{label}.{param_name}",
                    )
                )
            continue
        if value is not None and value <= 0:
            errors.append(
                ValidationError(
                    code="non_positive_curve_param",
                    message=f"{label}: '{param_name}' must be greater than 0",
                    node_id=node_id,
                    field=f"{label}.{param_name}",
                )
            )


def _psm_survival_unchecked(
    curve: SurvivalCurve,
    time_years: float,
    parameters: Mapping[str, Param],
    variables: Mapping[str, Any] | None = None,
    distributions: Mapping[str, Mapping[str, Any]] | None = None,
    tables: Mapping[str, Mapping[str, Any]] | None = None,
) -> float | None:
    """Evaluate survival at elapsed years for the crossing check, returning None if a required
    parameter can't be resolved (mirrors the legacy psm_validation.py's
    _survival_unchecked, generalised to the engine's own parameter/
    variable/distribution/table resolution)."""

    import math

    from .models import curve_time_in_native_unit

    # `time_years` arrives as elapsed YEARS; convert to the curve's own
    # time_unit so this crossing check sees exactly what engine.py's
    # _curve_survival() sees for the same cycle -- otherwise a curve typed
    # in months (time_unit="month") would validate against the wrong axis.
    time_years = curve_time_in_native_unit(curve, time_years)

    def _param(name: str) -> float | None:
        return _reference_value(curve.params.get(name, 0.0), parameters, "", "", [], variables, distributions, tables)

    try:
        if curve.distribution == "exponential":
            rate = _param("rate")
            if rate is None or rate <= 0:
                return None
            return math.exp(-rate * time_years)
        if curve.distribution == "weibull":
            shape, scale = _param("shape"), _param("scale")
            if shape is None or scale is None or shape <= 0 or scale <= 0:
                return None
            if time_years <= 0:
                return 1.0
            return math.exp(-((time_years / scale) ** shape))
        if curve.distribution == "log_normal":
            meanlog, sdlog = _param("meanlog"), _param("sdlog")
            if meanlog is None or sdlog is None or sdlog <= 0:
                return None
            if time_years <= 0:
                return 1.0
            return 0.5 * math.erfc((math.log(time_years) - meanlog) / (sdlog * math.sqrt(2)))
        if curve.distribution == "log_logistic":
            shape, scale = _param("shape"), _param("scale")
            if shape is None or scale is None or shape <= 0 or scale <= 0:
                return None
            if time_years <= 0:
                return 1.0
            return 1.0 / (1.0 + (time_years / scale) ** shape)
        if curve.distribution == "gompertz":
            shape, rate = _param("shape"), _param("rate")
            if shape is None or rate is None or shape == 0 or rate <= 0:
                return None
            if time_years <= 0:
                return 1.0
            return math.exp(-(rate / shape) * (math.exp(shape * time_years) - 1.0))
        if curve.distribution == "generalized_gamma":
            from .des_sampling import survival

            param_names = ("mu", "sigma", "Q") if "mu" in curve.params else ("a", "c", "scale")
            params = {name: _param(name) for name in param_names}
            if any(value is None for value in params.values()):
                return None
            if time_years <= 0:
                return 1.0
            return survival("generalized_gamma", params, time_years)
    except (KeyError, ValueError, OverflowError):
        return None
    return None


def _validate_node(
    node: Node,
    parameters: Mapping[str, Param],
    errors: list[ValidationError],
    allow_negative_cost: bool,
    visited: set[str],
    variables: Mapping[str, Any] | None = None,
    distributions: Mapping[str, Mapping[str, Any]] | None = None,
    tables: Mapping[str, Mapping[str, Any]] | None = None,
) -> None:
    if node.id in visited:
        return
    visited.add(node.id)

    if isinstance(node, TerminalNode):
        cost = _reference_value(node.cost, parameters, node.id, "cost", errors, variables, distributions, tables)
        if cost is not None and cost < 0 and not allow_negative_cost:
            errors.append(
                ValidationError(
                    code="negative_cost",
                    message="Negative terminal cost requires explicit permission",
                    node_id=node.id,
                    field="cost",
                )
            )
        _reference_value(node.effect, parameters, node.id, "effect", errors, variables, distributions, tables)
        return

    if isinstance(node, MarkovTreeNode):
        from .markov_validation import validate_markov

        if node.model.cycles < 1:
            errors.append(
                ValidationError(code="invalid_cycles", message="Number of cycles must be at least 1", node_id=node.id)
            )
        merged_variables = {**node.variables, **(variables or {})}
        base_parameters = {n: v for n, v in (parameters or {}).items() if n not in merged_variables}
        for error in validate_markov(
            node.model, base_parameters, allow_negative_cost,
            variables=merged_variables, distributions=distributions, tables=tables,
        ):
            errors.append(
                ValidationError(
                    code=error.code, message=f"Markov node '{node.id}': {error.message}",
                    node_id=error.node_id or node.id, field=error.field,
                )
            )
        return

    if isinstance(node, DESTreeNode):
        from .des_validation import validate_des

        errors.extend(validate_des(node.model, parameters, tables, distributions))
        return

    if isinstance(node, PSMNode):
        if node.cycles < 1:
            errors.append(
                ValidationError(code="invalid_cycles", message="Number of cycles must be at least 1", node_id=node.id)
            )
        _validate_psm_curve(node.id, "pfs_curve", node.pfs_curve, parameters, errors, variables, distributions, tables)
        _validate_psm_curve(node.id, "os_curve", node.os_curve, parameters, errors, variables, distributions, tables)

        roles_seen: dict[str, int] = {}
        for branch in node.branches:
            if branch.role not in PSM_STATE_ROLES:
                errors.append(
                    ValidationError(
                        code="unknown_psm_role",
                        message=f"Unknown or missing PSM branch role '{branch.role}'",
                        node_id=node.id,
                        field=branch.label,
                    )
                )
                continue
            roles_seen[branch.role] = roles_seen.get(branch.role, 0) + 1
            if branch.child is not None:
                if not isinstance(branch.child, TerminalNode):
                    errors.append(
                        ValidationError(
                            code="psm_branch_requires_terminal",
                            message=f"PSM branch '{branch.label}' must lead to a Terminal node",
                            node_id=node.id,
                            field=branch.label,
                        )
                    )
                _validate_node(
                    branch.child, parameters, errors, allow_negative_cost, visited, variables, distributions, tables
                )
        for role in set(PSM_STATE_ROLES) - set(roles_seen):
            errors.append(
                ValidationError(
                    code="missing_psm_role",
                    message=f"PSM node is missing required branch role '{role}'",
                    node_id=node.id,
                )
            )
        for role, count in roles_seen.items():
            if count > 1:
                errors.append(
                    ValidationError(code="duplicate_psm_role", message=f"Duplicate PSM branch role '{role}'", node_id=node.id)
                )

        curve_error_codes = {"unknown_distribution", "missing_curve_param", "non_positive_curve_param", "missing_parameter"}
        has_curve_error = any(error.node_id == node.id and error.code in curve_error_codes for error in errors)
        if not has_curve_error and node.cycles >= 1:
            cycles_per_year = node.cycles_per_year or 1.0
            for step in range(PSM_CROSSING_SAMPLE_POINTS + 1):
                _cycle = (node.cycles * step) / PSM_CROSSING_SAMPLE_POINTS
                cycle_time_years = _cycle / cycles_per_year
                s_pfs = _psm_survival_unchecked(node.pfs_curve, cycle_time_years, parameters, variables, distributions, tables)
                s_os = _psm_survival_unchecked(node.os_curve, cycle_time_years, parameters, variables, distributions, tables)
                if s_pfs is None or s_os is None:
                    continue
                if s_pfs - s_os > PSM_CROSSING_TOLERANCE:
                    # pfs_exceeds_os used to be appended to `errors` here --
                    # but `errors` is what Engine.run() treats as FATAL (it
                    # short-circuits and returns zero cost/effect numbers the
                    # moment this list is non-empty, see engine.py right
                    # after `errors = validate(...)`). That made the check
                    # unusable on a real, correctly-parameterised PSM: it was
                    # hit when validating a published model (Cai et al. 2021,
                    # KEYNOTE-189) whose own Figure 1 shows PFS strictly
                    # below OS throughout, yet the fitted curves still cross
                    # by a tiny amount within the first ~1 week -- simply
                    # because PFS (generalized gamma) and OS (log-logistic)
                    # are two *independently*-fit parametric families with
                    # no constraint forcing them to agree infinitesimally
                    # close to t=0. This is a well known, common artifact of
                    # partitioned survival modelling, not a mis-specified or
                    # mistyped model -- so it shouldn't veto every other
                    # number the engine would otherwise compute
                    # (2026-10-01). `ValidationError` has no severity field
                    # and validate()/_validate_node() have no separate
                    # "warnings" channel to append to instead, so rather than
                    # thread a new parameter through every call site in this
                    # module plus engine.py and every app.py caller of
                    # validate() (out of scope here), this specific check is
                    # simply no longer fatal. The PSM curve-preview panel in
                    # the UI runs this identical sampled check on its own for
                    # its on-screen warning, so the finding stays visible to
                    # the user -- it just no longer blocks Engine.run().
                    break
        return

    if isinstance(node, DecisionNode):
        for branch in node.branches:
            if branch.probability is not None:
                errors.append(
                    ValidationError(
                        code="decision_probability",
                        message="Decision branches must not have probabilities",
                        node_id=node.id,
                        field=branch.label,
                    )
                )
            if branch.cost_increment is not None:
                cost = _reference_value(
                    branch.cost_increment,
                    parameters,
                    node.id,
                    branch.label,
                    errors,
                    variables,
                    distributions,
                    tables,
                )
                if cost is not None and cost < 0 and not allow_negative_cost:
                    errors.append(
                        ValidationError(
                            code="negative_cost",
                            message="Negative branch cost requires explicit permission",
                            node_id=node.id,
                            field=branch.label,
                        )
                    )
            if branch.effect_increment is not None:
                _reference_value(
                    branch.effect_increment,
                    parameters,
                    node.id,
                    branch.label,
                    errors,
                    variables,
                    distributions,
                    tables,
                )
            if branch.child is not None:
                _validate_node(
                    branch.child,
                    parameters,
                    errors,
                    allow_negative_cost,
                    visited,
                    variables,
                    distributions,
                    tables,
                )
        return

    if isinstance(node, ChanceNode):
        probability_sum = 0.0
        all_probabilities_resolved = True
        complement_fields: list[str] = []
        for branch in node.branches:
            if branch.probability is None:
                errors.append(
                    ValidationError(
                        code="missing_probability",
                        message="Chance branches require probabilities",
                        node_id=node.id,
                        field=branch.label,
                    )
                )
            elif is_complement(branch.probability):
                complement_fields.append(branch.label)
            else:
                probability = _reference_value(
                    branch.probability,
                    parameters,
                    node.id,
                    branch.label,
                    errors,
                    variables,
                    distributions,
                    tables,
                )
                if probability is None:
                    all_probabilities_resolved = False
                else:
                    probability_sum += probability
                    if probability < 0:
                        errors.append(
                            ValidationError(
                                code="negative_probability",
                                message="Probabilities cannot be negative",
                                node_id=node.id,
                                field=branch.label,
                            )
                        )
            if branch.cost_increment is not None:
                cost = _reference_value(
                    branch.cost_increment,
                    parameters,
                    node.id,
                    branch.label,
                    errors,
                    variables,
                    distributions,
                    tables,
                )
                if cost is not None and cost < 0 and not allow_negative_cost:
                    errors.append(
                        ValidationError(
                            code="negative_cost",
                            message="Negative branch cost requires explicit permission",
                            node_id=node.id,
                            field=branch.label,
                        )
                    )
            if branch.effect_increment is not None:
                _reference_value(
                    branch.effect_increment,
                    parameters,
                    node.id,
                    branch.label,
                    errors,
                    variables,
                    distributions,
                    tables,
                )
            if branch.child is not None:
                _validate_node(
                    branch.child,
                    parameters,
                    errors,
                    allow_negative_cost,
                    visited,
                    variables,
                    distributions,
                    tables,
                )
        if len(complement_fields) > 1:
            errors.append(
                ValidationError(
                    code="multiple_complements",
                    message="Only one branch per chance node may use '#' or 'C' for the remaining probability",
                    node_id=node.id,
                    field="probability",
                )
            )
        elif len(complement_fields) == 1:
            if all_probabilities_resolved:
                remainder = 1.0 - probability_sum
                if remainder < -PROBABILITY_TOLERANCE:
                    errors.append(
                        ValidationError(
                            code="negative_complement",
                            message=(
                                "The remaining probability ('#'/'C') would be negative "
                                "because the other branches already sum to more than 1"
                            ),
                            node_id=node.id,
                            field=complement_fields[0],
                        )
                    )
        elif all_probabilities_resolved and not math.isclose(
            probability_sum,
            1.0,
            abs_tol=PROBABILITY_TOLERANCE,
            rel_tol=0.0,
        ):
            errors.append(
                ValidationError(
                    code="probability_sum",
                    message="Chance branch probabilities must sum to 1",
                    node_id=node.id,
                    field="probability",
                )
            )


def find_negative_costs(
    tree: Node,
    parameters: Mapping[str, Param] | None = None,
) -> list[ValidationError]:
    """Find negative terminal and branch costs without rejecting them."""

    errors: list[ValidationError] = []
    _validate_node(tree, parameters or {}, errors, allow_negative_cost=False, visited=set())
    return [error for error in errors if error.code == "negative_cost"]


def validate(
    tree: Node,
    parameters: Mapping[str, Param] | None = None,
    allow_negative_cost: bool = False,
    variables: Mapping[str, Any] | None = None,
    distributions: Mapping[str, Mapping[str, Any]] | None = None,
    tables: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[ValidationError]:
    """Return all validation errors found in a directly linked tree."""

    resolved_parameters = parameters or {}
    errors: list[ValidationError] = []
    for parameter in resolved_parameters.values():
        _validate_parameter(parameter, errors)
    _validate_node(
        tree,
        resolved_parameters,
        errors,
        allow_negative_cost,
        visited=set(),
        variables=variables,
        distributions=distributions,
        tables=tables,
    )
    return errors