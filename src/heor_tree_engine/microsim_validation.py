"""Validation rules for patient-level (microsimulation) models."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .markov_validation import _check_probability_group, _reference_value
from .microsim_models import MicrosimModel
from .models import PSM_STATE_ROLES, NumericRef, Param, ValidationError
from .validation import (
    PSM_CROSSING_SAMPLE_POINTS,
    PSM_CROSSING_TOLERANCE,
    _psm_survival_unchecked,
    _validate_psm_curve,
)


def validate_microsim(
    model: MicrosimModel,
    parameters: Mapping[str, Param] | None = None,
    allow_negative_cost: bool = False,
    variables: Mapping[str, Any] | None = None,
    distributions: Mapping[str, Mapping[str, Any]] | None = None,
    tables: Mapping[str, Mapping[str, Any]] | None = None,
    cycles_per_year: float = 1.0,
) -> list[ValidationError]:
    """Return all validation errors found in a Microsimulation model.

    Reuses the exact same state/transition rules as a Markov cohort model
    (states/transitions are structurally identical -- see
    microsim_models.py) via markov_validation's own building blocks, then
    adds Tracker-specific checks: every tracker's own name is seeded as a
    placeholder 0.0 variable so a Tracker's update_formula/reset_rules can
    self-reference without a spurious
    'missing_parameter' error during validation.
    """

    resolved_parameters = parameters or {}
    errors: list[ValidationError] = []

    if not model.states and not model.psm_nodes:
        # A Microsim strategy can now legitimately consist of nothing
        # but its root converted straight to PSM (the supported
        # workflow: Clear model -> Change type the lone root to PSM,
        # no separate health state needed) -- only reject a model with
        # NEITHER kind of entry, not one that's all PSM nodes.
        errors.append(
            ValidationError(
                code="no_states",
                message="A Microsimulation model requires at least one state or PSM node",
            )
        )
        return errors

    if model.states and not model.psm_nodes and not any(state.is_real_state for state in model.states):
        errors.append(
            ValidationError(
                code="no_real_states",
                message=(
                    "A Microsimulation model needs at least one health state directly under the "
                    "strategy's root node (every state here is a branch/terminal node under another state)"
                ),
            )
        )
        return errors

    if model.n_patients < 1:
        errors.append(
            ValidationError(
                code="invalid_n_patients", message="Number of patients must be at least 1", field="n_patients"
            )
        )

    state_ids = [state.id for state in model.states]
    seen_ids: set[str] = set()
    for state_id in state_ids:
        if state_id in seen_ids:
            errors.append(
                ValidationError(code="duplicate_state", message=f"Duplicate state id '{state_id}'", node_id=state_id)
            )
        seen_ids.add(state_id)
    state_id_set = set(state_ids)

    psm_node_ids = [node.id for node in model.psm_nodes]
    seen_psm_node_ids: set[str] = set()
    for node_id in psm_node_ids:
        if node_id in seen_psm_node_ids:
            errors.append(
                ValidationError(code="duplicate_psm_node", message=f"Duplicate PSM node id '{node_id}'", node_id=node_id)
            )
        seen_psm_node_ids.add(node_id)
    for node_id in seen_psm_node_ids & state_id_set:
        errors.append(
            ValidationError(
                code="psm_node_id_collision",
                message=f"PSM node id '{node_id}' collides with a state id",
                node_id=node_id,
            )
        )
    psm_node_id_set = set(psm_node_ids)

    # Every Tracker name is seeded as a placeholder 0.0 variable up front
    # so any state's cost/effect/probability formula -- or a Tracker's own
    # update_formula/reset_rules -- can reference a Tracker's value
    # without a spurious 'missing_parameter' error during validation
    # (the real per-cycle value is only known at simulation time).
    tracker_names = [tracker.name for tracker in model.trackers]
    tracker_placeholder_variables = {**(variables or {}), **{name: 0.0 for name in tracker_names}}
    variables = tracker_placeholder_variables

    for state in model.states:
        if state.tenure_applied and state.tenure_count < 1:
            errors.append(
                ValidationError(
                    code="invalid_tenure_count",
                    message="Number of tenure stages must be at least 1",
                    node_id=state.id,
                    field="tenure_count",
                )
            )
        costs = [("cost", state.cost)]
        if state.tenure_applied:
            costs = [
                ("initial_cost", state.initial_cost),
                ("recurring_cost", state.recurring_cost if state.recurring_cost is not None else state.cost),
                ("terminal_cost", state.terminal_cost),
            ]
        for field_name, value in costs:
            cost = _reference_value(
                value, resolved_parameters, f"{state.id}.{field_name}", errors, variables, distributions, tables
            )
            if cost is not None and cost < 0 and not allow_negative_cost:
                errors.append(
                    ValidationError(
                        code="negative_cost",
                        message="Negative state cost requires explicit permission",
                        node_id=state.id,
                        field=field_name,
                    )
                )
        _reference_value(
            state.effect, resolved_parameters, f"{state.id}.effect", errors, variables, distributions, tables
        )

    for node in model.psm_nodes:
        _validate_psm_curve(node.id, "pfs_curve", node.pfs_curve, resolved_parameters, errors, variables, distributions, tables)
        _validate_psm_curve(node.id, "os_curve", node.os_curve, resolved_parameters, errors, variables, distributions, tables)

        for role_name in set(node.roles) - set(PSM_STATE_ROLES):
            errors.append(
                ValidationError(code="unknown_psm_role", message=f"Unknown PSM role '{role_name}'", node_id=node.id)
            )
        for role_name in set(PSM_STATE_ROLES) - set(node.roles):
            errors.append(
                ValidationError(
                    code="missing_psm_role", message=f"PSM node is missing required role '{role_name}'", node_id=node.id
                )
            )
        for role_name, role in node.roles.items():
            if role_name not in PSM_STATE_ROLES:
                continue
            cost = _reference_value(
                role.cost, resolved_parameters, f"{node.id}.{role_name}.cost", errors, variables, distributions, tables
            )
            if cost is not None and cost < 0 and not allow_negative_cost:
                errors.append(
                    ValidationError(
                        code="negative_cost",
                        message=f"Negative PSM role cost requires explicit permission ('{role_name}')",
                        node_id=node.id,
                        field=f"{role_name}.cost",
                    )
                )
            _reference_value(
                role.effect, resolved_parameters, f"{node.id}.{role_name}.effect", errors, variables, distributions, tables
            )

        curve_error_codes = {"unknown_distribution", "missing_curve_param", "non_positive_curve_param", "missing_parameter"}
        has_curve_error = any(error.node_id == node.id and error.code in curve_error_codes for error in errors)
        if not has_curve_error and model.cycles >= 1:
            horizon_years = model.cycles / (cycles_per_year or 1.0)
            for step in range(PSM_CROSSING_SAMPLE_POINTS + 1):
                t = (horizon_years * step) / PSM_CROSSING_SAMPLE_POINTS
                s_pfs = _psm_survival_unchecked(node.pfs_curve, t, resolved_parameters, variables, distributions, tables)
                s_os = _psm_survival_unchecked(node.os_curve, t, resolved_parameters, variables, distributions, tables)
                if s_pfs is None or s_os is None:
                    continue
                if s_pfs - s_os > PSM_CROSSING_TOLERANCE:
                    errors.append(
                        ValidationError(
                            code="pfs_exceeds_os",
                            message=(
                                f"PFS survival exceeds OS survival at t={t:.2f} years "
                                f"(PFS={s_pfs:.4f} > OS={s_os:.4f}) -- progression-free population "
                                "can't be larger than the population still alive"
                            ),
                            node_id=node.id,
                        )
                    )
                    break

    _check_probability_group(
        "the initial distribution",
        [(state.id, state.initial_prob) for state in model.states if state.is_real_state]
        + [(node.id, node.initial_prob) for node in model.psm_nodes],
        resolved_parameters,
        errors,
        multiple_complements_code="multiple_initial_complements",
        negative_complement_code="negative_initial_complement",
        sum_code="initial_distribution_sum",
        negative_code="negative_initial_probability",
        variables=variables,
        distributions=distributions,
        tables=tables,
    )

    outgoing: dict[str, list[tuple[str, NumericRef]]] = {state_id: [] for state_id in state_ids}
    seen_pairs: set[tuple[str, str]] = set()
    for transition in model.transitions:
        if transition.from_state in psm_node_id_set:
            errors.append(
                ValidationError(
                    code="psm_node_no_outgoing_transitions",
                    message=(
                        f"PSM node '{transition.from_state}' can't have outgoing transitions "
                        "(a patient's role inside it is determined by its own PFS/OS curves)"
                    ),
                    node_id=transition.from_state,
                )
            )
            continue
        if transition.from_state not in state_id_set:
            errors.append(
                ValidationError(
                    code="unknown_state",
                    message=f"Transition references unknown state '{transition.from_state}'",
                    node_id=transition.from_state,
                )
            )
            continue
        if transition.to_state not in state_id_set and transition.to_state not in psm_node_id_set:
            errors.append(
                ValidationError(
                    code="unknown_state",
                    message=f"Transition references unknown state '{transition.to_state}'",
                    node_id=transition.to_state,
                )
            )
            continue
        pair = (transition.from_state, transition.to_state)
        if pair in seen_pairs:
            errors.append(
                ValidationError(
                    code="duplicate_transition",
                    message=f"Duplicate transition '{transition.from_state}' -> '{transition.to_state}'",
                    node_id=transition.from_state,
                )
            )
            continue
        seen_pairs.add(pair)
        outgoing[transition.from_state].append((transition.to_state, transition.probability))

    states_by_id = {state.id: state for state in model.states}
    real_state_names = {state.name: state.id for state in model.states if state.is_real_state}

    # A chain of non-real (branch/terminal) nodes is walked in zero elapsed
    # time, so a loop made ONLY of non-real nodes would never reach a real
    # health state -- same rule as the Markov cohort engine.
    non_real_ids = {state.id for state in model.states if not state.is_real_state}
    _loop_color: dict[str, int] = {}

    def _find_non_real_loop(node_id: str) -> str | None:
        _loop_color[node_id] = 1
        for to_state, _probability in outgoing.get(node_id, []):
            if to_state not in non_real_ids:
                continue
            color = _loop_color.get(to_state, 0)
            if color == 1:
                return to_state
            if color == 0:
                found = _find_non_real_loop(to_state)
                if found is not None:
                    return found
        _loop_color[node_id] = 2
        return None

    for non_real_id in sorted(non_real_ids):
        if _loop_color.get(non_real_id, 0) == 0:
            loop_state = _find_non_real_loop(non_real_id)
            if loop_state is not None:
                errors.append(
                    ValidationError(
                        code="non_real_state_loop",
                        message=(
                            f"There is a loop entirely among branch/terminal nodes -- '{loop_state}' is reached "
                            "again without ever passing through a health state. A branch chain must "
                            "eventually reach a health state."
                        ),
                        node_id=loop_state,
                    )
                )
                break

    for state_id, entries in outgoing.items():
        state = states_by_id[state_id]
        if not entries:
            if not state.is_real_state:
                # A non-real (canvas-child) dead-end leaf never needs the
                # "absorbing" flag: the patient's cycle simply ends in the
                # real health state this leaf stands for (alias_of, or the
                # same display name as a fallback), and the next cycle they
                # move on from THAT state. Only flag a leaf that can't be
                # linked to any real state at all.
                alias_target = state.alias_of or real_state_names.get(state.name)
                if (
                    alias_target is None
                    or alias_target not in states_by_id
                    or not states_by_id[alias_target].is_real_state
                ):
                    errors.append(
                        ValidationError(
                            code="unresolved_alias",
                            message=(
                                f"State '{state_id}' has no outgoing transitions and doesn't "
                                f"match any real state by name or alias -- use 'relabel as "
                                f"terminal' on it to link it to the real state it represents"
                            ),
                            node_id=state_id,
                        )
                    )
                continue
            if not state.is_absorbing:
                errors.append(
                    ValidationError(
                        code="missing_transitions",
                        message=f"State '{state_id}' has no outgoing transitions (mark it absorbing if intended)",
                        node_id=state_id,
                    )
                )
            continue
        _check_probability_group(
            f"the outgoing transitions of '{state_id}'",
            entries,
            resolved_parameters,
            errors,
            multiple_complements_code="multiple_transition_complements",
            negative_complement_code="negative_transition_complement",
            sum_code="transition_sum",
            negative_code="negative_transition_probability",
            variables=variables,
            distributions=distributions,
            tables=tables,
        )

    seen_tracker_names: set[str] = set()
    for name in tracker_names:
        if name in seen_tracker_names:
            errors.append(ValidationError(code="duplicate_tracker", message=f"Duplicate tracker name '{name}'", node_id=name))
        seen_tracker_names.add(name)

    for tracker in model.trackers:
        _reference_value(
            tracker.initial_value,
            resolved_parameters,
            f"tracker.{tracker.name}.initial_value",
            errors,
            variables,
            distributions,
            tables,
        )
        formula = tracker.update_formula if tracker.update_formula is not None else tracker.name
        _reference_value(
            formula,
            resolved_parameters,
            f"tracker.{tracker.name}.update_formula",
            errors,
            tracker_placeholder_variables,
            distributions,
            tables,
        )
        for reset_state_id, reset_value in tracker.reset_rules.items():
            if reset_state_id not in state_id_set:
                errors.append(
                    ValidationError(
                        code="unknown_state",
                        message=f"Tracker '{tracker.name}' reset_rules references unknown state '{reset_state_id}'",
                        node_id=tracker.name,
                    )
                )
                continue
            _reference_value(
                reset_value,
                resolved_parameters,
                f"tracker.{tracker.name}.reset_rules.{reset_state_id}",
                errors,
                tracker_placeholder_variables,
                distributions,
                tables,
            )

    return errors
