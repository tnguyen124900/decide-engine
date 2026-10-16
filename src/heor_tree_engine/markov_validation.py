"""Validation rules for Markov cohort models."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from .markov_models import MarkovModel
from .models import NumericRef, Param, ValidationError, is_complement
from .expressions import evaluate_expression

PROBABILITY_TOLERANCE = 1e-6


def _reference_value(
    value: NumericRef,
    parameters: Mapping[str, Param],
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
                    field=field,
                )
            )
            return None
            return None
    return float(value)


def _check_probability_group(
    group_label: str,
    entries: list[tuple[str, NumericRef]],
    parameters: Mapping[str, Param],
    errors: list[ValidationError],
    *,
    multiple_complements_code: str,
    negative_complement_code: str,
    sum_code: str,
    negative_code: str,
    variables: Mapping[str, Any] | None = None,
    distributions: Mapping[str, Mapping[str, Any]] | None = None,
    tables: Mapping[str, Mapping[str, Any]] | None = None,
) -> None:
    """Shared complement/sum-to-1 logic for initial_prob and per-state transitions."""

    probability_sum = 0.0
    all_resolved = True
    complement_fields: list[str] = []
    for field_label, value in entries:
        if is_complement(value):
            complement_fields.append(field_label)
            continue
        resolved = _reference_value(value, parameters, field_label, errors, variables, distributions, tables)
        if resolved is None:
            all_resolved = False
            continue
        probability_sum += resolved
        if resolved < 0:
            errors.append(
                ValidationError(
                    code=negative_code,
                    message="Probabilities cannot be negative",
                    field=field_label,
                )
            )

    if len(complement_fields) > 1:
        errors.append(
            ValidationError(
                code=multiple_complements_code,
                message=f"Only one entry in {group_label} may use '#' or 'C' for the remaining probability",
                field=complement_fields[0],
            )
        )
    elif len(complement_fields) == 1:
        if all_resolved:
            remainder = 1.0 - probability_sum
            if remainder < -PROBABILITY_TOLERANCE:
                errors.append(
                    ValidationError(
                        code=negative_complement_code,
                        message=(
                            "The remaining probability ('#'/'C') would be negative "
                            f"because the other entries in {group_label} already sum to more than 1"
                        ),
                        field=complement_fields[0],
                    )
                )
    elif all_resolved and not math.isclose(
        probability_sum, 1.0, abs_tol=PROBABILITY_TOLERANCE, rel_tol=0.0
    ):
        errors.append(
            ValidationError(
                code=sum_code,
                message=f"{group_label} must sum to 1",
            )
        )


def validate_markov(
    model: MarkovModel,
    parameters: Mapping[str, Param] | None = None,
    allow_negative_cost: bool = False,
    variables: Mapping[str, Any] | None = None,
    distributions: Mapping[str, Mapping[str, Any]] | None = None,
    tables: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[ValidationError]:
    """Return all validation errors found in a Markov model."""

    resolved_parameters = parameters or {}
    errors: list[ValidationError] = []

    if not model.states:
        errors.append(
            ValidationError(code="no_states", message="A Markov model requires at least one state")
        )
        return errors

    state_ids = [state.id for state in model.states]
    seen_ids: set[str] = set()
    for state_id in state_ids:
        if state_id in seen_ids:
            errors.append(
                ValidationError(code="duplicate_state", message=f"Duplicate state id '{state_id}'", node_id=state_id)
            )
        seen_ids.add(state_id)
    state_id_set = set(state_ids)

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
        for field, value in costs:
            cost = _reference_value(value, resolved_parameters, f"{state.id}.{field}", errors, variables, distributions, tables)
            if cost is not None and cost < 0 and not allow_negative_cost:
                errors.append(
                    ValidationError(
                        code="negative_cost",
                        message="Negative state cost requires explicit permission",
                        node_id=state.id,
                        field=field,
                    )
                )
        _reference_value(state.effect, resolved_parameters, f"{state.id}.effect", errors, variables, distributions, tables)

    # Initial distribution must sum to 1 (complement marker allowed on one state).
    _check_probability_group(
        "the initial distribution",
        [(state.id, state.initial_prob) for state in model.states],
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

    # Transitions must reference real states and cannot duplicate a (from, to) pair.
    outgoing: dict[str, list[tuple[str, NumericRef]]] = {state_id: [] for state_id in state_ids}
    seen_pairs: set[tuple[str, str]] = set()
    for transition in model.transitions:
        if transition.from_state not in state_id_set:
            errors.append(
                ValidationError(
                    code="unknown_state",
                    message=f"Transition references unknown state '{transition.from_state}'",
                    node_id=transition.from_state,
                )
            )
            continue
        if transition.to_state not in state_id_set:
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
    real_state_names = {
        state.name: state.id for state in model.states if state.is_real_state
    }
    for state_id, entries in outgoing.items():
        state = states_by_id[state_id]
        if not entries:
            if not state.is_real_state:
                # A non-real (canvas-child) dead-end leaf never needs the
                # "absorbing" flag: the engine resolves it straight into
                # the real state it aliases (via alias_of, or by matching
                # name as a fallback for models saved before alias_of
                # existed) instead of requiring it to have its own
                # transitions -- reaching a terminal-looking leaf simply
                # means the cohort already occupies that real state.
                # Only flag it if there's truly no real state it could
                # resolve into at all (alias_of set but unknown, AND no
                # name match) -- that would silently drop cohort mass.
                alias_target = state.alias_of or real_state_names.get(state.name)
                if alias_target is None or alias_target not in states_by_id or not states_by_id[alias_target].is_real_state:
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

    return errors
