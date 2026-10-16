"""Build a MicrosimModel from its flat JSON representation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .markov_parser import _numeric_ref, _parse_state, _parse_transition
from .microsim_models import (
    PSM_STATE_ROLES,
    TRACKER_RESET_TIMINGS,
    MicrosimModel,
    MicrosimPSMNode,
    PSMRoleOutcomes,
    Tracker,
)
from .models import SurvivalCurve


def _parse_tracker(raw: Mapping[str, Any]) -> Tracker:
    if "name" not in raw or not str(raw["name"]).strip():
        raise ValueError("A Tracker requires a 'name'")
    raw_reset_rules = raw.get("reset_rules") or {}
    if not isinstance(raw_reset_rules, Mapping):
        raise ValueError(f"Tracker '{raw['name']}' reset_rules must be a mapping of state_id -> value")
    reset_rules = {str(state_id): _numeric_ref(value) for state_id, value in raw_reset_rules.items()}
    update_formula = raw.get("update_formula")
    return Tracker(
        name=str(raw["name"]),
        initial_value=_numeric_ref(raw.get("initial_value", 0.0)),
        update_formula=_numeric_ref(update_formula) if update_formula not in (None, "") else None,
        reset_rules=reset_rules,
        include_in_output=bool(raw.get("include_in_output", True)),
    )


def _parse_curve(raw: Mapping[str, Any]) -> SurvivalCurve:
    if "distribution" not in raw:
        raise ValueError("A survival curve requires a 'distribution'")
    return SurvivalCurve(
        distribution=str(raw["distribution"]),
        params={str(key): _numeric_ref(value) for key, value in raw.get("params", {}).items()},
    )


def _parse_psm_role_outcomes(raw: Mapping[str, Any]) -> PSMRoleOutcomes:
    raw_extra = raw.get("extra_outcomes") or {}
    return PSMRoleOutcomes(
        cost=_numeric_ref(raw.get("cost", 0.0)),
        effect=_numeric_ref(raw.get("effect", 0.0)),
        extra_outcomes={str(key): _numeric_ref(value) for key, value in raw_extra.items()},
    )


def _parse_psm_node(raw: Mapping[str, Any]) -> MicrosimPSMNode:
    if "id" not in raw:
        raise ValueError("A PSM node requires an 'id'")
    node_id = str(raw["id"])
    if "pfs_curve" not in raw or "os_curve" not in raw:
        raise ValueError(f"PSM node '{node_id}' requires 'pfs_curve' and 'os_curve'")
    raw_roles = raw.get("roles") or {}
    roles = {
        role: _parse_psm_role_outcomes(raw_roles[role] if role in raw_roles else {})
        for role in PSM_STATE_ROLES
    }
    return MicrosimPSMNode(
        id=node_id,
        name=str(raw.get("name", node_id)),
        pfs_curve=_parse_curve(raw["pfs_curve"]),
        os_curve=_parse_curve(raw["os_curve"]),
        roles=roles,
        initial_prob=_numeric_ref(raw.get("initial_prob", 0.0)),
    )


def resolve_microsim(document: Mapping[str, Any]) -> MicrosimModel:
    """Parse a flat Microsimulation JSON document into a MicrosimModel.

    States/transitions are parsed with the exact same helpers
    resolve_markov() uses (_parse_state/_parse_transition) -- states are
    structurally identical between a Markov cohort and a Microsimulation
    model, so nothing extra is needed for those two fields.
    """

    if "states" not in document:
        raise ValueError("Microsimulation document requires 'states'")
    states = [_parse_state(raw) for raw in document["states"]]
    ids = [state.id for state in states]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate state id in Microsimulation document")
    transitions = [_parse_transition(raw) for raw in document.get("transitions", [])]

    psm_nodes = [_parse_psm_node(raw) for raw in document.get("psm_nodes", [])]
    psm_node_ids = [node.id for node in psm_nodes]
    if len(psm_node_ids) != len(set(psm_node_ids)):
        raise ValueError("Duplicate PSM node id in Microsimulation document")
    if set(psm_node_ids) & set(ids):
        raise ValueError("A PSM node id collides with a state id in Microsimulation document")

    trackers = [_parse_tracker(raw) for raw in document.get("trackers", [])]
    tracker_names = [tracker.name for tracker in trackers]
    if len(tracker_names) != len(set(tracker_names)):
        raise ValueError("Duplicate tracker name in Microsimulation document")

    reset_timing = str(document.get("tracker_reset_timing", "on_entry"))
    if reset_timing not in TRACKER_RESET_TIMINGS:
        raise ValueError(
            f"tracker_reset_timing must be one of {TRACKER_RESET_TIMINGS}, got '{reset_timing}'"
        )

    # n_patients is intentionally NOT range-checked here (mirrors how
    # tenure_count isn't either): a structurally valid but out-of-range
    # value is a *soft* validate_microsim() error (invalid_n_patients),
    # not a hard parse-time exception, so the UI can surface it inline
    # next to the field instead of failing to even load the document.
    n_patients = int(document.get("n_patients", 1000))
    # Same soft-validation treatment as n_patients above: out-of-range is
    # a validate_microsim() concern, not a parse-time exception.
    patient_trace_sample_size = int(document.get("patient_trace_sample_size", 0))

    return MicrosimModel(
        id=str(document.get("id", "microsim")),
        name=str(document.get("name", "Microsimulation model")),
        states=states,
        transitions=transitions,
        cycles=int(document.get("cycles", 10)),
        half_cycle_correction=bool(document.get("half_cycle_correction", True)),
        n_patients=n_patients,
        random_seed=int(document.get("random_seed", 1)),
        tracker_reset_timing=reset_timing,
        trackers=trackers,
        patient_trace_sample_size=patient_trace_sample_size,
        psm_nodes=psm_nodes,
    )
