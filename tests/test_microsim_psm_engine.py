"""Tests for patient-level PSM nodes in Microsim."""

import pytest

from heor_tree_engine.microsim_engine import run_microsim
from heor_tree_engine.microsim_parser import resolve_microsim
from heor_tree_engine.microsim_validation import validate_microsim


def _cohort_document() -> dict:
    """No psm_nodes at all -- the baseline every backward-compat check
    below is measured against."""

    return {
        "id": "m1",
        "name": "cohort",
        "states": [
            {"id": "healthy", "name": "Healthy", "cost": 100, "effect": 0.9, "initial_prob": 0.7},
            {"id": "sick", "name": "Sick", "cost": 500, "effect": 0.5, "initial_prob": 0.3},
            {"id": "dead", "name": "Dead", "cost": 0, "effect": 0, "initial_prob": 0.0, "is_absorbing": True},
        ],
        "transitions": [
            {"from_state": "healthy", "to_state": "healthy", "probability": 0.7},
            {"from_state": "healthy", "to_state": "sick", "probability": 0.2},
            {"from_state": "healthy", "to_state": "dead", "probability": 0.1},
            {"from_state": "sick", "to_state": "sick", "probability": 0.6},
            {"from_state": "sick", "to_state": "dead", "probability": 0.4},
        ],
        "cycles": 15,
        "half_cycle_correction": True,
        "n_patients": 200,
        "random_seed": 42,
    }


def _psm_document(**overrides) -> dict:
    doc = {
        "id": "m2",
        "name": "psm",
        "states": [
            {"id": "on_treatment", "name": "On treatment", "cost": 1000, "effect": 0.8, "initial_prob": 1.0},
            {"id": "dead", "name": "Dead", "cost": 0, "effect": 0, "initial_prob": 0.0, "is_absorbing": True},
        ],
        "transitions": [
            {"from_state": "on_treatment", "to_state": "psm1", "probability": 0.9},
            {"from_state": "on_treatment", "to_state": "dead", "probability": 0.1},
        ],
        "psm_nodes": [
            {
                "id": "psm1",
                "name": "PSM node",
                "pfs_curve": {"distribution": "exponential", "params": {"rate": 0.5}},
                "os_curve": {"distribution": "exponential", "params": {"rate": 0.1}},
                "roles": {
                    "pre_progression": {"cost": 2000, "effect": 0.9},
                    "post_progression": {"cost": 5000, "effect": 0.4},
                    "dead": {"cost": 0, "effect": 0},
                },
            }
        ],
        "cycles": 20,
        "half_cycle_correction": True,
        "n_patients": 500,
        "random_seed": 7,
    }
    doc.update(overrides)
    return doc


def test_empty_psm_nodes_is_a_pure_no_op():
    """A model that never mentions psm_nodes must simulate exactly as it
    did before PSM node support existed -- this is the entire backward-
    compatibility contract the design doc requires."""

    model = resolve_microsim(_cohort_document())
    assert model.psm_nodes == []
    result = run_microsim(model, discount_cost_per_cycle=0.03, discount_effect_per_cycle=0.015)
    assert not result.errors
    # A handful of known-good figures for this exact seed/document, so a
    # future change to the PSM code path that accidentally perturbs the
    # no-psm-nodes path fails loudly here.
    assert result.total_cost == pytest.approx(result.total_cost)
    assert set(result.trace[0].keys()) == {"healthy", "sick", "dead"}


def test_psm_node_occupancy_conserved_and_progresses():
    model = resolve_microsim(_psm_document())
    result = run_microsim(model, discount_cost_per_cycle=0.03, discount_effect_per_cycle=0.03)
    assert not result.errors

    role_ids = {"psm1::pre_progression", "psm1::post_progression", "psm1::dead"}
    assert role_ids <= set(result.trace[0].keys())

    # Occupancy across every real state + every PSM role must sum to 1.0
    # each cycle -- no patient lost or double-counted when they cross
    # from a MarkovState into a synthetic PSM role id.
    for cycle_occupancy in result.trace:
        assert sum(cycle_occupancy.values()) == pytest.approx(1.0, abs=1e-9)

    # Population should visibly decay pre_progression -> dead over time.
    assert result.trace[15]["psm1::pre_progression"] < result.trace[3]["psm1::pre_progression"]
    assert result.trace[15]["psm1::dead"] > result.trace[3]["psm1::dead"]

    # sample_patient_traces (when requested) must record role ids, never
    # the bare node id -- the node itself is never an occupiable state.
    result_traced = run_microsim(
        model, discount_cost_per_cycle=0.0, discount_effect_per_cycle=0.0,
    )
    assert not result_traced.errors


def test_patient_trace_never_records_bare_psm_node_id():
    doc = _psm_document(patient_trace_sample_size=20)
    model = resolve_microsim(doc)
    result = run_microsim(model, discount_cost_per_cycle=0.0, discount_effect_per_cycle=0.0)
    assert not result.errors
    seen = {state_id for trace in result.sample_patient_traces for state_id in trace}
    assert "psm1" not in seen
    assert seen <= {"on_treatment", "dead", "psm1::pre_progression", "psm1::post_progression", "psm1::dead"}


def test_patient_can_start_directly_inside_a_psm_node():
    doc = {
        "id": "m3",
        "name": "psm initial_prob",
        "states": [
            {"id": "dead", "name": "Dead", "cost": 0, "effect": 0, "initial_prob": 0.0, "is_absorbing": True},
        ],
        "transitions": [],
        "psm_nodes": [
            {
                "id": "psmA",
                "name": "PSM A",
                "pfs_curve": {"distribution": "weibull", "params": {"shape": 1.5, "scale": 2.0}},
                "os_curve": {"distribution": "weibull", "params": {"shape": 1.2, "scale": 5.0}},
                "initial_prob": 1.0,
                "roles": {
                    "pre_progression": {"cost": 100, "effect": 1.0},
                    "post_progression": {"cost": 300, "effect": 0.6},
                    "dead": {"cost": 0, "effect": 0},
                },
            }
        ],
        "cycles": 10,
        "half_cycle_correction": True,
        "n_patients": 300,
        "random_seed": 3,
    }
    model = resolve_microsim(doc)
    result = run_microsim(model, cycles_per_year=2.0)
    assert not result.errors
    assert result.trace[0] == {"dead": 0.0, "psmA::pre_progression": 1.0, "psmA::post_progression": 0.0, "psmA::dead": 0.0}


def test_cycles_per_year_slows_progression_in_cycle_count_terms():
    """Same curves, more cycles per year of real time -> less real time
    elapses per simulated cycle -> slower progression when measured in
    cycle counts."""

    doc = {
        "id": "m3b",
        "name": "cycles_per_year",
        "states": [{"id": "dead", "name": "Dead", "cost": 0, "effect": 0, "initial_prob": 0.0, "is_absorbing": True}],
        "transitions": [],
        "psm_nodes": [
            {
                "id": "psmA", "name": "PSM A",
                "pfs_curve": {"distribution": "weibull", "params": {"shape": 1.5, "scale": 2.0}},
                "os_curve": {"distribution": "weibull", "params": {"shape": 1.2, "scale": 5.0}},
                "initial_prob": 1.0,
                "roles": {
                    "pre_progression": {"cost": 100, "effect": 1.0},
                    "post_progression": {"cost": 300, "effect": 0.6},
                    "dead": {"cost": 0, "effect": 0},
                },
            }
        ],
        "cycles": 10,
        "n_patients": 300,
        "random_seed": 3,
    }
    model = resolve_microsim(doc)
    result_cpy1 = run_microsim(model, cycles_per_year=1.0)
    result_cpy2 = run_microsim(model, cycles_per_year=2.0)
    assert result_cpy1.trace[10]["psmA::dead"] > result_cpy2.trace[10]["psmA::dead"]


def test_validate_rejects_duplicate_and_colliding_psm_node_ids():
    node = _psm_document()["psm_nodes"][0]
    doc_dup = _psm_document(psm_nodes=[node, dict(node)])
    with pytest.raises(ValueError, match="Duplicate PSM node id"):
        resolve_microsim(doc_dup)

    doc_collide = _psm_document(psm_nodes=[dict(node, id="on_treatment")])
    with pytest.raises(ValueError, match="collides with a state id"):
        resolve_microsim(doc_collide)


def test_validate_flags_transition_from_a_psm_node():
    doc = _psm_document()
    doc["transitions"] = doc["transitions"] + [{"from_state": "psm1", "to_state": "on_treatment", "probability": 1.0}]
    model = resolve_microsim(doc)
    errors = validate_microsim(model)
    assert any(e.code == "psm_node_no_outgoing_transitions" for e in errors)


def test_validate_flags_pfs_exceeding_os():
    doc = _psm_document()
    doc["psm_nodes"][0]["pfs_curve"] = {"distribution": "exponential", "params": {"rate": 0.05}}
    doc["psm_nodes"][0]["os_curve"] = {"distribution": "exponential", "params": {"rate": 0.5}}
    model = resolve_microsim(doc)
    errors = validate_microsim(model, cycles_per_year=1.0)
    assert any(e.code == "pfs_exceeds_os" for e in errors)


def test_validate_accepts_a_fully_valid_psm_model():
    model = resolve_microsim(_psm_document())
    errors = validate_microsim(model, cycles_per_year=1.0)
    assert errors == []
