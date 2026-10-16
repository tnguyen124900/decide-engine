import pytest

from heor_tree_engine import (
    Param,
    resolve_markov,
    resolve_microsim,
    run_markov,
    run_microsim,
    validate_microsim,
)


def _deterministic_document(reset_timing: str) -> dict:
    """A single patient, deterministic (probability-1) transition from A
    to absorbing B, cost = the tracker's own value each cycle. This
    isolates execution order and reset-timing from any
    randomness, so the resulting cost_per_cycle sequence can be asserted
    exactly."""

    return {
        "id": "m1",
        "name": "deterministic reset test",
        "cycles": 4,
        "half_cycle_correction": False,
        "n_patients": 1,
        "random_seed": 1,
        "tracker_reset_timing": reset_timing,
        "states": [
            {"id": "A", "cost": "time_in_state", "effect": 0.0, "initial_prob": 1.0},
            {"id": "B", "cost": "time_in_state", "effect": 0.0, "initial_prob": 0.0, "is_absorbing": True},
        ],
        "transitions": [
            {"from_state": "A", "to_state": "B", "probability": 1.0},
        ],
        "trackers": [
            {
                "name": "time_in_state",
                "initial_value": 0.0,
                "update_formula": "time_in_state + 1",
                "reset_rules": {"B": 0.0},
            }
        ],
    }


def test_execution_order_on_entry_reset_timing() -> None:
    # Cost each cycle reads the Tracker value
    # carried IN (pre-update), and "on_entry" makes the reset take effect
    # starting the very first cycle in the new state.
    model = resolve_microsim(_deterministic_document("on_entry"))
    result = run_microsim(model, {})
    assert not result.errors
    assert result.cost_per_cycle == [0.0, 0.0, 1.0, 2.0]
    assert result.total_cost == pytest.approx(3.0)


def test_execution_order_next_cycle_reset_timing() -> None:
    # "next_cycle": the state's first cycle still uses the normal
    # update_formula result (computed from the OLD state's tracker value),
    # and the reset only takes effect starting the state's second cycle.
    model = resolve_microsim(_deterministic_document("next_cycle"))
    result = run_microsim(model, {})
    assert not result.errors
    assert result.cost_per_cycle == [0.0, 1.0, 0.0, 1.0]
    assert result.total_cost == pytest.approx(2.0)


def test_tracker_with_no_update_formula_is_a_no_op() -> None:
    document = {
        "id": "m1",
        "name": "no-op tracker",
        "cycles": 3,
        "half_cycle_correction": False,
        "n_patients": 1,
        "random_seed": 1,
        "states": [
            {"id": "A", "cost": "held", "effect": 0.0, "initial_prob": 1.0, "is_absorbing": True},
        ],
        "trackers": [{"name": "held", "initial_value": 7.0}],
    }
    model = resolve_microsim(document)
    result = run_microsim(model, {})
    assert not result.errors
    assert result.cost_per_cycle == [7.0, 7.0, 7.0]


def test_absorbing_state_self_loops_and_stops_accruing_extra_cost() -> None:
    document = {
        "id": "m1",
        "name": "absorbing",
        "cycles": 5,
        "half_cycle_correction": False,
        "n_patients": 1,
        "random_seed": 3,
        "states": [
            {"id": "Alive", "cost": 100.0, "effect": 1.0, "initial_prob": 1.0},
            {"id": "Dead", "cost": 0.0, "effect": 0.0, "initial_prob": 0.0, "is_absorbing": True},
        ],
        "transitions": [
            {"from_state": "Alive", "to_state": "Dead", "probability": 1.0},
        ],
    }
    model = resolve_microsim(document)
    result = run_microsim(model, {})
    assert not result.errors
    # Cycle 0 pays Alive's cost; every cycle after the deterministic
    # transition is spent absorbed in Dead (cost 0).
    assert result.cost_per_cycle == [100.0, 0.0, 0.0, 0.0, 0.0]
    assert result.trace[0] == {"Alive": 1.0, "Dead": 0.0}
    assert result.trace[-1] == {"Alive": 0.0, "Dead": 1.0}


def test_same_seed_is_reproducible() -> None:
    document = {
        "id": "m1",
        "name": "repro",
        "cycles": 10,
        "half_cycle_correction": False,
        "n_patients": 200,
        "random_seed": 99,
        "states": [
            {"id": "Healthy", "cost": 50.0, "effect": 1.0, "initial_prob": 1.0},
            {"id": "Dead", "cost": 0.0, "effect": 0.0, "initial_prob": 0.0, "is_absorbing": True},
        ],
        "transitions": [
            {"from_state": "Healthy", "to_state": "Dead", "probability": 0.1},
            {"from_state": "Healthy", "to_state": "Healthy", "probability": "#"},
        ],
    }
    model = resolve_microsim(document)
    result_a = run_microsim(model, {})
    result_b = run_microsim(model, {})
    assert result_a.patient_costs == result_b.patient_costs
    assert result_a.total_cost == result_b.total_cost


def test_matches_exact_markov_cohort_result_within_monte_carlo_tolerance() -> None:
    # Cross-validate the sampling mechanics (transitions, cost/effect
    # accrual, discounting) against the already-trusted exact cohort
    # engine for the same document -- large N should converge closely.
    document = {
        "id": "m1",
        "name": "cross-check",
        "cycles": 20,
        "half_cycle_correction": False,
        "states": [
            {"id": "Healthy", "cost": 100.0, "effect": 1.0, "initial_prob": 1.0},
            {"id": "Sick", "cost": 300.0, "effect": 0.6, "initial_prob": 0.0},
            {"id": "Dead", "cost": 0.0, "effect": 0.0, "initial_prob": 0.0, "is_absorbing": True},
        ],
        "transitions": [
            {"from_state": "Healthy", "to_state": "Sick", "probability": 0.10},
            {"from_state": "Healthy", "to_state": "Dead", "probability": 0.02},
            {"from_state": "Healthy", "to_state": "Healthy", "probability": "#"},
            {"from_state": "Sick", "to_state": "Healthy", "probability": 0.05},
            {"from_state": "Sick", "to_state": "Dead", "probability": 0.05},
            {"from_state": "Sick", "to_state": "Sick", "probability": "#"},
        ],
    }
    markov_model = resolve_markov(document)
    markov_result = run_markov(markov_model, {})
    assert not markov_result.errors

    micro_document = dict(document, n_patients=8000, random_seed=7)
    micro_model = resolve_microsim(micro_document)
    micro_result = run_microsim(micro_model, {})
    assert not micro_result.errors

    assert micro_result.total_cost == pytest.approx(markov_result.total_cost, rel=0.05)
    assert micro_result.total_effect == pytest.approx(markov_result.total_effect, rel=0.02)


def test_tenure_applied_cost_and_half_cycle_correction_match_markov_cohort() -> None:
    document = {
        "id": "m1",
        "name": "tenure+hcc",
        "cycles": 15,
        "half_cycle_correction": True,
        "states": [
            {
                "id": "OnTx",
                "effect": 0.8,
                "initial_prob": 1.0,
                "tenure_applied": True,
                "tenure_count": 3,
                "initial_cost": 500.0,
                "recurring_cost": 50.0,
                "terminal_cost": 20.0,
            },
            {"id": "Off", "cost": 10.0, "effect": 0.5, "initial_prob": 0.0},
            {"id": "Dead", "cost": 0.0, "effect": 0.0, "initial_prob": 0.0, "is_absorbing": True},
        ],
        "transitions": [
            {"from_state": "OnTx", "to_state": "Off", "probability": 0.15},
            {"from_state": "OnTx", "to_state": "Dead", "probability": 0.03},
            {"from_state": "OnTx", "to_state": "OnTx", "probability": "#"},
            {"from_state": "Off", "to_state": "Dead", "probability": 0.04},
            {"from_state": "Off", "to_state": "Off", "probability": "#"},
        ],
    }
    markov_model = resolve_markov(document)
    markov_result = run_markov(markov_model, {}, discount_cost_per_cycle=0.03, discount_effect_per_cycle=0.03)
    assert not markov_result.errors

    micro_document = dict(document, n_patients=10000, random_seed=42)
    micro_model = resolve_microsim(micro_document)
    micro_result = run_microsim(micro_model, {}, discount_cost_per_cycle=0.03, discount_effect_per_cycle=0.03)
    assert not micro_result.errors

    assert micro_result.total_cost == pytest.approx(markov_result.total_cost, rel=0.05)
    assert micro_result.total_effect == pytest.approx(markov_result.total_effect, rel=0.03)


def test_tracker_summary_reports_percentiles_for_included_trackers_only() -> None:
    document = {
        "id": "m1",
        "name": "tracker summary",
        "cycles": 6,
        "half_cycle_correction": False,
        "n_patients": 500,
        "random_seed": 5,
        "states": [
            {"id": "A", "cost": 0.0, "effect": 0.0, "initial_prob": 1.0, "is_absorbing": True},
        ],
        "trackers": [
            {"name": "shown", "initial_value": 0.0, "update_formula": "shown + 1", "include_in_output": True},
            {"name": "hidden", "initial_value": 0.0, "update_formula": "hidden + 1", "include_in_output": False},
        ],
    }
    model = resolve_microsim(document)
    result = run_microsim(model, {})
    assert not result.errors
    assert "shown" in result.tracker_summary
    assert "hidden" not in result.tracker_summary
    # every patient is identical (no randomness involved), so the tracker
    # reaches exactly 6 after 6 cycles for all of them.
    assert result.tracker_summary["shown"]["mean"] == pytest.approx(6.0)
    assert result.tracker_summary["shown"]["p50"] == pytest.approx(6.0)


def test_validation_rejects_fewer_than_one_patient() -> None:
    document = {
        "id": "m1",
        "name": "bad n_patients",
        "cycles": 1,
        "n_patients": 0,
        "states": [{"id": "A", "cost": 0.0, "effect": 0.0, "initial_prob": 1.0, "is_absorbing": True}],
    }
    model = resolve_microsim(document)
    errors = validate_microsim(model)
    assert any(error.code == "invalid_n_patients" for error in errors)


def test_validation_rejects_duplicate_tracker_names() -> None:
    document = {
        "id": "m1",
        "name": "dup tracker",
        "cycles": 1,
        "states": [{"id": "A", "cost": 0.0, "effect": 0.0, "initial_prob": 1.0, "is_absorbing": True}],
        "trackers": [
            {"name": "x", "initial_value": 0.0},
            {"name": "x", "initial_value": 1.0},
        ],
    }
    with pytest.raises(ValueError, match="Duplicate tracker name"):
        resolve_microsim(document)


def test_validation_rejects_reset_rule_for_unknown_state() -> None:
    document = {
        "id": "m1",
        "name": "bad reset target",
        "cycles": 1,
        "states": [{"id": "A", "cost": 0.0, "effect": 0.0, "initial_prob": 1.0, "is_absorbing": True}],
        "trackers": [{"name": "x", "initial_value": 0.0, "reset_rules": {"NoSuchState": 0.0}}],
    }
    model = resolve_microsim(document)
    errors = validate_microsim(model)
    assert any(error.code == "unknown_state" and "reset_rules" in error.message for error in errors)


def test_invalid_reset_timing_is_rejected_at_parse_time() -> None:
    document = {
        "id": "m1",
        "name": "bad reset timing",
        "cycles": 1,
        "tracker_reset_timing": "whenever",
        "states": [{"id": "A", "cost": 0.0, "effect": 0.0, "initial_prob": 1.0, "is_absorbing": True}],
    }
    with pytest.raises(ValueError, match="tracker_reset_timing"):
        resolve_microsim(document)


def test_run_microsim_honors_a_shared_parameters_mapping() -> None:
    document = {
        "id": "m1",
        "name": "params",
        "cycles": 3,
        "half_cycle_correction": False,
        "n_patients": 1,
        "random_seed": 1,
        "states": [{"id": "A", "cost": "unit_cost", "effect": 0.0, "initial_prob": 1.0, "is_absorbing": True}],
    }
    model = resolve_microsim(document)
    result = run_microsim(model, {"unit_cost": Param(name="unit_cost", base=42.0)})
    assert not result.errors
    assert result.cost_per_cycle == [42.0, 42.0, 42.0]


# ---------------------------------------------------------------------------
# Non-real (canvas-child) branch / terminal nodes under a health state.
#
# A terminal/branch node under a health state is NOT a state a patient
# occupies. It means "this cycle ends here, in the real health state this
# node stands for" -- the next cycle the patient moves on again from that
# real state. One-time Incre_$/Incre_E live on the transitions into them.
# Same semantics as the Markov cohort engine (see markov_engine._collapse()).
# ---------------------------------------------------------------------------


def _leaf(leaf_id: str, name: str, parent: str, alias_of: str) -> dict:
    return {
        "id": leaf_id,
        "name": name,
        "initial_prob": 0.0,
        "canvas_kind": "terminal",
        "canvas_parent": parent,
        "canvas_alias_of": alias_of,
    }


def _sick_sicker_nested_document(n_patients: int = 1) -> dict:
    """Sick-Sicker (Krijkamp 2018) drawn the way the DECIDE canvas draws
    it: four real health states directly under M, and every outgoing
    branch a non-real terminal leaf aliased to the real state it lands in.
    One-time Incre_$ on the Sick->Sicker branch and Incre_E on every death."""

    return {
        "id": "sick_sicker",
        "name": "Sick-Sicker nested",
        "cycles": 30,
        "half_cycle_correction": False,
        "n_patients": n_patients,
        "random_seed": 3,
        "states": [
            {"id": "H", "name": "Healthy", "cost": 2000.0, "effect": 1.0, "initial_prob": 1.0},
            {"id": "S1", "name": "Sick", "cost": 4000.0, "effect": 0.75, "initial_prob": 0.0},
            {"id": "S2", "name": "Sicker", "cost": 15000.0, "effect": 0.5, "initial_prob": 0.0},
            {"id": "D", "name": "Dead", "cost": 0.0, "effect": 0.0, "initial_prob": 0.0, "is_absorbing": True},
            _leaf("h_stay", "Healthy", "H", "H"),
            _leaf("h_sick", "Sick", "H", "S1"),
            _leaf("h_dead", "Dead", "H", "D"),
            _leaf("s1_recover", "Healthy", "S1", "H"),
            _leaf("s1_stay", "Sick", "S1", "S1"),
            _leaf("s1_worse", "Sicker", "S1", "S2"),
            _leaf("s1_dead", "Dead", "S1", "D"),
            _leaf("s2_stay", "Sicker", "S2", "S2"),
            _leaf("s2_dead", "Dead", "S2", "D"),
        ],
        "transitions": [
            {"from_state": "H", "to_state": "h_sick", "probability": 0.15},
            {"from_state": "H", "to_state": "h_dead", "probability": 0.005},
            {"from_state": "H", "to_state": "h_stay", "probability": "#"},
            {"from_state": "S1", "to_state": "s1_recover", "probability": 0.5},
            {"from_state": "S1", "to_state": "s1_worse", "probability": 0.105, "cost_increment": 1000.0},
            {"from_state": "S1", "to_state": "s1_dead", "probability": 0.015, "effect_increment": -0.1},
            {"from_state": "S1", "to_state": "s1_stay", "probability": "#"},
            {"from_state": "S2", "to_state": "s2_dead", "probability": 0.05, "effect_increment": -0.1},
            {"from_state": "S2", "to_state": "s2_stay", "probability": "#"},
            {"from_state": "D", "to_state": "D", "probability": 1.0},
        ],
    }


def test_terminal_leaf_ends_the_cycle_in_its_real_state_then_moves_on_next_cycle() -> None:
    # Deterministic, one patient: H --(p=1, one-time $5)--> leaf aliased to
    # S1. Cycle 0 is spent in H (cost 1) plus the branch's one-time 5; the
    # patient then OCCUPIES S1 (the real state) from cycle 1 on, and S1's
    # own dead-end leaf (aliased back to S1) keeps them there paying 10/cycle.
    document = {
        "id": "m1",
        "name": "terminal leaf",
        "cycles": 3,
        "half_cycle_correction": False,
        "n_patients": 1,
        "random_seed": 1,
        "states": [
            {"id": "H", "name": "Healthy", "cost": 1.0, "effect": 1.0, "initial_prob": 1.0},
            {"id": "S1", "name": "Sick", "cost": 10.0, "effect": 0.5, "initial_prob": 0.0},
            _leaf("to_sick", "Sick", "H", "S1"),
            _leaf("stay_sick", "Sick", "S1", "S1"),
        ],
        "transitions": [
            {"from_state": "H", "to_state": "to_sick", "probability": 1.0, "cost_increment": 5.0},
            {"from_state": "S1", "to_state": "stay_sick", "probability": 1.0},
        ],
    }
    model = resolve_microsim(document)
    assert validate_microsim(model, {}) == []
    result = run_microsim(model, {})
    assert not result.errors
    assert result.raw_cost_per_cycle == pytest.approx([6.0, 10.0, 10.0])
    assert result.total_cost == pytest.approx(26.0)
    # Only real states are ever occupied (the leaves never show up).
    assert set(result.trace[0]) == {"H", "S1"}
    assert result.trace[0]["H"] == pytest.approx(1.0)
    assert result.trace[1]["S1"] == pytest.approx(1.0)
    assert result.trace[3]["S1"] == pytest.approx(1.0)


def test_nested_terminal_model_matches_markov_cohort_engine() -> None:
    document = _sick_sicker_nested_document(n_patients=20000)
    markov_result = run_markov(resolve_markov(document), {})
    assert not markov_result.errors

    micro_result = run_microsim(resolve_microsim(document), {})
    assert not micro_result.errors
    assert micro_result.total_cost == pytest.approx(markov_result.total_cost, rel=0.03)
    assert micro_result.total_effect == pytest.approx(markov_result.total_effect, rel=0.02)
    # Occupancy trace agrees with the cohort trace (real states only).
    assert micro_result.trace[10]["D"] == pytest.approx(markov_result.trace[10]["D"], abs=0.01)
    assert micro_result.trace[10]["S2"] == pytest.approx(markov_result.trace[10]["S2"], abs=0.01)


def test_no_missing_transitions_error_for_aliased_dead_end_leaves() -> None:
    model = resolve_microsim(_sick_sicker_nested_document())
    errors = validate_microsim(model, {})
    assert [e for e in errors if e.code in ("missing_transitions", "unresolved_alias")] == []


def test_dead_end_leaf_without_a_real_state_to_stand_for_is_flagged() -> None:
    document = _sick_sicker_nested_document()
    # A leaf that is neither aliased nor named after any real state.
    document["states"].append(
        {"id": "orphan", "name": "Nowhere", "initial_prob": 0.0, "canvas_kind": "terminal", "canvas_parent": "H"}
    )
    document["transitions"].append({"from_state": "H", "to_state": "orphan", "probability": 0.0})
    errors = validate_microsim(resolve_microsim(document), {})
    assert any(e.code == "unresolved_alias" and e.node_id == "orphan" for e in errors)


def test_leaf_resolves_by_name_when_alias_of_is_missing() -> None:
    # Models saved before alias_of existed: a leaf named like a real state
    # still stands for it.
    document = _sick_sicker_nested_document()
    for state in document["states"]:
        state.pop("canvas_alias_of", None)
    model = resolve_microsim(document)
    assert validate_microsim(model, {}) == []
    assert not run_microsim(model, {}).errors


def test_loop_among_branch_nodes_is_rejected() -> None:
    document = _sick_sicker_nested_document()
    document["states"] += [
        {"id": "loop_a", "name": "A", "initial_prob": 0.0, "canvas_kind": "chance", "canvas_parent": "H"},
        {"id": "loop_b", "name": "B", "initial_prob": 0.0, "canvas_kind": "chance", "canvas_parent": "H"},
    ]
    document["transitions"] += [
        {"from_state": "loop_a", "to_state": "loop_b", "probability": 1.0},
        {"from_state": "loop_b", "to_state": "loop_a", "probability": 1.0},
    ]
    errors = validate_microsim(resolve_microsim(document), {})
    assert any(e.code == "non_real_state_loop" for e in errors)


def test_one_time_increments_are_charged_in_the_cycle_the_branch_is_taken() -> None:
    # Everyone takes the $1000 S1->S2 branch's equivalent here with p=1.
    document = {
        "id": "m1",
        "name": "increment timing",
        "cycles": 2,
        "half_cycle_correction": False,
        "n_patients": 1,
        "random_seed": 1,
        "states": [
            {"id": "A", "name": "A", "cost": 0.0, "effect": 0.0, "initial_prob": 1.0},
            {"id": "B", "name": "B", "cost": 0.0, "effect": 0.0, "initial_prob": 0.0, "is_absorbing": True},
            _leaf("to_b", "B", "A", "B"),
        ],
        "transitions": [{"from_state": "A", "to_state": "to_b", "probability": 1.0, "cost_increment": 100.0}],
    }
    result = run_microsim(resolve_microsim(document), {}, discount_cost_per_cycle=0.0)
    assert result.raw_cost_per_cycle == pytest.approx([100.0, 0.0])
