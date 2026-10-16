"""Tests for the DES engine (des_*.py). Each engine feature is checked
against a closed-form answer, not just "it runs"."""

import math

import pytest

from heor_tree_engine.des_engine import run_des, run_des_document
from heor_tree_engine.des_parser import resolve_des
from heor_tree_engine.des_validation import validate_des


def _doc(nodes, root="des", **extra):
    return {"root_id": root, "nodes": nodes, **extra}


def _run(document, des_id="des", cost_rate=0.0, effect_rate=0.0):
    results = run_des_document(document, cost_rate, effect_rate)
    if document["root_id"] == des_id:
        (result,) = results.values()
        return result
    return results


def _single_event_doc(rate=0.2, horizon=10.0, n=20000, seed=3):
    return _doc([
        {"id": "des", "type": "des", "time_horizon": horizon, "n_patients": n, "random_seed": seed,
         "branches": [{"label": "Alive", "child_id": "alive", "probability": 1}]},
        {"id": "alive", "type": "time", "effect_rate": 1.0,
         "branches": [{"label": "Death", "child_id": "dead",
                       "time_to_event": {"distribution": "exponential", "params": {"rate": rate}}}]},
        {"id": "dead", "type": "terminal"},
    ])


def test_single_exponential_event_matches_closed_form_with_continuous_discounting():
    rate, horizon = 0.2, 10.0
    result = _run(_single_event_doc(rate, horizon), effect_rate=0.03)
    assert not result.errors
    expected_raw = (1 - math.exp(-rate * horizon)) / rate
    delta = math.log(1.03)
    expected_disc = (1 - math.exp(-(rate + delta) * horizon)) / (rate + delta)
    assert result.total_effect_undiscounted == pytest.approx(expected_raw, rel=0.02)
    assert result.total_effect == pytest.approx(expected_disc, rel=0.02)
    # occupancy trace: S(t) = exp(-rate t) at the grid points
    assert result.trace_times[0] == 0.0 and result.trace_times[-1] == horizon
    assert result.trace[0]["alive"] == pytest.approx(1.0)
    assert result.trace[5]["alive"] == pytest.approx(math.exp(-rate * 5), abs=0.015)
    assert result.trace[5]["dead"] == pytest.approx(1 - math.exp(-rate * 5), abs=0.015)


def test_competing_exponential_events_pick_the_shortest_time():
    a, b = 0.3, 0.1
    document = _doc([
        {"id": "des", "type": "des", "time_horizon": 1000, "n_patients": 20000, "random_seed": 7,
         "branches": [{"label": "S", "child_id": "s", "probability": 1}]},
        {"id": "s", "type": "time", "branches": [
            {"label": "A", "child_id": "end_a", "time_to_event": {"distribution": "exponential", "params": {"rate": a}}},
            {"label": "B", "child_id": "end_b", "time_to_event": {"distribution": "exponential", "params": {"rate": b}}},
        ]},
        {"id": "end_a", "type": "terminal"},
        {"id": "end_b", "type": "terminal"},
    ])
    result = _run(document)
    assert result.event_counts["s:A"] == pytest.approx(a / (a + b), abs=0.012)
    assert result.event_counts["s:B"] == pytest.approx(b / (a + b), abs=0.012)
    assert result.time_in_state["s"] == pytest.approx(1 / (a + b), rel=0.02)


def test_time_event_and_startup_rewards_with_discounting_are_exact():
    """TreeAge 43.5/43.13: startup reward at t=0, time reward = rate x time
    spent (continuously discounted), event rewards at the event time."""
    document = _doc([
        {"id": "des", "type": "des", "time_horizon": 50, "n_patients": 1,
         "branches": [{"label": "Well", "child_id": "well", "probability": 1}]},
        {"id": "well", "type": "time", "cost_rate": 1000, "startup_cost": 50, "branches": [
            {"label": "Stroke", "child_id": "dead", "cost_increment": 300,
             "time_to_event": {"distribution": "fixed", "params": {"time": 2}}}]},
        {"id": "dead", "type": "terminal", "cost": 200},
    ])
    result = _run(document, cost_rate=0.03)
    expected = 50 + 1000 * (1 - 1.03 ** -2) / math.log(1.03) + (300 + 200) * 1.03 ** -2
    assert result.total_cost == pytest.approx(expected, rel=1e-12)
    assert result.total_cost_undiscounted == pytest.approx(50 + 2000 + 500)
    assert result.event_counts["well:Stroke"] == 1.0


def test_horizon_censors_time_reward_and_skips_the_event():
    document = _doc([
        {"id": "des", "type": "des", "time_horizon": 5, "n_patients": 1,
         "branches": [{"label": "Well", "child_id": "well", "probability": 1}]},
        {"id": "well", "type": "time", "cost_rate": 10, "branches": [
            {"label": "Late event", "child_id": "dead", "cost_increment": 999,
             "time_to_event": {"distribution": "fixed", "params": {"time": 10}}}]},
        {"id": "dead", "type": "terminal"},
    ])
    result = _run(document)
    assert result.total_cost == pytest.approx(50.0)
    assert result.event_counts["well:Late event"] == 0.0
    assert result.mean_time_simulated == pytest.approx(5.0)


def test_illness_death_model_with_jump_matches_ctmc_sojourn_times():
    a, b, c = 0.2, 0.05, 0.25  # healthy->sick, healthy->dead, sick->dead
    document = _doc([
        {"id": "des", "type": "des", "time_horizon": 2000, "n_patients": 20000, "random_seed": 11,
         "branches": [{"label": "Healthy", "child_id": "healthy", "probability": 1}]},
        {"id": "healthy", "type": "time", "name": "Healthy", "branches": [
            {"label": "Get sick", "child_id": "to_sick",
             "time_to_event": {"distribution": "exponential", "params": {"rate": a}}},
            {"label": "Die", "child_id": "dead",
             "time_to_event": {"distribution": "exponential", "params": {"rate": b}}}]},
        {"id": "to_sick", "type": "terminal", "jump_to": "sick"},
        {"id": "sick", "type": "time", "name": "Sick", "branches": [
            {"label": "Die", "child_id": "dead",
             "time_to_event": {"distribution": "exponential", "params": {"rate": c}}}]},
        {"id": "dead", "type": "terminal"},
    ])
    result = _run(document)
    assert not result.errors
    assert result.time_in_state["healthy"] == pytest.approx(1 / (a + b), rel=0.02)
    assert result.time_in_state["sick"] == pytest.approx(a / (a + b) / c, rel=0.03)


def test_chance_node_after_event_repeating_event_and_tracker_update():
    """Relapse every 1 year; each relapse: 50% back to the same state
    (repeating event via jump), 50% death. Relapses ~ Geometric(0.5): mean 2."""
    document = _doc([
        {"id": "des", "type": "des", "time_horizon": 1000, "n_patients": 20000, "random_seed": 5,
         "trackers": [{"name": "n_relapse", "initial_value": 0}],
         "branches": [{"label": "Remission", "child_id": "rem", "probability": 1}]},
        {"id": "rem", "type": "time", "branches": [
            {"label": "Relapse", "child_id": "outcome", "cost_increment": 100,
             "tracker_updates": {"n_relapse": "n_relapse + 1"},
             "time_to_event": {"distribution": "fixed", "params": {"time": 1}}}]},
        {"id": "outcome", "type": "chance", "branches": [
            {"label": "Recover", "child_id": "back", "probability": 0.5},
            {"label": "Die", "child_id": "dead", "probability": "#"}]},
        {"id": "back", "type": "terminal", "jump_to": "rem"},
        {"id": "dead", "type": "terminal"},
    ])
    result = _run(document)
    assert not result.errors
    assert result.tracker_summary["n_relapse"]["mean"] == pytest.approx(2.0, rel=0.03)
    assert result.event_counts["rem:Relapse"] == pytest.approx(2.0, rel=0.03)
    assert result.total_cost_undiscounted == pytest.approx(200.0, rel=0.03)


def _life_table(q=0.1, max_age=100):
    return [{"name": "life", "columns": ["age", "qx"],
             "data": [{"age": age, "qx": q} for age in range(max_age + 1)]}]


def _mortality_doc(start_age=50, interpolate=True, n=20000):
    return _doc([
        {"id": "des", "type": "des", "time_horizon": 200, "n_patients": n, "random_seed": 9,
         "trackers": [{"name": "startAge", "initial_value": start_age}],
         "event_log_sample_size": 200,
         "branches": [{"label": "Alive", "child_id": "alive", "probability": 1}]},
        {"id": "alive", "type": "time", "effect_rate": 1, "branches": [
            {"label": "Background death", "child_id": "dead",
             "time_to_event": {"distribution": "table_to_time",
                               "params": {"table": "life", "index_column": "age", "value_column": "qx"},
                               "prob_type": "conditional", "interpolate": interpolate,
                               "clock": "forward", "time_offset": "startAge", "sample_once": True}}]},
        {"id": "dead", "type": "terminal"},
    ], tables=_life_table())


def test_table_to_time_life_table_sampled_conditional_on_start_age():
    """TreeAge 43.8 + 43.9: TableToTime from a mortality table, conditional on
    the start age (DistSampleCond), time-to-death = sampled age - age."""
    result = _run(_mortality_doc(start_age=50))
    assert not result.errors
    # Uniform deaths within each year; last table interval closes at 101.
    expected = sum(0.9 ** k * 0.95 for k in range(50)) + 0.9 ** 50 * 0.5
    assert result.total_effect_undiscounted == pytest.approx(expected, rel=0.02)
    death_times = [log[-1].time for log in result.event_log_sample]
    assert min(death_times) >= 0.0 and max(death_times) <= 51.0


def test_table_to_time_without_interpolation_gives_whole_ages():
    result = _run(_mortality_doc(start_age=50, interpolate=False, n=500))
    for log in result.event_log_sample:
        age_at_death = 50 + log[-1].time
        assert age_at_death == pytest.approx(round(age_at_death))


def _weibull_switch_doc(clock, sample_once):
    tte = {"distribution": "weibull", "params": {"shape": 2.0, "scale": 5.0},
           "clock": clock, "sample_once": sample_once}
    return _doc([
        {"id": "des", "type": "des", "time_horizon": 500, "n_patients": 20000, "random_seed": 21,
         "branches": [{"label": "A", "child_id": "a", "probability": 1}]},
        {"id": "a", "type": "time", "branches": [
            {"label": "Switch", "child_id": "to_b", "time_to_event": {"distribution": "fixed", "params": {"time": 1}}},
            {"label": "Death", "child_id": "dead", "time_to_event": tte}]},
        {"id": "to_b", "type": "terminal", "jump_to": "b"},
        {"id": "b", "type": "time", "branches": [{"label": "Death", "child_id": "dead", "time_to_event": tte}]},
        {"id": "dead", "type": "terminal"},
    ])


def test_forward_clock_keeps_one_death_time_across_state_changes_reset_clock_restarts():
    weibull_mean = 5.0 * math.gamma(1 + 1 / 2.0)
    forward = _run(_weibull_switch_doc("forward", False))
    once = _run(_weibull_switch_doc("forward", True))
    reset = _run(_weibull_switch_doc("reset", False))
    assert forward.mean_time_simulated == pytest.approx(weibull_mean, rel=0.02)
    assert once.mean_time_simulated == pytest.approx(weibull_mean, rel=0.02)
    # reset: survivors of year 1 start a fresh Weibull clock in state B
    p_survive_1 = math.exp(-((1 / 5.0) ** 2))
    assert reset.mean_time_simulated > forward.mean_time_simulated + 0.5 * p_survive_1


def test_des_keywords_in_expressions():
    document = _doc([
        {"id": "des", "type": "des", "time_horizon": 50, "n_patients": 1,
         "branches": [{"label": "A", "child_id": "a", "probability": 1}]},
        {"id": "a", "type": "time", "cost_rate": 100, "branches": [
            {"label": "E", "child_id": "end",
             "time_to_event": {"distribution": "expression", "params": {"value": "2 + _time"}}}]},
        {"id": "end", "type": "terminal", "cost": "_total_cost", "effect": "_event_time + _stage_cost / 1000"},
    ])
    result = _run(document)
    assert result.total_cost_undiscounted == pytest.approx(400.0)  # 100*2 + _total_cost(200)
    assert result.total_effect_undiscounted == pytest.approx(2.0 + 0.2)


def test_same_seed_is_reproducible_and_different_seed_differs():
    first = _run(_single_event_doc(n=300, seed=1))
    second = _run(_single_event_doc(n=300, seed=1))
    other = _run(_single_event_doc(n=300, seed=2))
    assert first.patient_effects == second.patient_effects
    assert first.patient_effects != other.patient_effects


def test_decision_root_runs_one_des_per_strategy():
    nodes = [
        {"id": "root", "type": "decision", "branches": [
            {"label": "Standard", "child_id": "des_std"}, {"label": "New drug", "child_id": "des_new"}]},
    ]
    for suffix, rate in (("std", 0.2), ("new", 0.1)):
        nodes += [
            {"id": f"des_{suffix}", "type": "des", "time_horizon": 1000, "n_patients": 5000, "random_seed": 4,
             "branches": [{"label": "Alive", "child_id": f"alive_{suffix}", "probability": 1}]},
            {"id": f"alive_{suffix}", "type": "time", "effect_rate": 1, "branches": [
                {"label": "Death", "child_id": f"dead_{suffix}",
                 "time_to_event": {"distribution": "exponential", "params": {"rate": rate}}}]},
            {"id": f"dead_{suffix}", "type": "terminal"},
        ]
    results = run_des_document(_doc(nodes, root="root"))
    assert set(results) == {"Standard", "New drug"}
    assert results["Standard"].total_effect == pytest.approx(5.0, rel=0.04)
    assert results["New drug"].total_effect == pytest.approx(10.0, rel=0.04)


def test_parameters_and_variables_resolve_inside_des():
    document = _single_event_doc(n=1)
    document["parameters"] = [{"name": "p_cost", "base": 10}]
    document["variables"] = [{"name": "age", "definition": "60 + _time"}]
    document["nodes"][1]["cost_rate"] = "p_cost"
    document["nodes"][1]["branches"][0]["time_to_event"] = {
        "distribution": "expression", "params": {"value": "65 - age"}}
    result = _run(document)
    assert result.total_cost_undiscounted == pytest.approx(50.0)


def _codes(document, des_id="des"):
    model = resolve_des(document, des_id)
    return {error.code for error in validate_des(model, {}, document.get("tables_mapping", {}))}


def test_validation_catches_structural_mistakes():
    document = _doc([
        {"id": "des", "type": "des", "branches": [{"label": "A", "child_id": "a", "probability": 0.7}]},
        {"id": "a", "type": "time", "branches": [
            {"label": "No time", "child_id": "b"},
            {"label": "Bad table", "child_id": "end",
             "time_to_event": {"distribution": "table_to_time",
                               "params": {"table": "missing", "index_column": "age", "value_column": "qx"}}},
        ]},
        {"id": "b", "type": "time"},
        {"id": "end", "type": "terminal", "jump_to": "end"},
    ])
    codes = _codes(document)
    assert {"probabilities_do_not_sum_to_one", "missing_time_to_event", "invalid_event_child",
            "unknown_table", "invalid_jump_target"} <= codes


def test_run_returns_errors_instead_of_raising():
    document = _single_event_doc(n=1)
    document["nodes"][1]["branches"][0]["time_to_event"] = {"distribution": "weibull", "params": {"shape": 1}}
    result = _run(document)
    assert result.errors and result.errors[0].code == "missing_tte_parameter"


def test_table_to_time_table_name_must_not_contain_spaces():
    document = _mortality_doc(n=10)
    document["tables"][0]["name"] = "life table"
    document["nodes"][1]["branches"][0]["time_to_event"]["params"]["table"] = "life table"
    result = _run(document)
    assert result.errors and result.errors[0].code == "invalid_table_name"
    assert "life_table" in result.errors[0].message


def test_des_conditional_sampling_is_the_same_formula_as_disttoprob():
    """Same math -> one formula: P(DES time-to-event <= d | at t) with the
    patient clock equals DistToProb(dist, t, d) (shared conditional_survival)."""
    from heor_tree_engine.distributions import conditional_survival
    from heor_tree_engine.des_sampling import conditional_event_probability, sample_parametric, survival

    params = {"shape": 1.3, "scale": 6.0}
    exact = conditional_event_probability("weibull", params, 2.0, 1.0)
    by_hand = 1 - math.exp(-((3 / 6) ** 1.3)) / math.exp(-((2 / 6) ** 1.3))
    assert exact == pytest.approx(by_hand, rel=1e-12)
    assert exact == pytest.approx(1 - conditional_survival(lambda x: survival("weibull", params, x), 2.0, 1.0))
    import numpy as np
    rng = np.random.default_rng(0)
    draws = [sample_parametric("weibull", params, 1.0 - rng.random(), lower=2.0) - 2.0 for _ in range(40000)]
    assert sum(d <= 1.0 for d in draws) / len(draws) == pytest.approx(exact, abs=0.006)
    try:
        import scipy  # noqa: F401
    except ImportError:
        return
    from heor_tree_engine.distributions import distribution_to_prob
    definition = {"type": "Weibull", "parameters": {"shape": 1.3, "scale": 6.0}}
    assert distribution_to_prob(definition, 2.0, 1.0, float) == pytest.approx(exact, rel=1e-9)


def test_expression_time_to_event_can_use_dist_to_time():
    try:
        import scipy  # noqa: F401
    except ImportError:
        return
    document = _single_event_doc(rate=0.2, n=5000)
    document["distributions"] = [{"name": "death", "type": "Exponential", "parameters": {"rate": 0.2}}]
    document["nodes"][1]["branches"][0]["time_to_event"] = {
        "distribution": "expression", "params": {"value": "DistToTime(death, _time)"}}
    result = _run(document)
    assert not result.errors
    assert result.total_effect_undiscounted == pytest.approx((1 - math.exp(-2)) / 0.2, rel=0.03)
    again = _run(document)
    assert again.patient_effects == result.patient_effects  # reproducible (patient stream)


def test_old_table_prob_name_still_loads_as_table_to_time():
    document = _mortality_doc(n=200)
    document["nodes"][1]["branches"][0]["time_to_event"]["distribution"] = "table_prob"
    renamed = _run(document)
    document["nodes"][1]["branches"][0]["time_to_event"]["distribution"] = "table_to_time"
    assert not renamed.errors
    assert renamed.patient_effects == _run(document).patient_effects


def _life_tables_mapping(q=0.1, max_age=100):
    from heor_tree_engine.expressions import build_tables_mapping
    return build_tables_mapping(_life_table(q, max_age))


def test_table_to_prob_equals_qx_and_matches_table_to_time():
    """TableToProb(table, t1, t2) and TableToTime(table, t1) share one formula:
    P(TableToTime <= t2) = TableToProb; for a q(x) table at a whole age with
    t2 = 1, TableToProb is exactly q(x)."""
    import numpy as np
    from heor_tree_engine import evaluate_expression

    tables = _life_tables_mapping(q=0.1)
    assert evaluate_expression("TableToProb(life, 60, 1)", {}, tables=tables) == pytest.approx(0.1, rel=1e-12)
    # two years: 1 - 0.9^2
    assert evaluate_expression("TableToProb(life, 60, 2)", {}, tables=tables) == pytest.approx(0.19, rel=1e-12)
    # half a year (uniform deaths within the year): 1 - (1 - 0.05)/(1) = 0.05
    assert evaluate_expression("TableToProb(life, 60, 0.5)", {}, tables=tables) == pytest.approx(0.05, rel=1e-12)
    # explicit value column + type give the same
    assert evaluate_expression('TableToProb(life, 60, 1, "qx", "conditional")', {}, tables=tables) == pytest.approx(0.1)
    rng = np.random.default_rng(3)
    draws = [evaluate_expression("TableToTime(life, 60)", {}, {"__rng__": rng}, tables=tables) for _ in range(20000)]
    assert min(draws) >= 0
    assert sum(d <= 2 for d in draws) / len(draws) == pytest.approx(0.19, abs=0.008)


def test_table_to_prob_in_a_markov_style_formula_with_variables():
    from heor_tree_engine import evaluate_expression

    tables = _life_tables_mapping(q=0.1)
    value = evaluate_expression("TableToProb(life, startAge + _cycle, 1)", {}, {"startAge": 60, "_cycle": 3}, tables=tables)
    assert value == pytest.approx(0.1)


def test_des_expression_can_use_table_to_time():
    document = _mortality_doc(start_age=50, n=4000)
    document["nodes"][1]["branches"][0]["time_to_event"] = {
        "distribution": "expression", "params": {"value": "TableToTime(life_table, startAge + _time)"}}
    document["tables"][0]["name"] = "life_table"
    result = _run(document)
    assert not result.errors, result.errors
    expected = sum(0.9 ** k * 0.95 for k in range(50)) + 0.9 ** 50 * 0.5
    assert result.total_effect_undiscounted == pytest.approx(expected, rel=0.03)


def _named_doc(dist_type="Weibull", params=None, n=20000, clock="reset"):
    document = _single_event_doc(n=n, horizon=500)
    document["distributions"] = [{"name": "death_dist", "type": dist_type,
                                  "parameters": params or {"shape": 1.3, "scale": "scale_death"}}]
    document["parameters"] = [{"name": "scale_death", "base": 6.0}]
    document["nodes"][1]["branches"][0]["time_to_event"] = {
        "distribution": "named", "params": {"name": "death_dist"}, "clock": clock}
    return document


def test_event_time_to_event_by_distribution_name():
    """The event stores only the NAME of a Distribution from the
    Distributions section (type + parameters live there, parameters may
    reference Parameters)."""
    result = _run(_named_doc())
    assert not result.errors, result.errors
    assert result.mean_time_simulated == pytest.approx(6 * math.gamma(1 + 1 / 1.3), rel=0.02)
    inline = _single_event_doc(n=20000, horizon=500)
    inline["nodes"][1]["branches"][0]["time_to_event"] = {
        "distribution": "weibull", "params": {"shape": 1.3, "scale": 6.0}}
    assert _run(inline).patient_effects == result.patient_effects  # same draws as the inline form


def test_named_distribution_must_exist_and_be_continuous():
    missing = _named_doc()
    missing["distributions"] = []
    assert _run(missing).errors[0].code == "unknown_distribution_name"
    discrete = _named_doc("Poisson", {"lambda": 2})
    assert _run(discrete).errors[0].code == "unsuitable_distribution"


def test_named_gamma_uses_scipy_fallback_when_available():
    try:
        import scipy  # noqa: F401
    except ImportError:
        return
    result = _run(_named_doc("Gamma", {"shape": 2.0, "rate": 1.0 / 3.0}, n=10000))
    assert not result.errors
    assert result.mean_time_simulated == pytest.approx(6.0, rel=0.03)


def test_named_distribution_missing_parameter_is_a_validation_error():
    document = _named_doc(params={"shape": 1.3})
    assert _run(document).errors[0].code == "missing_distribution_parameter"


def test_extra_cost_effect_outcomes_accumulate_like_the_primary_pair():
    """Outcomes-table extra outcomes (e.g. cost_2, effect_2): time rates,
    startup, branch and terminal event rewards, each discounted with its
    own kind's rate."""
    from heor_tree_engine.des_engine import run_des
    from heor_tree_engine.des_parser import resolve_des

    document = _doc([
        {"id": "des", "type": "des", "time_horizon": 50, "n_patients": 1,
         "branches": [{"label": "Well", "child_id": "well", "probability": 1}]},
        {"id": "well", "type": "time", "cost_rate": 1000, "extra_rates": {"cost_2": 10, "ly": 1},
         "extra_startup": {"cost_2": 3}, "branches": [
            {"label": "Stroke", "child_id": "dead", "extra_outcomes": {"cost_2": 7},
             "time_to_event": {"distribution": "fixed", "params": {"time": 2}}}]},
        {"id": "dead", "type": "terminal", "extra_outcomes": {"cost_2": 5}},
    ])
    model = resolve_des(document, "des")
    result = run_des(model, {}, 0.03, 0.015, outcome_kinds={"cost_2": "cost", "ly": "effect"})
    assert not result.errors
    assert result.extra_totals_undiscounted["cost_2"] == pytest.approx(3 + 20 + 7 + 5)
    assert result.extra_totals_undiscounted["ly"] == pytest.approx(2.0)
    expected_cost2 = 3 + 10 * (1 - 1.03 ** -2) / math.log(1.03) + (7 + 5) * 1.03 ** -2
    assert result.extra_totals["cost_2"] == pytest.approx(expected_cost2, rel=1e-12)
    assert result.extra_totals["ly"] == pytest.approx((1 - 1.015 ** -2) / math.log(1.015), rel=1e-12)


def test_logic_node_routes_by_tracker_condition_and_node_tracker_updates():
    """Relapse every year; a Logic node sends patients to Dead once
    n_relapse >= 3 (else back to Remission). n_relapse is incremented by a
    tracker update on ENTERING the Relapse-counting state path's Terminal.
    Deterministic: exactly 3 relapses, then death at t = 3."""
    document = _doc([
        {"id": "des", "type": "des", "time_horizon": 100, "n_patients": 1,
         "trackers": [{"name": "n_relapse", "initial_value": 0}, {"name": "n_entries", "initial_value": 0}],
         "branches": [{"label": "Remission", "child_id": "rem", "probability": 1}]},
        {"id": "rem", "type": "time", "effect_rate": 1, "tracker_updates": {"n_entries": "n_entries + 1"},
         "branches": [{"label": "Relapse", "child_id": "check", "tracker_updates": {"n_relapse": "n_relapse + 1"},
                       "time_to_event": {"distribution": "fixed", "params": {"time": 1}}}]},
        {"id": "check", "type": "logic", "branches": [
            {"label": "Third relapse", "child_id": "dead", "condition": "n_relapse >= 3"},
            {"label": "Otherwise", "child_id": "back", "condition": "#"}]},
        {"id": "back", "type": "terminal", "jump_to": "rem"},
        {"id": "dead", "type": "terminal", "tracker_updates": {"n_entries": "n_entries + 100"}},
    ])
    result = _run(document)
    assert not result.errors, result.errors
    assert result.mean_time_simulated == pytest.approx(3.0)
    assert result.tracker_summary["n_relapse"]["mean"] == 3
    assert result.tracker_summary["n_entries"]["mean"] == 3 + 100  # 3 state entries + terminal update


def test_logic_node_validation():
    document = _doc([
        {"id": "des", "type": "des", "branches": [{"label": "A", "child_id": "a", "probability": 1}]},
        {"id": "a", "type": "time", "branches": [
            {"label": "E", "child_id": "lg", "time_to_event": {"distribution": "fixed", "params": {"time": 1}}}]},
        {"id": "lg", "type": "logic", "branches": [
            {"label": "x", "child_id": "end"}, {"label": "y", "child_id": "end2", "condition": "#"},
            {"label": "z", "child_id": "end3", "condition": "else"}]},
        {"id": "end", "type": "terminal"}, {"id": "end2", "type": "terminal"}, {"id": "end3", "type": "terminal"},
    ])
    codes = _codes(document)
    assert {"missing_condition", "multiple_else"} <= codes


def test_logic_node_without_true_condition_or_else_reports_error():
    document = _doc([
        {"id": "des", "type": "des", "n_patients": 1, "branches": [{"label": "A", "child_id": "a", "probability": 1}]},
        {"id": "a", "type": "time", "branches": [
            {"label": "E", "child_id": "lg", "time_to_event": {"distribution": "fixed", "params": {"time": 1}}}]},
        {"id": "lg", "type": "logic", "branches": [{"label": "never", "child_id": "end", "condition": "1 > 2"}]},
        {"id": "end", "type": "terminal"},
    ])
    result = _run(document)
    assert result.errors and "no condition is true" in result.errors[0].message


def test_per_patient_resample_gives_patient_characteristics():
    """A Distribution with Resample = per_patient is one value per patient
    (e.g. start age), usable by name -- patient heterogeneity. Uses the
    patient's own stream -> reproducible."""
    try:
        import scipy  # noqa: F401
    except ImportError:
        return
    document = _single_event_doc(n=4000, horizon=500)
    document["distributions"] = [{"name": "age0", "type": "Normal", "parameters": {"mu": 60, "sigma": 8},
                                  "resample": "per_patient"}]
    document["nodes"][0]["trackers"] = [{"name": "startAge", "initial_value": "age0"}]
    result = _run(document)
    assert not result.errors
    assert result.tracker_summary["startAge"]["mean"] == pytest.approx(60, abs=0.5)
    assert result.tracker_summary["startAge"]["p90"] - result.tracker_summary["startAge"]["p10"] > 15
    again = _run(document)
    assert again.tracker_summary == result.tracker_summary


def test_dist_sample_in_a_tracker_uses_the_patient_stream():
    try:
        import scipy  # noqa: F401
    except ImportError:
        return
    document = _single_event_doc(n=300, horizon=500)
    document["distributions"] = [{"name": "age0", "type": "Normal", "parameters": {"mu": 60, "sigma": 8}}]
    document["nodes"][0]["trackers"] = [{"name": "startAge", "initial_value": "DistSample(age0)"}]
    first, second = _run(document), _run(document)
    assert first.tracker_summary == second.tracker_summary


def test_termination_condition_stops_exactly_at_an_age():
    """TreeAge "Termination condition": startAge + _time >= 100 with
    startAge 62 stops every patient exactly at t = 38 (located between
    events, not at the next event), time rewards up to then."""
    document = _doc([
        {"id": "des", "type": "des", "time_horizon": 80, "n_patients": 3,
         "termination_condition": "startAge + _time >= 100",
         "branches": [{"label": "Well", "child_id": "well", "probability": 1}]},
        {"id": "well", "type": "time", "cost_rate": 10, "branches": [
            {"label": "Check-up", "child_id": "back", "cost_increment": 1,
             "time_to_event": {"distribution": "fixed", "params": {"time": 5}}}]},
        {"id": "back", "type": "terminal", "jump_to": "well"},
    ], parameters=[{"name": "startAge", "base": 62}])
    result = _run(document)
    assert not result.errors
    assert result.mean_time_simulated == pytest.approx(38.0, abs=1e-6)
    # 7 check-ups (t = 5..35), then 3 more years in Well before the stop
    assert result.total_cost_undiscounted == pytest.approx(10 * 38 + 7, abs=1e-5)
    assert result.event_counts["well:Terminated"] == pytest.approx(1.0)
    assert result.trace[-1]["Terminated"] == pytest.approx(1.0)
    assert result.trace[37]["well"] == pytest.approx(1.0)


def test_termination_condition_with_competing_death_matches_closed_form():
    rate = 0.2
    document = _single_event_doc(rate, horizon=50, n=20000)
    document["nodes"][0]["termination_condition"] = "_time >= 5"
    result = _run(document)
    assert not result.errors
    assert result.total_effect_undiscounted == pytest.approx((1 - math.exp(-rate * 5)) / rate, rel=0.02)
    assert result.event_counts["alive:Terminated"] == pytest.approx(math.exp(-rate * 5), abs=0.012)
    assert result.trace[10]["dead"] + result.trace[10]["Terminated"] == pytest.approx(1.0)


def test_termination_condition_true_at_start_and_tracker_based():
    document = _doc([
        {"id": "des", "type": "des", "time_horizon": 50, "n_patients": 1,
         "termination_condition": "n_events >= 3",
         "trackers": [{"name": "n_events", "initial_value": 0}],
         "branches": [{"label": "S", "child_id": "s", "probability": 1}]},
        {"id": "s", "type": "time", "effect_rate": 1, "branches": [
            {"label": "E", "child_id": "back", "tracker_updates": {"n_events": "n_events + 1"},
             "time_to_event": {"distribution": "fixed", "params": {"time": 2}}}]},
        {"id": "back", "type": "terminal", "jump_to": "s"},
    ])
    result = _run(document)
    assert result.mean_time_simulated == pytest.approx(6.0)
    assert result.total_effect_undiscounted == pytest.approx(6.0)
    document["nodes"][0]["termination_condition"] = "1"
    assert _run(document).mean_time_simulated == 0.0
    document["nodes"][0]["termination_condition"] = ""
    assert _run(document).mean_time_simulated == pytest.approx(50.0)


def test_total_and_stage_keywords_for_every_outcome_by_symbol_and_id():
    """TreeAge _total_rwdx[i]/_stage_rwdx[i]: every Cost/Effect outcome of
    the Outcomes table gets _total_<symbol>/_stage_<symbol> (plain-name
    symbols) and _total_<id>/_stage_<id>. Here QALY accrues 0.8/yr in
    Well; the stop rule _total_QALY >= 2 fires at exactly t = 2.5."""
    from heor_tree_engine.des_engine import run_des
    from heor_tree_engine.des_parser import resolve_des

    document = _doc([
        {"id": "des", "type": "des", "time_horizon": 50, "n_patients": 1,
         "termination_condition": "_total_QALY >= 2",
         "trackers": [{"name": "stage_q", "initial_value": 0}, {"name": "total_e", "initial_value": 0}],
         "branches": [{"label": "Well", "child_id": "well", "probability": 1}]},
        {"id": "well", "type": "time", "effect_rate": 1, "extra_rates": {"effect_2": 0.8}, "branches": [
            {"label": "Visit", "child_id": "back",
             "tracker_updates": {"stage_q": "_stage_effect_2", "total_e": "_total_E"},
             "time_to_event": {"distribution": "fixed", "params": {"time": 1}}}]},
        {"id": "back", "type": "terminal", "jump_to": "well"},
    ], outcomes=[
        {"id": "cost", "type": "cost", "symbol": "$"},
        {"id": "effect", "type": "effect", "symbol": "E"},
        {"id": "effect_2", "type": "effect", "symbol": "QALY"},
    ])
    model = resolve_des(document, "des")
    assert model.outcome_symbols["effect_2"] == "QALY"
    result = run_des(model, {}, 0.0, 0.0, outcome_kinds={"effect_2": "effect"})
    assert not result.errors
    assert result.mean_time_simulated == pytest.approx(2.5, abs=1e-6)
    assert result.extra_totals_undiscounted["effect_2"] == pytest.approx(2.0, abs=1e-6)
    # at the 2nd visit (t = 2): QALY since entering Well = 0.8, E total = 2
    assert result.tracker_summary["stage_q"]["mean"] == pytest.approx(0.8)
    assert result.tracker_summary["total_e"]["mean"] == pytest.approx(2.0)


def test_patient_rows_give_trial_level_output_consistent_with_totals():
    result = _run(_single_event_doc(n=500), effect_rate=0.03)
    rows = result.patient_rows
    assert len(rows) == 500 and rows[0]["patient"] == 1
    assert sum(r["effect"] for r in rows) / 500 == pytest.approx(result.total_effect, rel=1e-12)
    assert sum(r["effect_undiscounted"] for r in rows) / 500 == pytest.approx(result.total_effect_undiscounted, rel=1e-12)
    assert sum(r["time_simulated"] for r in rows) / 500 == pytest.approx(result.mean_time_simulated, rel=1e-12)
    assert {r["final_state"] for r in rows} <= {"alive", "dead"}
    assert all(r["final_state"] == "dead" for r in rows if r["n_events"] == 1)


def test_calculation_trace_of_one_patient_matches_the_full_run():
    """TreeAge Calculation Trace: patient k traced alone (n_patients = k)
    has the same history as patient k of the full run."""
    import copy
    from heor_tree_engine.des_engine import run_des
    from heor_tree_engine.des_parser import resolve_des

    document = _single_event_doc(n=50, horizon=30)
    full = run_des(resolve_des(document, "des"), {}, 0.0, 0.03)
    small_doc = copy.deepcopy(document)
    small_doc["nodes"][0]["n_patients"] = 7
    traced = run_des(resolve_des(small_doc, "des"), {}, 0.0, 0.03, trace_patient=7)
    steps = [row["step"] for row in traced.calculation_trace]
    assert steps[:4] == ["Patient", "Start", "Enter state", "Sample time"]
    assert "Time rewards" in steps
    assert traced.calculation_trace[-1]["effect (disc.)"] == pytest.approx(full.patient_effects[6], rel=1e-12)
    assert full.calculation_trace == []


def _clone_doc(clone=True):
    nodes = [
        {"id": "root", "type": "decision", "branches": [
            {"label": "SoC", "child_id": "soc"}, {"label": "New", "child_id": "new"}]},
        {"id": "soc", "type": "des", "time_horizon": 40, "n_patients": 3000, "random_seed": 11,
         "branches": [{"label": "Well", "child_id": "well", "probability": 1}]},
        {"id": "well", "type": "time", "effect_rate": 1, "cost_rate": "c_well", "branches": [
            {"label": "Sick", "child_id": "to_sick", "time_to_event": {"distribution": "named", "params": {"name": "prog_soc"}}},
            {"label": "Die", "child_id": "dead_w", "time_to_event": {"distribution": "exponential", "params": {"rate": 0.02}}}]},
        {"id": "to_sick", "type": "terminal", "jump_to": "sick"},
        {"id": "sick", "type": "time", "effect_rate": 0.6, "branches": [
            {"label": "Die", "child_id": "dead_s", "time_to_event": {"distribution": "exponential", "params": {"rate": 0.2}}}]},
        {"id": "dead_w", "type": "terminal", "name": "Dead"},
        {"id": "dead_s", "type": "terminal", "name": "Dead"},
    ]
    if clone:
        nodes.append({"id": "new", "type": "des", "name": "New", "clone_of": "soc",
                      "clone_substitutions": {"prog_soc": "prog_new", "c_well": "c_well_new"}})
    return _doc(nodes, root="root",
                parameters=[{"name": "c_well", "base": 100}, {"name": "c_well_new", "base": 900}],
                distributions=[{"name": "prog_soc", "type": "Exponential", "parameters": {"rate": 0.3}},
                               {"name": "prog_new", "type": "Exponential", "parameters": {"rate": 0.1}}])


def test_des_clone_equals_an_explicit_copy_with_substituted_names():
    """TreeAge clone copy: the clone runs the master's subtree with
    prog_soc -> prog_new and c_well -> c_well_new; results equal a
    hand-made copy (same seed = same patients)."""
    import copy

    from heor_tree_engine.des_clone import expand_des_clones

    cloned = _clone_doc()
    manual = _clone_doc(clone=False)
    master_nodes = [copy.deepcopy(n) for n in manual["nodes"] if n["id"] in ("soc", "well", "to_sick", "sick", "dead_w", "dead_s")]
    rename = {n["id"]: n["id"] + "_x" for n in master_nodes}
    for n in master_nodes:
        n["id"] = rename[n["id"]]
        for b in n.get("branches", []):
            b["child_id"] = rename.get(b["child_id"], b["child_id"])
        if n.get("jump_to"):
            n["jump_to"] = rename[n["jump_to"]]
    master_nodes[0]["id"] = "new"
    master_nodes[1]["cost_rate"] = "c_well_new"
    master_nodes[1]["branches"][0]["time_to_event"]["params"]["name"] = "prog_new"
    manual["nodes"].extend(master_nodes)
    a = run_des_document(cloned, 0.0, 0.0)
    b = run_des_document(manual, 0.0, 0.0)
    assert a["New"].total_cost_undiscounted == pytest.approx(b["New"].total_cost_undiscounted, rel=1e-12)
    assert a["New"].total_effect_undiscounted == pytest.approx(b["New"].total_effect_undiscounted, rel=1e-12)
    assert a["New"].total_cost_undiscounted > 5 * a["SoC"].total_cost_undiscounted
    # substitution touched formulas + distribution names only, not labels
    expanded = {n["id"]: n for n in expand_des_clones(cloned)["nodes"]}
    assert expanded["well__new"]["cost_rate"] == "c_well_new"
    assert expanded["well__new"]["branches"][0]["label"] == "Sick"
    assert expanded["to_sick__new"]["jump_to"] == "sick__new"


def test_des_clone_follows_master_edits_and_reports_bad_masters():
    from heor_tree_engine import Engine, resolve

    document = _clone_doc()
    before = run_des_document(document, 0.0, 0.0)
    document["nodes"][4]["effect_rate"] = 0.1  # edit the master's Sick state only
    after = run_des_document(document, 0.0, 0.0)
    # same patients, Sick now worth 0.1 instead of 0.6 per year -> the
    # clone changes by exactly 0.5 x its mean time in Sick
    for label, sick_id in (("SoC", "sick"), ("New", "sick__new")):
        drop = before[label].total_effect_undiscounted - after[label].total_effect_undiscounted
        assert drop == pytest.approx(0.5 * after[label].time_in_state[sick_id], rel=1e-9)
        assert drop > 0.5
    document["nodes"][-1]["clone_of"] = "missing"
    with pytest.raises(ValueError, match="does not exist"):
        resolve(document)
    document["nodes"][-1]["clone_of"] = "soc"
    root, params = resolve(document)
    result = Engine().run(root, params, distributions={d["name"]: d for d in document["distributions"]})
    assert not getattr(result, "errors", [])


def test_simulated_kaplan_meier_matches_the_input_curve_under_competing_risks():
    """Survival check: KM of one event (other events = censoring) recovers
    the Weibull it was sampled from, even with a competing exponential."""
    from heor_tree_engine.des_engine import run_des
    from heor_tree_engine.des_parser import resolve_des
    from heor_tree_engine.des_engine import input_survival_curve, kaplan_meier

    document = _named_doc(n=20000)
    document["nodes"][1]["branches"].append(
        {"label": "Other", "child_id": "dead",
         "time_to_event": {"distribution": "exponential", "params": {"rate": 0.15}}})
    model = resolve_des(document, "des")
    from heor_tree_engine.des_parser import resolve_des_namespace
    parameters, variables, distributions, tables = resolve_des_namespace(document)
    result = run_des(model, parameters, distributions=distributions, tables=tables)
    times, km = kaplan_meier(result.event_exposures["alive:Death"])
    grid = [1.0, 3.0, 6.0, 9.0]
    expected, note = input_survival_curve(model, "alive:Death", grid, parameters, variables, distributions, tables)
    for value, t in zip(expected, grid):
        assert value == pytest.approx(math.exp(-((t / 6) ** 1.3)), rel=1e-9)
    import bisect
    for t, s_expected in zip(grid, expected):
        s_km = km[bisect.bisect_right(times, t) - 1]
        assert s_km == pytest.approx(s_expected, abs=0.015)
    none_curve, why = input_survival_curve(model, "alive:Nope", grid, parameters, variables, distributions, tables)
    assert none_curve is None
