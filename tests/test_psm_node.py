"""Tests for PSMNode -- the PSM-as-a-Tree-node-type architecture. Mirrors test_engine.py's
plain-assert style."""
import math

import pytest
from psm_editor_support import rename_psm_node

from heor_tree_engine import (
    Branch,
    ChanceNode,
    DecisionNode,
    Param,
    TerminalNode,
    evaluate_decision,
    expected_cost,
    expected_utility,
    resolve,
    validate,
)
from heor_tree_engine.models import PSMNode, SurvivalCurve


def _psm_node(node_id="psm", cycles=10, half_cycle_correction=True):
    return PSMNode(
        id=node_id,
        pfs_curve=SurvivalCurve("weibull", {"shape": 1.5, "scale": 8.0}),
        os_curve=SurvivalCurve("weibull", {"shape": 1.8, "scale": 12.0}),
        cycles=cycles,
        half_cycle_correction=half_cycle_correction,
        branches=[
            Branch(label="Pre-progression", role="pre_progression",
                   child=TerminalNode(id="pre", cost=100.0, effect=1.0)),
            Branch(label="Post-progression", role="post_progression",
                   child=TerminalNode(id="post", cost=200.0, effect=0.6)),
            Branch(label="Dead", role="dead",
                   child=TerminalNode(id="dead", cost=0.0, effect=0.0)),
        ],
    )


def _reference_occupancy(cycles, half_cycle_correction, cycles_per_year=1.0):
    """Recompute the same trapezoidal integration by hand (independent of
    engine.py's implementation) to check against."""
    rate_cost = {"pre_progression": 100.0, "post_progression": 200.0, "dead": 0.0}
    rate_effect = {"pre_progression": 1.0, "post_progression": 0.6, "dead": 0.0}

    def occ(_cycle):
        cycle_time_years = _cycle / cycles_per_year
        s_pfs = math.exp(-((cycle_time_years / 8.0) ** 1.5)) if cycle_time_years > 0 else 1.0
        s_os = math.exp(-((cycle_time_years / 12.0) ** 1.8)) if cycle_time_years > 0 else 1.0
        return {
            "pre_progression": s_pfs,
            "post_progression": max(0.0, s_os - s_pfs),
            "dead": 1.0 - s_os,
        }

    def reward(o):
        return (
            sum(o[r] * rate_cost[r] for r in rate_cost),
            sum(o[r] * rate_effect[r] for r in rate_effect),
        )

    cost_per_cycle = []
    effect_per_cycle = []
    for _cycle in range(cycles):
        c, e = reward(occ(_cycle))
        cost_per_cycle.append(c)
        effect_per_cycle.append(e)
    total_cost = sum(cost_per_cycle)
    total_effect = sum(effect_per_cycle)
    if half_cycle_correction and cycles > 0:
        final_c, final_e = reward(occ(cycles))
        total_cost += 0.5 * (final_c - cost_per_cycle[0])
        total_effect += 0.5 * (final_e - effect_per_cycle[0])
    return total_cost, total_effect


def test_psm_node_matches_hand_computed_occupancy_integration() -> None:
    node = _psm_node()
    expected_c, expected_e = _reference_occupancy(10, True)
    assert expected_cost(node) == pytest.approx(expected_c)
    assert expected_utility(node) == pytest.approx(expected_e)


def test_psm_node_uses_cycle_index_for_curve_time() -> None:
    node = _psm_node(cycles=24)
    node.cycles_per_year = 12.0
    expected_c, expected_e = _reference_occupancy(24, True, cycles_per_year=12.0)

    assert expected_cost(node) == pytest.approx(expected_c)
    assert expected_utility(node) == pytest.approx(expected_e)


@pytest.mark.parametrize(
    ("distribution", "pfs_params", "os_params"),
    [
        ("exponential", {"rate": 0.2}, {"rate": 0.1}),
        ("weibull", {"shape": 1.5, "scale": 8.0}, {"shape": 1.5, "scale": 12.0}),
        ("log_normal", {"meanlog": 0.0, "sdlog": 0.5}, {"meanlog": 0.5, "sdlog": 0.5}),
        ("log_logistic", {"shape": 2.0, "scale": 4.0}, {"shape": 2.0, "scale": 8.0}),
        ("gompertz", {"shape": 0.2, "rate": 0.2}, {"shape": 0.2, "rate": 0.1}),
        (
            "generalized_gamma",
            {"mu": 1.0, "sigma": 0.7, "Q": -0.4},
            {"mu": 1.5, "sigma": 0.7, "Q": -0.4},
        ),
    ],
)
def test_psm_node_editor_distributions_run(distribution, pfs_params, os_params) -> None:
    if distribution == "generalized_gamma":
        pytest.importorskip("scipy")
    node = _psm_node(cycles=3)
    node.pfs_curve = SurvivalCurve(distribution, pfs_params)
    node.os_curve = SurvivalCurve(distribution, os_params)

    assert validate(node) == []
    assert math.isfinite(expected_cost(node))
    assert math.isfinite(expected_utility(node))


def test_psm_node_generalized_gamma_q_zero_matches_lognormal() -> None:
    node = _psm_node(cycles=5)
    node.pfs_curve = SurvivalCurve("generalized_gamma", {"mu": 0.0, "sigma": 0.5, "Q": 0.0})
    node.os_curve = SurvivalCurve("log_normal", {"meanlog": 0.4, "sdlog": 0.5})

    assert validate(node) == []
    assert math.isfinite(expected_cost(node))
    assert math.isfinite(expected_utility(node))


def test_psm_node_legacy_generalized_gamma_params_still_run() -> None:
    node = _psm_node(cycles=5)
    node.pfs_curve = SurvivalCurve("generalized_gamma", {"a": 2.0, "c": 1.0, "scale": 4.0})
    node.os_curve = SurvivalCurve("generalized_gamma", {"a": 2.0, "c": 1.0, "scale": 8.0})

    assert validate(node) == []
    assert math.isfinite(expected_cost(node))
    assert math.isfinite(expected_utility(node))


def test_psm_terminal_names_can_repeat_across_strategies() -> None:
    document = {
        "root_id": "strategies",
        "nodes": [
            {
                "id": "strategies",
                "type": "decision",
                "branches": [
                    {"label": "Strategy 1", "child_id": "pre_1"},
                    {"label": "Strategy 2", "child_id": "pre_2"},
                ],
            },
            {"id": "pre_1", "type": "terminal"},
            {"id": "pre_2", "type": "terminal"},
        ],
    }

    first_id = rename_psm_node(document, "pre_1", "Pre-progression")
    second_id = rename_psm_node(document, "pre_2", "Pre-progression")

    assert first_id == "Pre-progression"
    assert second_id == "Pre-progression_2"
    assert document["nodes"][1]["label"] == document["nodes"][2]["label"] == "Pre-progression"
    assert document["nodes"][0]["branches"][0]["child_id"] == first_id
    assert document["nodes"][0]["branches"][1]["child_id"] == second_id


def test_psm_node_half_cycle_correction_toggle_changes_result() -> None:
    with_correction = _psm_node(half_cycle_correction=True)
    without_correction = _psm_node(half_cycle_correction=False)
    assert expected_cost(with_correction) != expected_cost(without_correction)
    expected_c, _ = _reference_occupancy(10, False)
    assert expected_cost(without_correction) == pytest.approx(expected_c)


def test_psm_node_nests_under_chance_and_decision_like_any_other_node() -> None:
    # This is the whole point of the chosen architecture: a PSM node is
    # just another Node, so it works anywhere a Chance/Terminal would --
    # multiple strategies (top-level Decision branches), multiple PSM
    # sub-models, and nesting all fall out for free.
    tree = DecisionNode(
        id="strategies",
        branches=[
            Branch(label="drug_a", child=_psm_node("psm_a"), cost_increment=50.0),
            Branch(label="drug_b", child=ChanceNode(
                id="chance_after_b",
                branches=[
                    Branch(label="responds", probability=0.5, child=_psm_node("psm_b_responds")),
                    Branch(label="no_response", probability=0.5,
                           child=TerminalNode(id="b_fail", cost=10.0, effect=0.1)),
                ],
            )),
        ],
    )
    result = evaluate_decision(tree)
    expected_a_cost, expected_a_effect = _reference_occupancy(10, True)
    assert result.strategies["drug_a"].cost == pytest.approx(expected_a_cost + 50.0)
    assert result.strategies["drug_a"].effect == pytest.approx(expected_a_effect)
    # drug_b: 0.5 * (psm result) + 0.5 * (10.0, 0.1)
    expected_b_psm_cost, expected_b_psm_effect = _reference_occupancy(10, True)
    assert result.strategies["drug_b"].cost == pytest.approx(0.5 * expected_b_psm_cost + 0.5 * 10.0)
    assert result.strategies["drug_b"].effect == pytest.approx(0.5 * expected_b_psm_effect + 0.5 * 0.1)


def test_psm_node_requires_all_three_roles() -> None:
    node = _psm_node()
    node.branches.pop()  # drop the "dead" branch
    with pytest.raises(ValueError, match="missing branch role"):
        expected_cost(node)


def test_psm_node_rejects_non_terminal_child() -> None:
    node = _psm_node()
    node.branches[0].child = ChanceNode(id="not_allowed", branches=[])
    with pytest.raises(ValueError, match="must lead to a Terminal node"):
        expected_cost(node)


def test_validate_flags_missing_curve_param_and_bad_cycles() -> None:
    node = _psm_node(cycles=0)
    node.pfs_curve = SurvivalCurve("weibull", {"shape": 1.5})  # missing "scale"
    errors = validate(node)
    codes = {e.code for e in errors}
    assert "invalid_cycles" in codes
    assert "missing_curve_param" in codes


@pytest.mark.skip(reason="pfs_exceeds_os is now a non-fatal warning (psm_validation), not a validate() error")
def test_validate_flags_pfs_exceeding_os() -> None:
    node = _psm_node()
    # Make PFS decay much slower than OS so PFS(t) > OS(t) somewhere.
    node.pfs_curve = SurvivalCurve("exponential", {"rate": 0.01})
    node.os_curve = SurvivalCurve("exponential", {"rate": 0.5})
    errors = validate(node)
    codes = {e.code for e in errors}
    assert "pfs_exceeds_os" in codes


def test_validate_flags_duplicate_and_missing_roles() -> None:
    node = _psm_node()
    node.branches[1].role = "pre_progression"  # duplicate role, "dead" now missing
    errors = validate(node)
    codes = {e.code for e in errors}
    assert "duplicate_psm_role" in codes
    assert "missing_psm_role" in codes


def test_psm_node_supports_parameter_references_in_curve_params() -> None:
    node = _psm_node()
    node.pfs_curve = SurvivalCurve("weibull", {"shape": "pfs_shape", "scale": 8.0})
    params = {"pfs_shape": Param(name="pfs_shape", base=1.5)}
    expected_c, expected_e = _reference_occupancy(10, True)
    assert expected_cost(node, params) == pytest.approx(expected_c)
    assert expected_utility(node, params) == pytest.approx(expected_e)


def test_resolve_parses_psm_node_from_flat_document() -> None:
    document = {
        "root_id": "root",
        "nodes": [
            {
                "id": "root",
                "type": "decision",
                "branches": [{"label": "only_strategy", "child_id": "psm"}],
            },
            {
                "id": "psm",
                "type": "psm",
                "cycles": 10,
                "half_cycle_correction": True,
                "pfs_curve": {"distribution": "weibull", "params": {"shape": 1.5, "scale": 8.0}},
                "os_curve": {"distribution": "weibull", "params": {"shape": 1.8, "scale": 12.0}},
                "branches": [
                    {"label": "Pre-progression", "role": "pre_progression", "child_id": "pre"},
                    {"label": "Post-progression", "role": "post_progression", "child_id": "post"},
                    {"label": "Dead", "role": "dead", "child_id": "dead"},
                ],
            },
            {"id": "pre", "type": "terminal", "cost": 100.0, "effect": 1.0},
            {"id": "post", "type": "terminal", "cost": 200.0, "effect": 0.6},
            {"id": "dead", "type": "terminal", "cost": 0.0, "effect": 0.0},
        ],
    }
    tree, parameters = resolve(document)
    result = evaluate_decision(tree, parameters)
    expected_c, expected_e = _reference_occupancy(10, True)
    assert result.strategies["only_strategy"].cost == pytest.approx(expected_c)
    assert result.strategies["only_strategy"].effect == pytest.approx(expected_e)


def test_psm_branch_cost_increment_adds_to_the_per_cycle_rate() -> None:
    # Confirms PSM branches reuse the exact same (cost_increment + child
    # cost) quantity a Chance branch sums -- this is what lets editing a
    # PSM branch's reward reuse Tree's existing edge-editable $=/E= UI
    # with no new form needed (see engine.py's PSMNode case).
    node = _psm_node()
    node.branches[0].cost_increment = 5.0  # pre_progression branch
    node.branches[0].effect_increment = 0.1

    def rate_only_reference():
        rate_cost = {"pre_progression": 105.0, "post_progression": 200.0, "dead": 0.0}
        rate_effect = {"pre_progression": 1.1, "post_progression": 0.6, "dead": 0.0}

        def occ(t):
            s_pfs = math.exp(-((t / 8.0) ** 1.5)) if t > 0 else 1.0
            s_os = math.exp(-((t / 12.0) ** 1.8)) if t > 0 else 1.0
            return {
                "pre_progression": s_pfs,
                "post_progression": max(0.0, s_os - s_pfs),
                "dead": 1.0 - s_os,
            }

        def reward(o):
            return (
                sum(o[r] * rate_cost[r] for r in rate_cost),
                sum(o[r] * rate_effect[r] for r in rate_effect),
            )

        cost_per_cycle, effect_per_cycle = [], []
        for t in range(10):
            c, e = reward(occ(t))
            cost_per_cycle.append(c)
            effect_per_cycle.append(e)
        total_cost = sum(cost_per_cycle)
        total_effect = sum(effect_per_cycle)
        final_c, final_e = reward(occ(10))
        total_cost += 0.5 * (final_c - cost_per_cycle[0])
        total_effect += 0.5 * (final_e - effect_per_cycle[0])
        return total_cost, total_effect

    expected_c, expected_e = rate_only_reference()
    assert expected_cost(node) == pytest.approx(expected_c)
    assert expected_utility(node) == pytest.approx(expected_e)
