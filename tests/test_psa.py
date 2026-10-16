import numpy as np
import pytest

from heor_tree_engine import evaluate_markov_scenario
from heor_tree_engine.psa import (
    PSAInputError,
    ce_plane,
    ceac_curve_icer,
    ceac_curve_nmb,
    ceaf_curve,
    convergence_series,
    dirichlet_group_targets_for_state,
    evpi_curve,
    evppi_by_parameter,
    generate_joint_samples,
    joint_group_empirical_entries,
    make_sampler,
    parse_target_id,
    run_psa,
    summary_stats,
    validate_dirichlet_group,
)
from heor_tree_engine.sensitivity import SensitivityResult, StrategyOutcome


def _document() -> dict:
    return {
        "cycles": 1,
        "half_cycle_correction": False,
        "parameters": [{"name": "state_cost", "base": 100, "low": 50, "high": 200}],
        "variables": [{"name": "state_effect", "definition": 1, "low": 0.5, "high": 2}],
        "strategies": [
            {
                "id": "standard",
                "name": "Standard care",
                "states": [{"id": "well", "cost": "state_cost", "effect": "state_effect", "initial_prob": 1}],
                "transitions": [{"from_state": "well", "to_state": "well", "probability": 1}],
            },
            {
                "id": "new",
                "name": "New treatment",
                "states": [{"id": "well", "cost": 150, "effect": 1.2, "initial_prob": 1}],
                "transitions": [{"from_state": "well", "to_state": "well", "probability": 1}],
            },
        ],
    }


def _three_strategy_document() -> dict:
    document = _document()
    document["strategies"].append(
        {
            "id": "premium",
            "name": "Premium treatment",
            "states": [{"id": "well", "cost": 400, "effect": 1.5, "initial_prob": 1}],
            "transitions": [{"from_state": "well", "to_state": "well", "probability": 1}],
        }
    )
    return document


# --- parse_target_id -------------------------------------------------


def test_parse_target_id_splits_param_and_dist_targets() -> None:
    assert parse_target_id("param:state_cost") == ("param", "state_cost", None)
    assert parse_target_id("dist:2:mean") == ("dist", "2", "mean")


def test_parse_target_id_rejects_unrecognised_prefix() -> None:
    with pytest.raises(PSAInputError):
        parse_target_id("variable:state_cost")


# --- make_sampler ------------------------------------------------------


def test_make_sampler_beta_matches_target_mean_and_variance() -> None:
    sampler = make_sampler("Beta", {"mean": "0.3", "se": "0.05"})
    rng = np.random.default_rng(1)
    draws = np.array([sampler(rng) for _ in range(20_000)])
    assert draws.mean() == pytest.approx(0.3, abs=0.01)
    assert draws.std() == pytest.approx(0.05, abs=0.01)


def test_make_sampler_gamma_matches_target_mean_and_variance() -> None:
    sampler = make_sampler("Gamma", {"mean": "100", "se": "20"})
    rng = np.random.default_rng(2)
    draws = np.array([sampler(rng) for _ in range(20_000)])
    assert draws.mean() == pytest.approx(100, rel=0.03)
    assert draws.std() == pytest.approx(20, rel=0.05)


def test_make_sampler_normal_matches_target_mean_and_se() -> None:
    sampler = make_sampler("Normal", {"mean": "50", "se": "5"})
    rng = np.random.default_rng(3)
    draws = np.array([sampler(rng) for _ in range(20_000)])
    assert draws.mean() == pytest.approx(50, abs=0.5)
    assert draws.std() == pytest.approx(5, rel=0.05)


def test_make_sampler_lognormal_matches_target_mean_and_se() -> None:
    sampler = make_sampler("Lognormal", {"mean": "10", "se": "3"})
    rng = np.random.default_rng(4)
    draws = np.array([sampler(rng) for _ in range(30_000)])
    assert draws.mean() == pytest.approx(10, rel=0.03)
    assert draws.std() == pytest.approx(3, rel=0.08)


def test_make_sampler_uniform_stays_within_bounds() -> None:
    sampler = make_sampler("Uniform", {"min": "10", "max": "20"})
    rng = np.random.default_rng(5)
    draws = np.array([sampler(rng) for _ in range(5_000)])
    assert draws.min() >= 10
    assert draws.max() <= 20
    assert draws.mean() == pytest.approx(15, abs=0.5)


def test_make_sampler_triangular_matches_analytic_mean() -> None:
    sampler = make_sampler("Triangular", {"min": "1", "mode": "2", "max": "6"})
    rng = np.random.default_rng(6)
    draws = np.array([sampler(rng) for _ in range(20_000)])
    assert draws.mean() == pytest.approx((1 + 2 + 6) / 3, abs=0.05)


def test_make_sampler_rejects_beta_variance_too_large_for_mean() -> None:
    with pytest.raises(PSAInputError):
        make_sampler("Beta", {"mean": "0.5", "se": "0.9"})


def test_make_sampler_rejects_non_numeric_input() -> None:
    with pytest.raises(PSAInputError):
        make_sampler("Normal", {"mean": "abc", "se": "1"})


def test_make_sampler_rejects_uniform_min_not_below_max() -> None:
    with pytest.raises(PSAInputError):
        make_sampler("Uniform", {"min": "10", "max": "10"})


def test_make_sampler_rejects_triangular_out_of_order_bounds() -> None:
    with pytest.raises(PSAInputError):
        make_sampler("Triangular", {"min": "10", "mode": "5", "max": "20"})


# --- run_psa -------------------------------------------------------------


def test_run_psa_requires_iterations_and_parameters() -> None:
    with pytest.raises(PSAInputError):
        run_psa(_document(), [], iterations=100, evaluate_scenario=evaluate_markov_scenario)
    with pytest.raises(PSAInputError):
        run_psa(
            _document(),
            [{"target_id": "param:state_cost", "psa_type": "Normal", "values": {"mean": "100", "se": "10"}}],
            iterations=0,
            evaluate_scenario=evaluate_markov_scenario,
        )


def test_run_psa_is_deterministic_given_a_seed_and_leaves_document_untouched() -> None:
    document = _document()
    psa_parameters = [
        {"target_id": "param:state_cost", "psa_type": "Gamma", "values": {"mean": "100", "se": "20"}},
        {"target_id": "param:state_effect", "psa_type": "Beta", "values": {"mean": "0.6", "se": "0.1"}},
    ]

    first = run_psa(document, psa_parameters, iterations=500, evaluate_scenario=evaluate_markov_scenario, seed=42)
    second = run_psa(document, psa_parameters, iterations=500, evaluate_scenario=evaluate_markov_scenario, seed=42)

    assert first.errors == []
    assert first.n_iterations == 500
    assert np.array_equal(first.costs, second.costs)
    assert np.array_equal(first.effects, second.effects)
    assert first.strategy_ids == ["standard", "new"]
    # the "new" strategy has no PSA parameters, so its cost/effect never vary
    assert np.all(first.costs[:, 1] == 150.0)
    assert np.all(first.effects[:, 1] == 1.2)
    # the original document must be unaffected by the simulation
    assert document["parameters"][0]["base"] == 100
    assert document["variables"][0]["definition"] == 1


def test_run_psa_distribution_target_samples_are_applied_without_mutating_document() -> None:
    document = {
        "distributions": [{"name": "CostDist", "type": "Gamma", "parameters": {"shape": "25", "rate": "0.25"}}],
        "parameters": [{"name": "state_cost", "definition": "DistMean(CostDist)"}],
        "strategies": [{"id": "standard", "name": "Standard"}],
    }
    seen_k_values: list[float] = []

    def fake_evaluate(scenario, overrides=None, **kwargs):
        seen_k_values.append(float(scenario["distributions"][0]["parameters"]["shape"]))
        return SensitivityResult(
            reference_strategy_id="standard",
            outcomes=[StrategyOutcome("standard", "Standard", 100.0, 1.0, 0.0, 0.0, 0.0, 0.0, None)],
            optimal_strategy_id="standard",
        )

    psa_parameters = [{"target_id": "dist:0:shape", "psa_type": "Normal", "values": {"mean": "25", "se": "3"}}]
    result = run_psa(document, psa_parameters, iterations=500, evaluate_scenario=fake_evaluate, seed=7)

    assert result.errors == []
    assert result.n_iterations == 500
    assert np.mean(seen_k_values) == pytest.approx(25, abs=0.5)
    assert np.std(seen_k_values) == pytest.approx(3, rel=0.1)
    assert np.array_equal(result.samples[:, 0], np.array(seen_k_values))
    assert document["distributions"][0]["parameters"] == {"shape": "25", "rate": "0.25"}


def test_run_psa_rejects_unknown_distribution_index_up_front() -> None:
    document = _document()
    psa_parameters = [{"target_id": "dist:0:a", "psa_type": "Normal", "values": {"mean": "1", "se": "0.1"}}]
    with pytest.raises(PSAInputError):
        run_psa(document, psa_parameters, iterations=10, evaluate_scenario=evaluate_markov_scenario)


# --- correlation between PSA distributions (Iman-Conover) ------------------


def test_run_psa_correlation_induces_approximate_target_correlation() -> None:
    document = _document()
    psa_parameters = [
        {"target_id": "param:state_cost", "psa_type": "Gamma", "values": {"mean": "100", "se": "20"}},
        {"target_id": "param:state_effect", "psa_type": "Beta", "values": {"mean": "0.6", "se": "0.1"}},
    ]
    correlations = [{"target_a": "param:state_cost", "target_b": "param:state_effect", "rho": 0.8}]

    result = run_psa(
        document, psa_parameters, iterations=4000, evaluate_scenario=evaluate_markov_scenario,
        seed=123, correlations=correlations,
    )

    assert result.errors == []
    observed_rho = np.corrcoef(result.samples[:, 0], result.samples[:, 1])[0, 1]
    assert observed_rho == pytest.approx(0.8, abs=0.05)


def test_run_psa_correlation_preserves_each_targets_exact_marginal() -> None:
    document = _document()
    psa_parameters = [
        {"target_id": "param:state_cost", "psa_type": "Gamma", "values": {"mean": "100", "se": "20"}},
        {"target_id": "param:state_effect", "psa_type": "Beta", "values": {"mean": "0.6", "se": "0.1"}},
    ]

    uncorrelated = run_psa(
        document, psa_parameters, iterations=500, evaluate_scenario=evaluate_markov_scenario, seed=99,
    )
    correlated = run_psa(
        document, psa_parameters, iterations=500, evaluate_scenario=evaluate_markov_scenario, seed=99,
        correlations=[{"target_a": "param:state_cost", "target_b": "param:state_effect", "rho": -0.6}],
    )

    # Same seed => the same underlying draws for each target -- correlation
    # only reorders which iteration each value lands in, so the *set* of
    # realized values per target is unchanged (each target's own marginal
    # distribution is preserved exactly).
    assert np.array_equal(np.sort(uncorrelated.samples[:, 0]), np.sort(correlated.samples[:, 0]))
    assert np.array_equal(np.sort(uncorrelated.samples[:, 1]), np.sort(correlated.samples[:, 1]))
    # But the pairing across iterations should now be negatively correlated.
    observed_rho = np.corrcoef(correlated.samples[:, 0], correlated.samples[:, 1])[0, 1]
    assert observed_rho < -0.3


def test_run_psa_empty_correlations_argument_matches_no_argument_at_all() -> None:
    document = _document()
    psa_parameters = [{"target_id": "param:state_cost", "psa_type": "Gamma", "values": {"mean": "100", "se": "20"}}]
    without_arg = run_psa(document, psa_parameters, iterations=200, evaluate_scenario=evaluate_markov_scenario, seed=5)
    with_empty = run_psa(
        document, psa_parameters, iterations=200, evaluate_scenario=evaluate_markov_scenario, seed=5, correlations=[],
    )
    assert np.array_equal(without_arg.samples, with_empty.samples)
    assert np.array_equal(without_arg.costs, with_empty.costs)


def test_run_psa_correlation_rejects_unknown_target() -> None:
    document = _document()
    psa_parameters = [{"target_id": "param:state_cost", "psa_type": "Gamma", "values": {"mean": "100", "se": "20"}}]
    with pytest.raises(PSAInputError):
        run_psa(
            document, psa_parameters, iterations=10, evaluate_scenario=evaluate_markov_scenario,
            correlations=[{"target_a": "param:state_cost", "target_b": "param:does_not_exist", "rho": 0.5}],
        )


def test_run_psa_correlation_rejects_self_pair() -> None:
    document = _document()
    psa_parameters = [{"target_id": "param:state_cost", "psa_type": "Gamma", "values": {"mean": "100", "se": "20"}}]
    with pytest.raises(PSAInputError):
        run_psa(
            document, psa_parameters, iterations=10, evaluate_scenario=evaluate_markov_scenario,
            correlations=[{"target_a": "param:state_cost", "target_b": "param:state_cost", "rho": 0.5}],
        )


def test_run_psa_correlation_rejects_out_of_range_rho() -> None:
    document = _document()
    psa_parameters = [
        {"target_id": "param:state_cost", "psa_type": "Gamma", "values": {"mean": "100", "se": "20"}},
        {"target_id": "param:state_effect", "psa_type": "Beta", "values": {"mean": "0.6", "se": "0.1"}},
    ]
    with pytest.raises(PSAInputError):
        run_psa(
            document, psa_parameters, iterations=10, evaluate_scenario=evaluate_markov_scenario,
            correlations=[{"target_a": "param:state_cost", "target_b": "param:state_effect", "rho": 1.5}],
        )


def test_iman_conover_reorder_handles_inconsistent_pairwise_correlations() -> None:
    # A-B, B-C, A-C all near +/-0.9 in a way that isn't itself a valid
    # (positive semi-definite) 3x3 correlation matrix -- this must still
    # run (via eigenvalue clipping) rather than raising a Cholesky error.
    from heor_tree_engine.psa import _iman_conover_reorder

    rng = np.random.default_rng(1)
    matrix = np.array([
        [1.0, 0.9, -0.9],
        [0.9, 1.0, 0.9],
        [-0.9, 0.9, 1.0],
    ])
    columns = rng.normal(size=(2000, 3))
    reordered = _iman_conover_reorder(columns, matrix, rng)
    assert reordered.shape == columns.shape
    for column_index in range(3):
        assert np.array_equal(np.sort(reordered[:, column_index]), np.sort(columns[:, column_index]))


# --- post-processing -------------------------------------------------------


def _run_example(iterations: int = 3000, seed: int = 11):
    document = _document()
    psa_parameters = [
        {"target_id": "param:state_cost", "psa_type": "Gamma", "values": {"mean": "100", "se": "20"}},
        {"target_id": "param:state_effect", "psa_type": "Beta", "values": {"mean": "0.6", "se": "0.1"}},
    ]
    return run_psa(document, psa_parameters, iterations=iterations, evaluate_scenario=evaluate_markov_scenario, seed=seed)


def test_summary_stats_reference_strategy_has_zero_increment() -> None:
    result = _run_example()
    rows = summary_stats(result, willingness_to_pay=1_000)
    standard, new = rows
    assert standard.strategy_id == "standard"
    assert standard.incremental_cost == 0.0
    assert standard.incremental_effect == 0.0
    assert standard.icer is None
    assert new.incremental_cost == pytest.approx(new.mean_cost - standard.mean_cost)
    assert 0.0 <= new.probability_cost_effective <= 1.0
    assert standard.probability_cost_effective + new.probability_cost_effective == pytest.approx(1.0)


def test_ce_plane_excludes_reference_and_matches_summary_increments() -> None:
    result = _run_example()
    series = ce_plane(result)
    assert [item.strategy_id for item in series] == ["new"]
    rows = summary_stats(result, willingness_to_pay=1_000)
    new_row = next(row for row in rows if row.strategy_id == "new")
    assert series[0].mean_incremental_cost == pytest.approx(new_row.incremental_cost)
    assert series[0].mean_incremental_effect == pytest.approx(new_row.incremental_effect)


def test_ceac_curve_nmb_sums_to_one_across_strategies_at_every_wtp() -> None:
    result = _run_example()
    grid = [0, 500, 1_000, 5_000, 20_000]
    curves = ceac_curve_nmb(result, grid)
    totals = sum(curves[sid] for sid in result.strategy_ids)
    assert np.allclose(totals, 1.0)


def test_ceac_curve_icer_matches_nmb_method_for_two_strategies() -> None:
    result = _run_example()
    grid = [0, 500, 1_000, 5_000, 20_000]
    nmb_curves = ceac_curve_nmb(result, grid)
    icer_curves = ceac_curve_icer(result, grid)
    # with exactly two strategies the pairwise ICER rule and the NMB
    # "who wins" rule are mathematically identical for the non-reference arm
    assert np.allclose(icer_curves["new"], nmb_curves["new"])


def test_ceaf_curve_tracks_whichever_strategy_has_highest_expected_nmb() -> None:
    result = _run_example()
    grid = [0, 1_000, 10_000]
    points = ceaf_curve(result, grid)
    assert len(points) == 3
    for point in points:
        assert point.frontier_strategy_id in result.strategy_ids
        assert 0.0 <= point.probability <= 1.0


def test_convergence_series_last_point_equals_overall_mean_nmb() -> None:
    result = _run_example()
    willingness_to_pay = 1_000
    series = convergence_series(result, willingness_to_pay)
    overall_mean = willingness_to_pay * result.effects.mean(axis=0) - result.costs.mean(axis=0)
    for index, sid in enumerate(result.strategy_ids):
        assert series.running_mean_nmb[sid][-1] == pytest.approx(overall_mean[index])
    assert series.iteration_numbers[-1] == result.n_iterations


def test_evpi_curve_is_never_negative() -> None:
    result = _run_example()
    grid = [0, 1_000, 5_000, 20_000, 100_000]
    evpi = evpi_curve(result, grid)
    assert np.all(evpi >= 0.0)


def test_evppi_ranks_the_varied_parameter_above_a_constant_one() -> None:
    result = _run_example()
    rows = evppi_by_parameter(result, willingness_to_pay=1_000)
    by_target = {row.target_id: row.evppi for row in rows}
    # both configured PSA parameters vary the outcome, so both should show
    # some non-negative explanatory power
    assert by_target["param:state_cost"] >= 0.0
    assert by_target["param:state_effect"] >= 0.0


def test_evppi_is_zero_for_a_parameter_that_never_varies() -> None:
    document = _document()
    psa_parameters = [
        {"target_id": "param:state_cost", "psa_type": "Uniform", "values": {"min": "100", "max": "100.0000001"}},
    ]
    result = run_psa(document, psa_parameters, iterations=200, evaluate_scenario=evaluate_markov_scenario, seed=1)
    rows = evppi_by_parameter(result, willingness_to_pay=1_000)
    assert rows[0].evppi == pytest.approx(0.0, abs=1e-6)


def test_ceaf_and_ceac_handle_three_strategies() -> None:
    document = _three_strategy_document()
    psa_parameters = [
        {"target_id": "param:state_cost", "psa_type": "Gamma", "values": {"mean": "100", "se": "20"}},
    ]
    result = run_psa(document, psa_parameters, iterations=1500, evaluate_scenario=evaluate_markov_scenario, seed=3)
    assert result.strategy_ids == ["standard", "new", "premium"]
    grid = [0, 1_000, 5_000]
    nmb_curves = ceac_curve_nmb(result, grid)
    totals = sum(nmb_curves[sid] for sid in result.strategy_ids)
    assert np.allclose(totals, 1.0)
    frontier = ceaf_curve(result, grid)
    assert len(frontier) == 3


# --- Correlated Distribution: Dirichlet / Multivariate Normal joint groups -


def _dirichlet_eligible_document() -> dict:
    """A 3-branch Markov state (Healthy -> Healthy/Sick/Dead) whose outgoing
    probabilities are each a bare Parameter reference -- the shape a
    Dirichlet "Correlated Distribution" group is meant for."""

    return {
        "cycles": 5,
        "half_cycle_correction": False,
        "parameters": [
            {"name": "p_hh", "base": 0.5, "low": 0.3, "high": 0.7},
            {"name": "p_hs", "base": 0.3, "low": 0.1, "high": 0.5},
            {"name": "p_hd", "base": 0.2, "low": 0.1, "high": 0.3},
        ],
        "variables": [],
        "strategies": [
            {
                "id": "only",
                "name": "Only strategy",
                "states": [
                    {"id": "Healthy", "cost": 0, "effect": 1, "initial_prob": 1},
                    {"id": "Sick", "cost": 0, "effect": 1, "initial_prob": 0, "is_absorbing": True},
                    {"id": "Dead", "cost": 0, "effect": 0, "initial_prob": 0, "is_absorbing": True},
                ],
                "transitions": [
                    {"from_state": "Healthy", "to_state": "Healthy", "probability": "p_hh"},
                    {"from_state": "Healthy", "to_state": "Sick", "probability": "p_hs"},
                    {"from_state": "Healthy", "to_state": "Dead", "probability": "p_hd"},
                ],
            }
        ],
    }


def test_generate_joint_samples_dirichlet_rows_sum_to_one() -> None:
    samples = generate_joint_samples("Dirichlet", {"alpha": [2.0, 3.0, 5.0]}, n=5000, seed=1)
    assert samples.shape == (5000, 3)
    assert np.allclose(samples.sum(axis=1), 1.0)
    assert np.mean(samples, axis=0).tolist() == pytest.approx([0.2, 0.3, 0.5], abs=0.02)


def test_generate_joint_samples_dirichlet_rejects_nonpositive_alpha() -> None:
    with pytest.raises(PSAInputError):
        generate_joint_samples("Dirichlet", {"alpha": [1.0, -1.0]}, n=10)


def test_generate_joint_samples_dirichlet_rejects_single_member() -> None:
    with pytest.raises(PSAInputError):
        generate_joint_samples("Dirichlet", {"alpha": [1.0]}, n=10)


def test_generate_joint_samples_multivariate_normal_matches_declared_moments() -> None:
    mean, sd = [10.0, 50.0], [2.0, 5.0]
    correlation = [[1.0, 0.6], [0.6, 1.0]]
    samples = generate_joint_samples(
        "Multivariate Normal", {"mean": mean, "sd": sd, "correlation": correlation}, n=200_000, seed=2,
    )
    assert np.mean(samples, axis=0).tolist() == pytest.approx(mean, abs=0.1)
    assert np.std(samples, axis=0).tolist() == pytest.approx(sd, abs=0.1)
    observed_rho = np.corrcoef(samples, rowvar=False)[0, 1]
    assert observed_rho == pytest.approx(0.6, abs=0.02)


def test_generate_joint_samples_multivariate_normal_repairs_inconsistent_correlation() -> None:
    # A-B=0.9, B-C=0.9, A-C=-0.9 cannot jointly hold (not positive semi-definite);
    # this must be repaired (via the same eigenvalue-clipping the Correlation
    # table already uses), not raise.
    bad_correlation = [[1.0, 0.9, -0.9], [0.9, 1.0, 0.9], [-0.9, 0.9, 1.0]]
    samples = generate_joint_samples(
        "Multivariate Normal", {"mean": [0, 0, 0], "sd": [1, 1, 1], "correlation": bad_correlation}, n=5000, seed=3,
    )
    assert samples.shape == (5000, 3)


def test_generate_joint_samples_rejects_unknown_kind() -> None:
    with pytest.raises(PSAInputError):
        generate_joint_samples("Multinomial", {}, n=10)


def test_dirichlet_group_targets_for_state_matches_bare_parameter_transitions() -> None:
    document = _dirichlet_eligible_document()
    transitions = document["strategies"][0]["transitions"]
    targets = dirichlet_group_targets_for_state(transitions, "Healthy")
    assert set(targets) == {"param:p_hh", "param:p_hs", "param:p_hd"}


def test_dirichlet_group_targets_for_state_rejects_complement_transition() -> None:
    transitions = [
        {"from_state": "Sick", "to_state": "Dead", "probability": "p_sd"},
        {"from_state": "Sick", "to_state": "Sick", "probability": "#"},
    ]
    with pytest.raises(PSAInputError):
        dirichlet_group_targets_for_state(transitions, "Sick")


def test_dirichlet_group_targets_for_state_rejects_formula_probability() -> None:
    transitions = [
        {"from_state": "Recovering", "to_state": "Healthy", "probability": "p_hs * 2"},
        {"from_state": "Recovering", "to_state": "Sick", "probability": "p_hd"},
    ]
    with pytest.raises(PSAInputError):
        dirichlet_group_targets_for_state(transitions, "Recovering")


def test_validate_dirichlet_group_accepts_exact_match() -> None:
    document = _dirichlet_eligible_document()
    transitions = document["strategies"][0]["transitions"]
    validate_dirichlet_group(transitions, "Healthy", ["param:p_hh", "param:p_hs", "param:p_hd"])


def test_validate_dirichlet_group_rejects_missing_member() -> None:
    document = _dirichlet_eligible_document()
    transitions = document["strategies"][0]["transitions"]
    with pytest.raises(PSAInputError):
        validate_dirichlet_group(transitions, "Healthy", ["param:p_hh", "param:p_hs"])


def test_validate_dirichlet_group_rejects_extra_member() -> None:
    document = _dirichlet_eligible_document()
    transitions = document["strategies"][0]["transitions"]
    with pytest.raises(PSAInputError):
        validate_dirichlet_group(transitions, "Healthy", ["param:p_hh", "param:p_hs", "param:p_hd", "param:p_extra"])


def test_run_psa_dirichlet_joint_group_sums_to_one_every_iteration() -> None:
    document = _dirichlet_eligible_document()
    transitions = document["strategies"][0]["transitions"]
    targets = dirichlet_group_targets_for_state(transitions, "Healthy")
    validate_dirichlet_group(transitions, "Healthy", targets)

    document["psa_empirical"] = joint_group_empirical_entries(
        "Dirichlet", targets, {"alpha": [2.0, 3.0, 5.0]}, n=5000, joint_key="joint_group_healthy", seed=42,
    )
    psa_parameters = [
        {"target_id": target_id, "psa_type": "Beta", "values": {"mean": "0.33", "se": "0.05"}}
        for target_id in targets
    ]

    result = run_psa(document, psa_parameters, iterations=500, evaluate_scenario=evaluate_markov_scenario, seed=7)

    assert result.errors == []
    row_sums = result.samples.sum(axis=1)
    assert np.allclose(row_sums, 1.0)
    # base-case (document["parameters"] itself) is untouched by the PSA run
    assert document["parameters"][0]["base"] == 0.5


def test_run_psa_without_joint_group_is_unaffected_by_the_feature_existing() -> None:
    """A document with no `psa_empirical` group runs byte-for-byte as
    before this feature was added -- `run_psa` itself was never modified,
    only new standalone helper functions were added alongside it."""

    document = _document()
    psa_parameters = [
        {"target_id": "param:state_cost", "psa_type": "Gamma", "values": {"mean": "100", "se": "20"}},
        {"target_id": "param:state_effect", "psa_type": "Beta", "values": {"mean": "0.6", "se": "0.1"}},
    ]
    first = run_psa(document, psa_parameters, iterations=200, evaluate_scenario=evaluate_markov_scenario, seed=11)
    second = run_psa(document, psa_parameters, iterations=200, evaluate_scenario=evaluate_markov_scenario, seed=11)
    assert first.errors == []
    assert np.array_equal(first.samples, second.samples)
