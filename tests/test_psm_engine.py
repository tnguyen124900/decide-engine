import math

import pytest

from heor_tree_engine import PartitionedSurvivalModel, PSMState, SurvivalCurve, run_psm
from heor_tree_engine.des_sampling import inverse_survival, stacy_to_prentice_params, survival


def _model(cycles: int = 3, half_cycle_correction: bool = False) -> PartitionedSurvivalModel:
    return PartitionedSurvivalModel(
        id="psm1",
        name="test",
        pfs_curve=SurvivalCurve("exponential", {"rate": 0.2}),
        os_curve=SurvivalCurve("exponential", {"rate": 0.1}),
        states=[
            PSMState(role="pre_progression", name="Pre", cost=100.0, effect=1.0),
            PSMState(role="post_progression", name="Post", cost=300.0, effect=0.6),
            PSMState(role="dead", name="Dead", cost=0.0, effect=0.0),
        ],
        cycles=cycles,
        half_cycle_correction=half_cycle_correction,
    )


def test_occupancy_matches_closed_form_exponential_survival() -> None:
    result = run_psm(_model())

    assert result.errors == []
    for t in range(4):
        s_pfs = math.exp(-0.2 * t)
        s_os = math.exp(-0.1 * t)
        assert result.trace[t]["pre_progression"] == pytest.approx(s_pfs)
        assert result.trace[t]["post_progression"] == pytest.approx(max(0.0, s_os - s_pfs))
        assert result.trace[t]["dead"] == pytest.approx(1.0 - s_os)


@pytest.mark.parametrize(
    ("distribution", "pfs_params", "os_params", "expected_pfs_at_one"),
    [
        ("exponential", {"rate": 0.2}, {"rate": 0.1}, math.exp(-0.2)),
        (
            "weibull",
            {"shape": 1.5, "scale": 8.0},
            {"shape": 1.5, "scale": 12.0},
            math.exp(-((1.0 / 8.0) ** 1.5)),
        ),
        ("log_normal", {"meanlog": 0.0, "sdlog": 0.5}, {"meanlog": 0.5, "sdlog": 0.5}, 0.5),
        ("log_logistic", {"shape": 2.0, "scale": 4.0}, {"shape": 2.0, "scale": 8.0}, 16.0 / 17.0),
        (
            "gompertz",
            {"shape": 0.2, "rate": 0.2},
            {"shape": 0.2, "rate": 0.1},
            math.exp(-math.expm1(0.2)),
        ),
        (
            "generalized_gamma",
            {"mu": math.log(4.0), "sigma": 0.5, "Q": 1.0},
            {"mu": math.log(8.0), "sigma": 0.5, "Q": 1.0},
            math.exp(-((1.0 / 4.0) ** 2)),
        ),
    ],
)
def test_all_supported_survival_distributions_run(distribution, pfs_params, os_params, expected_pfs_at_one) -> None:
    if distribution == "generalized_gamma":
        pytest.importorskip("scipy")
    model = _model(cycles=3)
    model.pfs_curve = SurvivalCurve(distribution, pfs_params)
    model.os_curve = SurvivalCurve(distribution, os_params)

    result = run_psm(model)

    assert result.errors == []
    assert result.trace[1]["pre_progression"] == pytest.approx(expected_pfs_at_one)
    assert sum(result.trace[1].values()) == pytest.approx(1.0)


def test_prentice_generalized_gamma_q_zero_is_lognormal() -> None:
    params = {"mu": 0.7, "sigma": 0.4, "Q": 0.0}
    time_years = 3.0
    survival_probability = 0.8

    assert survival("generalized_gamma", params, time_years) == pytest.approx(
        survival("log_normal", {"meanlog": params["mu"], "sdlog": params["sigma"]}, time_years)
    )
    assert inverse_survival("generalized_gamma", params, survival_probability) == pytest.approx(
        inverse_survival("log_normal", {"meanlog": params["mu"], "sdlog": params["sigma"]}, survival_probability)
    )


@pytest.mark.parametrize("q", [-0.6, 0.4, 1.0])
def test_prentice_generalized_gamma_matches_scipy_stacy_conversion(q) -> None:
    scipy = pytest.importorskip("scipy.stats")
    params = {"mu": 0.7, "sigma": 0.8, "Q": q}
    a = 1.0 / q**2
    c = q / params["sigma"]
    scale = math.exp(params["mu"] + (2.0 * params["sigma"] / q) * math.log(abs(q)))
    time_years = 2.5

    expected = scipy.gengamma(a, c, scale=scale).sf(time_years)

    assert survival("generalized_gamma", params, time_years) == pytest.approx(expected)


def test_legacy_stacy_generalized_gamma_parameters_remain_supported() -> None:
    scipy = pytest.importorskip("scipy.stats")
    params = {"a": 2.0, "c": 1.0, "scale": 4.0}

    assert survival("generalized_gamma", params, 2.0) == pytest.approx(
        scipy.gengamma(params["a"], params["c"], scale=params["scale"]).sf(2.0)
    )


def test_legacy_stacy_parameters_convert_to_prentice_without_losing_references() -> None:
    from heor_tree_engine import Param, evaluate_expression

    converted = stacy_to_prentice_params({"a": "shape_a", "c": "shape_c", "scale": "time_scale"})
    assert converted is not None
    values = {"shape_a": 2.0, "shape_c": -1.0, "time_scale": 4.0}
    parameters = {name: Param(name=name, base=value) for name, value in values.items()}
    resolved = {name: evaluate_expression(value, parameters) for name, value in converted.items()}
    direct = stacy_to_prentice_params({"a": 2.0, "c": -1.0, "scale": 4.0})

    assert direct is not None
    assert resolved == pytest.approx(direct)


def test_half_cycle_correction_matches_trapezoidal_formula() -> None:
    plain = run_psm(_model())
    corrected = run_psm(_model(half_cycle_correction=True))

    final_occ = corrected.trace[-1]
    final_cost = final_occ["pre_progression"] * 100.0 + final_occ["post_progression"] * 300.0
    final_effect = final_occ["pre_progression"] * 1.0 + final_occ["post_progression"] * 0.6
    expected_cost = plain.total_cost + 0.5 * (final_cost - corrected.cost_per_cycle[0])
    expected_effect = plain.total_effect + 0.5 * (final_effect - corrected.effect_per_cycle[0])

    assert corrected.total_cost == pytest.approx(expected_cost)
    assert corrected.total_effect == pytest.approx(expected_effect)


def test_discounting_reduces_later_cycle_contribution() -> None:
    result = run_psm(_model(), discount_cost_per_cycle=0.1)

    s_pfs1 = math.exp(-0.2 * 1)
    s_os1 = math.exp(-0.1 * 1)
    raw_cost_1 = s_pfs1 * 100.0 + max(0.0, s_os1 - s_pfs1) * 300.0
    assert result.cost_per_cycle[1] == pytest.approx(raw_cost_1 / 1.1)


def test_crossing_curves_are_rejected() -> None:
    model = _model()
    model.pfs_curve = SurvivalCurve("exponential", {"rate": 0.05})  # now decays slower than OS

    result = run_psm(model)

    assert any(error.code == "pfs_exceeds_os" for error in result.errors)
    assert result.total_cost == 0.0
