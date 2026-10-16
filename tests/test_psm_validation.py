import pytest

from heor_tree_engine import PartitionedSurvivalModel, PSMState, SurvivalCurve, validate_psm


def _states() -> list[PSMState]:
    return [
        PSMState(role="pre_progression", name="Pre", cost=100.0, effect=1.0),
        PSMState(role="post_progression", name="Post", cost=300.0, effect=0.6),
        PSMState(role="dead", name="Dead", cost=0.0, effect=0.0),
    ]


def _model() -> PartitionedSurvivalModel:
    return PartitionedSurvivalModel(
        id="psm1",
        name="test",
        pfs_curve=SurvivalCurve("exponential", {"rate": 0.2}),
        os_curve=SurvivalCurve("exponential", {"rate": 0.1}),
        states=_states(),
        cycles=5,
    )


def test_valid_model_has_no_errors() -> None:
    assert validate_psm(_model()) == []


def test_missing_state_role_is_flagged() -> None:
    model = _model()
    model.states = [state for state in model.states if state.role != "dead"]

    errors = validate_psm(model)

    assert any(error.code == "missing_state" and error.node_id == "dead" for error in errors)


def test_unknown_distribution_is_flagged() -> None:
    model = _model()
    model.pfs_curve = SurvivalCurve("unsupported", {})

    errors = validate_psm(model)

    assert any(error.code == "unknown_distribution" for error in errors)


@pytest.mark.parametrize(
    ("distribution", "params"),
    [
        ("log_normal", {"meanlog": -0.5, "sdlog": 0.8}),
        ("log_logistic", {"shape": 1.5, "scale": 4.0}),
        ("gompertz", {"shape": -0.1, "rate": 0.2}),
        ("generalized_gamma", {"mu": -0.5, "sigma": 0.8, "Q": -0.4}),
    ],
)
def test_additional_survival_distributions_validate(distribution, params) -> None:
    model = _model()
    model.pfs_curve = SurvivalCurve(distribution, params)

    errors = validate_psm(model)

    assert not any(error.code == "unknown_distribution" for error in errors)


@pytest.mark.parametrize(
    ("distribution", "params"),
    [
        ("log_normal", {"meanlog": 0.0, "sdlog": 0.0}),
        ("log_logistic", {"shape": 0.0, "scale": 4.0}),
        ("gompertz", {"shape": 0.0, "rate": 0.2}),
        ("generalized_gamma", {"mu": 0.0, "sigma": 0.0, "Q": 0.4}),
    ],
)
def test_additional_survival_distributions_validate_parameter_domains(distribution, params) -> None:
    model = _model()
    model.pfs_curve = SurvivalCurve(distribution, params)

    errors = validate_psm(model)

    assert any(error.code == "non_positive_curve_param" for error in errors)


def test_generalized_gamma_q_zero_is_valid_and_old_stacy_params_remain_valid() -> None:
    model = _model()
    model.pfs_curve = SurvivalCurve("generalized_gamma", {"mu": 0.0, "sigma": 0.8, "Q": 0.0})
    assert not any(error.code == "non_positive_curve_param" for error in validate_psm(model))

    model.pfs_curve = SurvivalCurve("generalized_gamma", {"a": 2.0, "c": -1.0, "scale": 4.0})
    assert not any(error.code == "unknown_distribution" for error in validate_psm(model))


def test_non_positive_curve_param_is_flagged() -> None:
    model = _model()
    model.os_curve = SurvivalCurve("exponential", {"rate": -0.1})

    errors = validate_psm(model)

    assert any(error.code == "non_positive_curve_param" for error in errors)


def test_negative_cost_requires_explicit_permission() -> None:
    model = _model()
    model.states[0].cost = -10.0

    errors = validate_psm(model)
    assert any(error.code == "negative_cost" for error in errors)

    errors_allowed = validate_psm(model, allow_negative_cost=True)
    assert not any(error.code == "negative_cost" for error in errors_allowed)


def test_invalid_cycles_is_flagged() -> None:
    model = _model()
    model.cycles = 0

    errors = validate_psm(model)

    assert any(error.code == "invalid_cycles" for error in errors)
