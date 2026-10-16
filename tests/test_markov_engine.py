import pytest

from heor_tree_engine import MarkovModel, MarkovState, MarkovTransition, run_markov


def _two_state_model(cycles: int = 2, half_cycle_correction: bool = False) -> MarkovModel:
    return MarkovModel(
        id="m1",
        name="two state",
        states=[
            MarkovState(id="healthy", name="Healthy", cost=100.0, effect=1.0, initial_prob=1.0),
            MarkovState(id="dead", name="Dead", cost=0.0, effect=0.0, initial_prob=0.0, is_absorbing=True),
        ],
        transitions=[
            MarkovTransition(from_state="healthy", to_state="dead", probability=0.5),
            MarkovTransition(from_state="healthy", to_state="healthy", probability="#"),
        ],
        cycles=cycles,
        half_cycle_correction=half_cycle_correction,
    )


def test_two_state_cohort_trace_and_totals() -> None:
    result = run_markov(_two_state_model())

    assert result.errors == []
    assert result.trace == [
        {"healthy": 1.0, "dead": 0.0},
        {"healthy": 0.5, "dead": 0.5},
        {"healthy": 0.25, "dead": 0.75},
    ]
    assert result.cost_per_cycle == [100.0, 50.0]
    assert result.effect_per_cycle == [1.0, 0.5]
    assert result.total_cost == 150.0
    assert result.total_effect == 1.5


def test_half_cycle_correction_adjusts_totals() -> None:
    result = run_markov(_two_state_model(half_cycle_correction=True))

    assert result.errors == []
    assert result.total_cost == pytest.approx(112.5)
    assert result.total_effect == pytest.approx(1.125)


def test_discounting_reduces_later_cycle_contribution() -> None:
    result = run_markov(_two_state_model(), discount_cost_per_cycle=0.1)

    assert result.errors == []
    assert result.cost_per_cycle[0] == pytest.approx(100.0)
    assert result.cost_per_cycle[1] == pytest.approx(50.0 / 1.1)
    assert result.total_cost == pytest.approx(100.0 + 50.0 / 1.1)


def test_markov_provides_cycle_and_tenure_variables() -> None:
    model = MarkovModel(
        id="m1",
        name="cycle variables",
        states=[MarkovState(id="healthy", name="Healthy", cost="_cycle + _tenure", initial_prob=1.0)],
        transitions=[MarkovTransition(from_state="healthy", to_state="healthy", probability=1.0)],
        cycles=3,
        half_cycle_correction=False,
    )

    result = run_markov(model)

    assert result.errors == []
    assert result.cost_per_cycle == [1.0, 2.0, 3.0]


def test_tenure_state_applies_initial_recurring_and_terminal_costs() -> None:
    model = MarkovModel(
        id="m1",
        name="tenure costs",
        states=[
            MarkovState(
                id="event",
                name="Event",
                tenure_applied=True,
                tenure_count=2,
                initial_cost=10.0,
                recurring_cost=2.0,
                terminal_cost=100.0,
                initial_prob=1.0,
            ),
            MarkovState(id="dead", name="Dead", is_absorbing=True),
        ],
        transitions=[
            MarkovTransition(from_state="event", to_state="event", probability=0.5),
            MarkovTransition(from_state="event", to_state="dead", probability=0.5),
        ],
        cycles=3,
        half_cycle_correction=False,
    )

    result = run_markov(model)

    assert result.errors == []
    assert result.trace == [
        {"event": 1.0, "dead": 0.0},
        {"event": 0.5, "dead": 0.5},
        {"event": 0.25, "dead": 0.75},
        {"event": 0.125, "dead": 0.875},
    ]
    # Tenure 1: everyone in the state pays Initial (10); leaving in that cycle adds no Terminal.
    # Tenure >= 2: stayers pay Recurring (2), leavers pay Terminal (100): 0.5*2 + 0.5*100 = 51 per unit mass.
    assert result.cost_per_cycle == pytest.approx([10.0, 0.5 * 51.0, 0.25 * 51.0])


def test_markov_resolves_transition_expressions_for_each_cycle() -> None:
    model = MarkovModel(
        id="m1",
        name="cycle transitions",
        states=[
            MarkovState(id="healthy", name="Healthy", initial_prob=1.0),
            MarkovState(id="dead", name="Dead", is_absorbing=True),
        ],
        transitions=[
            MarkovTransition(from_state="healthy", to_state="dead", probability="if(_cycle == 0, 0, 1)"),
            MarkovTransition(from_state="healthy", to_state="healthy", probability="#"),
        ],
        cycles=2,
        half_cycle_correction=False,
    )

    result = run_markov(model)

    assert result.errors == []
    assert result.trace[-1] == {"healthy": 0.0, "dead": 1.0}


def test_absorbing_state_needs_no_explicit_self_loop() -> None:
    model = _two_state_model(cycles=1)
    result = run_markov(model)

    assert result.errors == []
    # dead has no declared transitions but is_absorbing=True, so it should
    # keep 100% of any mass that reaches it rather than raising an error.
    assert result.trace[-1]["dead"] == 0.5


def test_invalid_model_returns_errors_without_running() -> None:
    model = _two_state_model()
    model.transitions = [MarkovTransition(from_state="healthy", to_state="dead", probability=0.3)]

    result = run_markov(model)

    assert result.total_cost == 0.0
    assert any(error.code == "transition_sum" for error in result.errors)
