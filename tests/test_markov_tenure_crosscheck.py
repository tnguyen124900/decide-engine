"""Independent hand-computed cross-check of tenure ("tunnel state") handling.

Reference semantics (tracker per stage; stage advances each cycle the cohort
stays; last stage absorbs): trace mass moves stage i -> min(i+1, T) on
staying, and to the exit state at stage 1 on leaving. Cost with tenure:
initial_cost at stage 1, recurring cost (defaults to `cost`) afterwards,
scaled by (1 - exit probability), plus terminal cost * exit probability.
"""
import pytest

from heor_tree_engine import MarkovModel, MarkovState, MarkovTransition, run_markov

STAGE_COST = [10.0, 20.0, 30.0]


def _model(cycles: int) -> MarkovModel:
    return MarkovModel(
        id="m", name="tenure crosscheck",
        states=[
            MarkovState(id="s", name="S", cost="if(_tenure==1,10,if(_tenure==2,20,30))",
                        effect=1.0, initial_prob=1.0, tenure_applied=True, tenure_count=3),
            MarkovState(id="d", name="D", cost=0.0, effect=0.0, initial_prob=0.0, is_absorbing=True),
        ],
        transitions=[
            MarkovTransition(from_state="s", to_state="d", probability="0.1*_tenure"),
            MarkovTransition(from_state="s", to_state="s", probability="#"),
        ],
        cycles=cycles,
    )


def test_tenure_dependent_transition_matches_hand_calculation() -> None:
    cycles = 6
    result = run_markov(_model(cycles))
    assert result.errors == []
    alive = [1.0, 0.0, 0.0]
    for c in range(cycles):
        assert result.trace[c]["s"] == pytest.approx(sum(alive))
        new = [0.0, 0.0, 0.0]
        for i, a in enumerate(alive):
            new[min(i + 1, 2)] += a * (1 - 0.1 * (i + 1))
        alive = new
    assert result.trace[cycles]["s"] == pytest.approx(sum(alive))


def test_tenure_recurring_cost_matches_hand_calculation() -> None:
    cycles = 6
    result = run_markov(_model(cycles))
    alive = [1.0, 0.0, 0.0]
    expected = []
    for _ in range(cycles):
        cost = 0.0  # initial_cost defaults to 0 at stage 1
        for i, a in enumerate(alive):
            if i >= 1:
                cost += a * STAGE_COST[i] * (1 - 0.1 * (i + 1))
        expected.append(cost)
        new = [0.0, 0.0, 0.0]
        for i, a in enumerate(alive):
            new[min(i + 1, 2)] += a * (1 - 0.1 * (i + 1))
        alive = new
    assert result.cost_per_cycle == pytest.approx(expected)
