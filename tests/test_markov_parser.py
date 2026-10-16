from heor_tree_engine import resolve_markov


def test_markov_parser_reads_tenure_state_costs() -> None:
    model = resolve_markov(
        {
            "states": [
                {
                    "id": "event",
                    "tenure_applied": True,
                    "tenure_count": 3,
                    "initial_cost": "entry_cost",
                    "recurring_cost": 20,
                    "terminal_cost": 50,
                    "initial_prob": 1,
                }
            ],
            "transitions": [{"from_state": "event", "to_state": "event", "probability": 1}],
        }
    )

    state = model.states[0]
    assert state.tenure_applied is True
    assert state.tenure_count == 3
    assert state.initial_cost == "entry_cost"
    assert state.recurring_cost == 20.0
    assert state.terminal_cost == 50.0