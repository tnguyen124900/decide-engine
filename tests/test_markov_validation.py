from heor_tree_engine import MarkovModel, MarkovState, MarkovTransition, validate_markov


def _base_states() -> list[MarkovState]:
    return [
        MarkovState(id="healthy", name="Healthy", cost=100.0, effect=1.0, initial_prob=1.0),
        MarkovState(id="dead", name="Dead", cost=0.0, effect=0.0, initial_prob=0.0, is_absorbing=True),
    ]


def test_valid_model_has_no_errors() -> None:
    model = MarkovModel(
        id="m1",
        name="two state",
        states=_base_states(),
        transitions=[
            MarkovTransition(from_state="healthy", to_state="dead", probability=0.5),
            MarkovTransition(from_state="healthy", to_state="healthy", probability=0.5),
        ],
    )

    assert validate_markov(model) == []


def test_transition_sum_mismatch_is_flagged() -> None:
    model = MarkovModel(
        id="m1",
        name="two state",
        states=_base_states(),
        transitions=[
            MarkovTransition(from_state="healthy", to_state="dead", probability=0.3),
            MarkovTransition(from_state="healthy", to_state="healthy", probability=0.3),
        ],
    )

    errors = validate_markov(model)

    assert any(error.code == "transition_sum" for error in errors)


def test_missing_transitions_flagged_for_non_absorbing_state() -> None:
    states = _base_states()
    states[0].is_absorbing = False
    model = MarkovModel(id="m1", name="two state", states=states, transitions=[])

    errors = validate_markov(model)

    assert any(error.code == "missing_transitions" and error.node_id == "healthy" for error in errors)


def test_absorbing_state_does_not_require_transitions() -> None:
    model = MarkovModel(
        id="m1",
        name="two state",
        states=_base_states(),
        transitions=[
            MarkovTransition(from_state="healthy", to_state="dead", probability=0.5),
            MarkovTransition(from_state="healthy", to_state="healthy", probability=0.5),
        ],
    )

    errors = validate_markov(model)

    assert not any(error.node_id == "dead" for error in errors)


def test_initial_distribution_must_sum_to_one() -> None:
    states = _base_states()
    states[0].initial_prob = 0.3
    states[1].initial_prob = 0.3
    model = MarkovModel(
        id="m1",
        name="two state",
        states=states,
        transitions=[
            MarkovTransition(from_state="healthy", to_state="dead", probability=0.5),
            MarkovTransition(from_state="healthy", to_state="healthy", probability=0.5),
        ],
    )

    errors = validate_markov(model)

    assert any(error.code == "initial_distribution_sum" for error in errors)


def test_unknown_state_reference_is_flagged() -> None:
    model = MarkovModel(
        id="m1",
        name="two state",
        states=_base_states(),
        transitions=[
            MarkovTransition(from_state="healthy", to_state="ghost", probability=1.0),
        ],
    )

    errors = validate_markov(model)

    assert any(error.code == "unknown_state" for error in errors)


def test_duplicate_transition_pair_is_flagged() -> None:
    model = MarkovModel(
        id="m1",
        name="two state",
        states=_base_states(),
        transitions=[
            MarkovTransition(from_state="healthy", to_state="dead", probability=0.5),
            MarkovTransition(from_state="healthy", to_state="dead", probability=0.5),
        ],
    )

    errors = validate_markov(model)

    assert any(error.code == "duplicate_transition" for error in errors)


def test_negative_cost_requires_explicit_permission() -> None:
    states = _base_states()
    states[0].cost = -10.0
    model = MarkovModel(
        id="m1",
        name="two state",
        states=states,
        transitions=[
            MarkovTransition(from_state="healthy", to_state="dead", probability=0.5),
            MarkovTransition(from_state="healthy", to_state="healthy", probability=0.5),
        ],
    )

    errors = validate_markov(model)
    assert any(error.code == "negative_cost" for error in errors)

    errors_allowed = validate_markov(model, allow_negative_cost=True)
    assert not any(error.code == "negative_cost" for error in errors_allowed)
