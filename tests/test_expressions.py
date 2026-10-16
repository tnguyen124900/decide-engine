import pytest

from heor_tree_engine import (
    Branch,
    ChanceNode,
    Param,
    TerminalNode,
    evaluate_expression,
    expected_cost,
    Engine,
    MarkovModel,
    MarkovState,
    MarkovTransition,
    run_markov,
    resolve_variables,
)


def test_rate_to_probability_variable_can_drive_decision_tree() -> None:
    parameters = {"rate_AE_3yr": Param(name="rate_AE_3yr", base=0.5)}
    variables = {"prob_AE": "RateToProb(rate_AE_3yr, 1, 1/4)"}
    probability = resolve_variables(parameters, variables)["prob_AE"].base

    tree = ChanceNode(
        id="adverse_event",
        branches=[
            Branch(label="event", child=TerminalNode(id="event_outcome", cost=100), probability="prob_AE"),
            Branch(label="no_event", child=TerminalNode(id="no_event_outcome", cost=0), probability="#"),
        ],
    )

    result = expected_cost(tree, parameters, variables)
    assert probability == pytest.approx(0.11750309741540454)
    assert result == pytest.approx(100 * probability)


def test_expression_functions_and_python_operators() -> None:
    parameters = {"rate_AE_3yr": Param(name="rate_AE_3yr", base=0.5)}
    assert evaluate_expression("RateToProb(rate_AE_3yr, 1, 1/4)", parameters) == pytest.approx(
        0.11750309741540454
    )
    assert evaluate_expression("if(1 < 2, 10, 20)", parameters) == 10.0
    assert evaluate_expression("quantile({1, 2, 3, 4}, 0.75)", parameters) == 3.25


def test_variable_definitions_resolve_in_dependency_order() -> None:
    parameters = {"rate": Param(name="rate", base=0.5)}
    resolved = resolve_variables(
        parameters,
        {
            "prob": "RateToProb(rate, 1, 1/4)",
            "bounded": "bound(prob, [0, 1])",
        },
    )
    assert resolved["bounded"].base == pytest.approx(0.11750309741540454)


def test_rate_to_probability_variable_can_drive_markov_transition() -> None:
    model = MarkovModel(
        id="m1",
        name="variable probability",
        states=[
            MarkovState(id="well", name="Well", initial_prob=1.0),
            MarkovState(id="event", name="Event", initial_prob=0.0, is_absorbing=True),
        ],
        transitions=[
            MarkovTransition(from_state="well", to_state="event", probability="prob_AE"),
            MarkovTransition(from_state="well", to_state="well", probability="#"),
        ],
        cycles=1,
    )
    result = run_markov(
        model,
        {"rate_AE_3yr": Param(name="rate_AE_3yr", base=0.5)},
        variables={"prob_AE": "RateToProb(rate_AE_3yr, 1, 1/4)"},
    )
    assert result.errors == []
    assert result.trace[-1]["event"] == pytest.approx(0.11750309741540454)


def test_engine_run_resolves_variables_before_validation() -> None:
    from heor_tree_engine import DecisionNode

    tree = DecisionNode(
        id="strategies",
        branches=[
            Branch(
                label="strategy",
                child=TerminalNode(id="outcome", cost="prob_AE", effect=1.0),
            )
        ],
    )
    result = Engine().run(
        tree,
        {"rate_AE_3yr": Param(name="rate_AE_3yr", base=0.5)},
        variables={"prob_AE": "RateToProb(rate_AE_3yr, 1, 1/4)"},
    )
    assert result.errors == []
    assert result.strategies["strategy"].cost == pytest.approx(0.11750309741540454)


def test_circular_variables_are_rejected() -> None:
    with pytest.raises(ValueError, match="Circular variable"):
        resolve_variables({}, {"a": "b + 1", "b": "a + 1"})


def test_parameter_definition_is_calculated_when_parsing_document() -> None:
    from heor_tree_engine import resolve

    _, parameters = resolve(
        {
            "root_id": "root",
            "parameters": [
                {"name": "rate", "base": 0.5},
                {
                    "name": "prob",
                    "base": 0.0,
                    "definition": "RateToProb(rate, 1, 1/4)",
                },
            ],
            "nodes": [{"id": "root", "type": "terminal", "cost": "prob"}],
        }
    )
    assert parameters["prob"].base == pytest.approx(0.11750309741540454)
