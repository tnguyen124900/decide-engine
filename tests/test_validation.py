from heor_tree_engine import (
    Branch,
    ChanceNode,
    DecisionNode,
    Param,
    TerminalNode,
    validate,
)


def error_codes(errors):
    return {error.code for error in errors}


def test_validation_checks_probability_sum_and_decision_probability() -> None:
    chance = ChanceNode(
        id="chance",
        branches=[
            Branch(label="a", child=TerminalNode(id="a_terminal"), probability=0.6),
            Branch(label="b", child=TerminalNode(id="b_terminal"), probability=0.3),
        ],
    )
    decision = DecisionNode(
        id="decision",
        branches=[
            Branch(label="strategy", child=chance, probability=1.0),
        ],
    )

    errors = validate(decision)

    assert "probability_sum" in error_codes(errors)
    assert "decision_probability" in error_codes(errors)


def test_validation_distinguishes_parameter_bound_errors() -> None:
    tree = TerminalNode(id="outcome", cost="cost", effect="utility")
    parameters = {
        "cost": Param(name="cost", base=5.0, low=10.0, high=2.0),
        "utility": Param(name="utility", base=-0.5),
    }

    errors = validate(tree, parameters)
    codes = error_codes(errors)

    assert "invalid_parameter_bounds" in codes
    assert "base_below_low" in codes
    assert "base_above_high" in codes
    assert "negative_cost" not in codes


def test_validation_checks_each_optional_bound_independently() -> None:
    tree = TerminalNode(id="outcome")
    parameters = {
        "below": Param(name="below", base=1.0, low=2.0),
        "above": Param(name="above", base=5.0, high=4.0),
    }

    codes = error_codes(validate(tree, parameters))

    assert "base_below_low" in codes
    assert "base_above_high" in codes


def test_negative_cost_requires_explicit_permission_but_negative_effect_is_valid() -> None:
    tree = TerminalNode(id="outcome", cost=-10.0, effect=-0.2)

    assert "negative_cost" in error_codes(validate(tree))
    assert validate(tree, allow_negative_cost=True) == []