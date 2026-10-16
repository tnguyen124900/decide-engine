from heor_tree_engine import Branch, DecisionNode, Engine, TerminalNode


def test_engine_does_not_run_with_negative_cost_by_default() -> None:
    tree = DecisionNode(
        id="strategies",
        branches=[
            Branch(
                label="strategy",
                child=TerminalNode(id="outcome", cost=-10.0, effect=1.0),
            )
        ],
    )

    blocked = Engine().run(tree)
    allowed = Engine().run(tree, allow_negative_cost=True)

    assert blocked.strategies == {}
    assert [error.code for error in blocked.errors] == ["negative_cost"]
    assert allowed.strategies["strategy"].cost == -10.0