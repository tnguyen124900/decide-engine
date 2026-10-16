from heor_tree_engine import Branch, ChanceNode, TerminalNode, expected_cost, validate


def test_hash_probability_uses_remaining_probability() -> None:
    tree = ChanceNode(
        id="outcomes",
        branches=[
            Branch(
                label="common",
                child=TerminalNode(id="common_outcome", cost=100.0, effect=1.0),
                probability=0.7,
            ),
            Branch(
                label="rare",
                child=TerminalNode(id="rare_outcome", cost=300.0, effect=0.5),
                probability="#",
            ),
        ],
    )

    assert expected_cost(tree) == 160.0
    assert validate(tree) == []
