from heor_tree_engine import Branch, ChanceNode, TerminalNode


def test_three_node_tree_can_be_created_directly() -> None:
    good_outcome = TerminalNode(id="good", cost=100.0, effect=2.0)
    poor_outcome = TerminalNode(id="poor", cost=250.0, effect=1.0)
    tree = ChanceNode(
        id="treatment_outcomes",
        branches=[
            Branch(label="good", child=good_outcome, probability=0.7),
            Branch(label="poor", child=poor_outcome, probability=0.3),
        ],
    )

    assert tree.branches[0].child is good_outcome
    assert tree.branches[1].child is poor_outcome
    assert tree.branches[0].probability == 0.7
    assert good_outcome.cost == 100.0
    assert poor_outcome.effect == 1.0
