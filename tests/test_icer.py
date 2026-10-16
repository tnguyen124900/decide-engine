from heor_tree_engine import incremental_analysis


def _by_id(rows):
    return {row.strategy_id: row for row in rows}


def test_dominated_strategy_is_not_the_comparator_for_the_next_one():
    # DARTH Sick-Sicker shape: A is strictly dominated by B, AB must be compared with B, not A.
    rows = _by_id(incremental_analysis([
        ("SoC", 109_561.0, 20.43), ("B", 193_092.0, 21.66), ("A", 207_311.0, 21.27), ("AB", 283_092.0, 22.63),
    ]))
    assert rows["SoC"].status == "reference" and rows["SoC"].icer is None
    assert rows["B"].compared_with == "SoC" and abs(rows["B"].icer - 83_531 / 1.23) < 1.0
    assert rows["A"].status == "dominated" and rows["A"].icer is None
    assert rows["AB"].compared_with == "B"
    assert abs(rows["AB"].icer - (283_092.0 - 193_092.0) / (22.63 - 21.66)) < 1e-6


def test_rows_are_ordered_by_cost():
    rows = incremental_analysis([("c", 30.0, 3.0), ("a", 10.0, 1.0), ("b", 20.0, 2.0)])
    assert [row.strategy_id for row in rows] == ["a", "b", "c"]
    assert [row.icer for row in rows] == [None, 10.0, 10.0]


def test_extended_dominance_removes_the_middle_strategy():
    # b: ICER 100 vs a, then c: ICER 20 vs b -> b is extendedly dominated and c is compared with a (120/2 = 60)
    rows = _by_id(incremental_analysis([("a", 0.0, 1.0), ("b", 100.0, 2.0), ("c", 120.0, 3.0)]))
    assert rows["b"].status == "extended_dominated" and rows["b"].icer is None
    assert rows["c"].compared_with == "a" and rows["c"].icer == 60.0


def test_single_and_empty_inputs():
    assert incremental_analysis([]) == []
    only = incremental_analysis([("x", 5.0, 1.0)])
    assert only[0].status == "reference" and only[0].icer is None


def test_strictly_dominating_cheaper_strategy_and_equal_effect():
    rows = _by_id(incremental_analysis([("a", 10.0, 2.0), ("b", 20.0, 2.0), ("c", 30.0, 3.0)]))
    assert rows["b"].status == "dominated"
    assert rows["c"].compared_with == "a" and rows["c"].icer == 20.0
