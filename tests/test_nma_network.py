from heor_tree_engine.nma_network import summarize_network
from heor_tree_engine.nma_parser import resolve_nma


def _arm(treatment: str, n: float = 100) -> dict:
    return {"treatment": treatment, "events": 10, "n": n}


def _model():
    # Placebo-A (x3), Placebo-B (x2), Placebo-C, A-B, A-C, B-D: two closed loops.
    studies = [
        {"id": "S1", "arms": [_arm("Placebo", 120), _arm("A", 118)]},
        {"id": "S2", "arms": [_arm("Placebo", 200), _arm("A", 196)]},
        {"id": "S3", "arms": [_arm("Placebo", 100), _arm("A", 98)]},
        {"id": "S4", "arms": [_arm("Placebo"), _arm("B")]},
        {"id": "S5", "arms": [_arm("Placebo"), _arm("B")]},
        {"id": "S6", "arms": [_arm("Placebo"), _arm("C")]},
        {"id": "S7", "arms": [_arm("A"), _arm("B")]},
        {"id": "S8", "arms": [_arm("A"), _arm("C")]},
        {"id": "S9", "arms": [_arm("B"), _arm("D")]},
    ]
    return resolve_nma({"spec": {"reference": "Placebo"}, "studies": studies})


def test_edges_count_studies_and_patients() -> None:
    summary = summarize_network(_model())
    edge = summary.edges[("A", "Placebo")]

    assert edge.n_studies == 3
    assert edge.study_ids == ["S1", "S2", "S3"]
    assert edge.patients == 120 + 118 + 200 + 196 + 100 + 98
    assert summary.patients["Placebo"] == 120 + 200 + 100 + 300
    assert summary.n_studies["D"] == 1
    assert len(summary.edges) == 6
    assert summary.n_arms == 18


def test_connected_network_with_two_independent_loops() -> None:
    summary = summarize_network(_model())

    assert summary.is_connected
    assert summary.n_independent_loops == 2
    loops = [set(loop.treatments) for loop in summary.loops]
    assert {"Placebo", "A", "B"} in loops
    assert {"Placebo", "A", "C"} in loops
    assert sorted(summary.neighbours("D")) == ["B"]


def test_loop_formed_only_by_a_multi_arm_study_is_marked() -> None:
    model = resolve_nma({"studies": [
        {"id": "S1", "arms": [_arm("P"), _arm("A"), _arm("B")]},
        {"id": "S2", "arms": [_arm("P"), _arm("A")]},
    ]})
    summary = summarize_network(model)

    assert summary.multi_arm_studies == ["S1"]
    assert len(summary.loops) == 1
    # The A-B and P-B edges come only from S1, but P-A also has S2, and S2
    # does not contain B - so the loop is not purely single-study.
    assert summary.loops[0].single_study is False

    only_multi = summarize_network(resolve_nma({"studies": [{"id": "S1", "arms": [_arm("P"), _arm("A"), _arm("B")]}]}))
    assert only_multi.loops[0].single_study is True
    assert only_multi.n_independent_loops == 0


def test_disconnected_components_are_listed() -> None:
    model = resolve_nma({"studies": [
        {"id": "S1", "arms": [_arm("P"), _arm("A")]},
        {"id": "S2", "arms": [_arm("B"), _arm("C")]},
    ]})
    summary = summarize_network(model)

    assert not summary.is_connected
    assert summary.components == [["P", "A"], ["B", "C"]]
