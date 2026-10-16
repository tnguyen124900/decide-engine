import json
from pathlib import Path

from heor_tree_engine.psm_dashboard import (
    build_series, cycle_weights, occupancy_records, state_breakdown, strategy_by_psm_node, survival_overlay_records,
)

SAMPLE = Path(__file__).resolve().parent.parent / "samples" / "psm_cai2021_sample.json"


def _scipy():
    try:
        import scipy  # noqa: F401
        return True
    except ImportError:
        return False


def test_strategy_labels_and_series():
    d = json.loads(SAMPLE.read_text())
    assert strategy_by_psm_node(d) == {"psm_1": ["Pembrolizumab plus Chemo"], "psm_0": ["Chemo only"]}
    if not _scipy():
        return
    series = build_series(d)
    assert [s["label"] for s in series] == ["Pembrolizumab plus Chemo", "Chemo only"]
    n = series[0]["trace"].cycles + 1
    assert len(survival_overlay_records(series)) == 2 * 2 * n
    recs = occupancy_records(series)
    assert len(recs) == 2 * 3 * n
    # occupancy of each cohort sums to 1 at every time point
    by = {}
    for r in recs:
        by.setdefault((r["Strategy"], r["Years"]), 0.0)
        by[(r["Strategy"], r["Years"])] += r["Proportion"]
    assert all(abs(v - 1.0) < 1e-9 for v in by.values())


def test_state_breakdown_ties_out_to_node_totals_even_with_cycle_dependent_rates():
    if not _scipy():
        return
    d = json.loads(SAMPLE.read_text())  # the sample's rates depend on _cycle
    for s in build_series(d):
        t = s["trace"]
        rows = state_breakdown(t)
        assert abs(sum(r["cost_discounted"] for r in rows) - t.total_cost) < 1e-6
        assert abs(sum(r["effect_discounted"] for r in rows) - t.total_effect) < 1e-9
        assert abs(sum(r["cost_undiscounted"] for r in rows) - t.total_cost_undiscounted) < 1e-6


def test_cycle_weights_without_half_cycle():
    d = json.loads(SAMPLE.read_text())
    for n in d["nodes"]:
        if n.get("type") == "psm":
            n["half_cycle_correction"] = False
    if not _scipy():
        return
    t = build_series(d)[0]["trace"]
    w = cycle_weights(t)
    assert w[0] == 1.0 and w[-1] == 0.0 and sum(w) == t.cycles
    assert abs(sum(r["cost_discounted"] for r in state_breakdown(t)) - t.total_cost) < 1e-6
