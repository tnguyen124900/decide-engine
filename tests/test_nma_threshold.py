"""Threshold analysis (B2): hand-computed triangle, NMB basis, and a property
test that shifting a data point just inside / beyond its limit flips the
best treatment exactly as predicted by refitting the frequentist NMA."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from heor_tree_engine.nma_freq import run_nma_frequentist  # noqa: E402
from heor_tree_engine.nma_parser import resolve_nma  # noqa: E402
from heor_tree_engine.nma_threshold import DecisionInputs, run_threshold_analysis  # noqa: E402


def _network(contrasts, small="bad"):
    """contrasts: list of (study, treatment, versus, estimate, se); MD, reference A."""
    studies = {}
    for study, t, v, est, se in contrasts:
        studies.setdefault(study, []).append({"treatment": t, "versus": v, "estimate": est, "se": se})
    names = sorted({c[1] for c in contrasts} | {c[2] for c in contrasts})
    return resolve_nma({
        "spec": {"outcome": "continuous", "measure": "MD", "reference": "A", "small_values": small},
        "treatments": [{"name": n} for n in names],
        "studies": [{"id": s, "contrasts": c} for s, c in studies.items()]})


def _triangle(yab=1.0, yac=2.0, ybc=0.5):
    return _network([("s1", "B", "A", yab, 1.0), ("s2", "C", "A", yac, 1.0), ("s3", "C", "B", ybc, 1.0)])


def _fixed_effects(model):
    fit = run_nma_frequentist(model).fixed
    return {e.treatment: e.estimate for e in fit.effects}


def test_triangle_hand_calculation():
    # H = 1/3 [[2, 1, -1], [1, 2, 1]]; d_hat = (3.5/3, 5.5/3), best = C
    out = run_threshold_analysis(_triangle(), effects="fixed")
    assert not out.errors
    assert out.best == "C"
    assert out.order == ["C", "B", "A"]
    row = {r.study: r for r in out.rows}["s1"]  # column (2/3, 1/3)
    assert abs(row.upper_delta - 2.0) < 1e-9 and row.upper_new_best == "B"
    assert abs(row.lower_delta - (-5.5)) < 1e-9 and row.lower_new_best == "A"
    assert abs(row.upper_value - 3.0) < 1e-9 and abs(row.lower_value - (-4.5)) < 1e-9
    row = {r.study: r for r in out.rows}["s2"]  # column (1/3, 2/3): C moves up, never overtaken by A; B overtakes downward
    assert row.upper_delta is None
    assert abs(row.lower_delta - (-2.0)) < 1e-9 and row.lower_new_best == "B"
    row = {r.study: r for r in out.rows}["s3"]  # column (-1/3, 1/3)
    assert abs(row.lower_delta - (-1.0)) < 1e-9 and row.lower_new_best == "B"


def test_small_values_good_reverses_the_decision():
    out = run_threshold_analysis(_network([("s1", "B", "A", 1.0, 1.0), ("s2", "C", "A", 2.0, 1.0),
                                           ("s3", "C", "B", 0.5, 1.0)], small="good"), effects="fixed")
    assert out.best == "A"  # smaller is better and all effects are positive vs A


def test_nmb_basis_hand_calculation():
    # lambda = 1, E_k = effect estimate, C_C = 0.5 and d_effect = 1 for both active treatments
    values = {"A": {"cost": 0, "effect": 0}, "B": {"cost": 0, "effect": 3.5 / 3, "d_effect": 1.0},
              "C": {"cost": 0.5, "effect": 5.5 / 3, "d_effect": 1.0, "d_cost": 0.0}}
    out = run_threshold_analysis(_triangle(), effects="fixed",
                                 decision=DecisionInputs(basis="nmb", wtp=1.0, values=values))
    assert out.best == "C"
    assert abs(out.benefits["C"] - (5.5 / 3 - 0.5)) < 1e-9
    # NMB_B = 3.5/3, NMB_C = 5.5/3 - 0.5 = 1.3333; gap 0.16667; slope(B-C) for s1 = 1/3 -> delta = 0.5
    row = {r.study: r for r in out.rows}["s1"]
    assert abs(row.upper_delta - 0.5) < 1e-9 and row.upper_new_best == "B"
    # A: NMB 0, gap 4/3 ; slope = -(1/3) -> delta = -4.0
    assert abs(row.lower_delta - (-4.0)) < 1e-9 and row.lower_new_best == "A"
    assert out.icer and any(item["treatment"] == "C" for item in out.icer)


def test_limits_flip_the_decision_in_a_refit():
    rng = np.random.default_rng(7)
    checked = 0
    for _ in range(6):
        names = ["A", "B", "C", "D"]
        contrasts = []
        for i in range(7):
            v, t = rng.choice(4, size=2, replace=False)
            if names[t] == "A" and False:
                continue
            contrasts.append([f"s{i}", names[t], names[v], float(rng.normal(0.5, 1.0)), float(rng.uniform(0.3, 0.8))])
        for i, t in enumerate(["B", "C", "D"]):  # connect the network
            contrasts.append([f"c{i}", t, "A", float(rng.normal(0.5, 1.0)), 0.5])
        model = _network([tuple(c) for c in contrasts])
        out = run_threshold_analysis(model, effects="fixed")
        for index, row in enumerate(out.rows[: len(contrasts)]):
            for delta, new_best in ((row.upper_delta, row.upper_new_best), (row.lower_delta, row.lower_new_best)):
                if delta is None:
                    continue
                for factor, flipped in ((0.98, False), (1.02, True)):
                    changed = [list(c) for c in contrasts]
                    # rows appear in study order, one contrast each: map by study id
                    target = next(c for c in changed if c[0] == row.study)
                    target[3] += factor * delta
                    est = _fixed_effects(_network([tuple(c) for c in changed]))
                    est["A"] = 0.0
                    best_now = max(est, key=est.get)
                    if flipped:
                        assert best_now == new_best, (row.study, delta, best_now, new_best)
                    else:
                        assert best_now == out.best
                    checked += 1
    assert checked > 20


def test_threshold_reaches_the_frame_and_the_report():
    import io
    import zipfile

    import nma_editor_support as sup
    import nma_report
    from heor_tree_engine.nma_network import summarize_network

    document = sup.default_document()
    model = resolve_nma(document)
    result = run_nma_frequentist(model)
    out = run_threshold_analysis(model, "random", None, result)
    assert not out.errors and out.rows and out.network_rows
    frame = sup.threshold_frame(out)
    assert list(frame.columns)[:2] == ["Study", "Comparison"] and len(frame) == len(out.rows)
    bundle = nma_report.build_report(nma_report.ReportInput(
        document=document, model=model, summary=summarize_network(model), result=result, effects="random",
        title="Thr", threshold=out))
    with zipfile.ZipFile(io.BytesIO(bundle.docx_bytes)) as zf:
        text = zf.read("word/document.xml").decode("utf-8")
    assert "Table 10c" in text and "Phillippo" in text
    import openpyxl

    assert "Threshold" in openpyxl.load_workbook(io.BytesIO(bundle.xlsx_bytes)).sheetnames
