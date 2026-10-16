"""MAIC (C2-a): weights match the target exactly, ESS, weighted Cox against brute force, population-adjustment
recovery on simulated data with an effect modifier, and the anchored / unanchored formulas."""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from heor_tree_engine.nma_maic import (_weighted_cox, anchored_indirect, estimate_weights, run_maic,  # noqa: E402
                                        unanchored_binary)


def test_weights_match_the_target_means_and_sds():
    rng = np.random.default_rng(1)
    X = np.column_stack([rng.normal(0, 1, 500), rng.binomial(1, 0.4, 500)])
    w = estimate_weights(X, [0.5, 0.55], [1.2, float("nan")], ["age", "male"])
    assert w.converged
    x = X[:, 0]
    wm = float((w.weights * x).sum() / w.weights.sum())
    wsd2 = float((w.weights * x ** 2).sum() / w.weights.sum()) - wm ** 2
    assert wm == pytest.approx(0.5, abs=1e-6) and math.sqrt(wsd2) == pytest.approx(1.2, abs=1e-6)
    assert float((w.weights * X[:, 1]).sum() / w.weights.sum()) == pytest.approx(0.55, abs=1e-6)
    assert w.weights.mean() == pytest.approx(1.0) and w.ess < 500
    assert {b["covariate"] for b in w.balance} == {"age", "male", "age²"}


def test_target_equal_to_sample_mean_gives_unit_weights():
    X = np.random.default_rng(2).normal(size=(200, 1))
    w = estimate_weights(X, [float(X.mean())])
    np.testing.assert_allclose(w.weights, 1.0, atol=1e-6)
    assert w.ess == pytest.approx(200, rel=1e-6)


def test_target_outside_the_data_is_flagged():
    X = np.random.default_rng(3).normal(size=(100, 1))
    w = estimate_weights(X, [8.0])
    assert not w.converged and w.warnings


def test_weighted_cox_matches_a_brute_force_likelihood():
    time = np.array([1.0, 2, 3, 4, 5, 6, 7, 8])
    event = np.array([1, 1, 1, 0, 1, 1, 0, 1], float)
    z = np.array([1, 0, 1, 0, 0, 1, 1, 0], float)
    w = np.array([1.0, 2, 1, 0.5, 1, 1.5, 1, 1])
    beta, se = _weighted_cox(time, event, z, w)

    def loglik(b):
        total = 0.0
        for i in range(8):
            if event[i]:
                risk = time >= time[i]
                total += w[i] * (b * z[i] - math.log(np.sum(w[risk] * np.exp(b * z[risk]))))
        return total

    grid = np.linspace(-3, 3, 60001)
    best = grid[int(np.argmax([loglik(b) for b in grid]))]
    assert beta == pytest.approx(best, abs=2e-4)
    curvature = -(loglik(beta + 1e-3) - 2 * loglik(beta) + loglik(beta - 1e-3)) / 1e-6
    assert se == pytest.approx(1 / math.sqrt(curvature), rel=1e-3)


def test_binary_two_by_two_reproduces_the_log_odds_ratio():
    n1, e1, n0, e0 = 120, 60, 100, 30
    event = np.array([1] * e1 + [0] * (n1 - e1) + [1] * e0 + [0] * (n0 - e0), float)
    arm = np.array([1] * n1 + [0] * n0)
    X = np.random.default_rng(4).normal(size=(n1 + n0, 1))
    res = run_maic(X, arm, "binary", {"event": event}, [float(X.mean())])
    expected = math.log((e1 / (n1 - e1)) / (e0 / (n0 - e0)))
    assert res.unadjusted == pytest.approx(expected, abs=1e-8)
    assert res.estimate == pytest.approx(expected, abs=1e-5)
    assert res.se == pytest.approx(math.sqrt(1 / e1 + 1 / (n1 - e1) + 1 / e0 + 1 / (n0 - e0)), rel=0.01)


def _effect_modifier_trial(seed=5, n=4000):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, n)
    arm = rng.integers(0, 2, n)
    y = 1 + 0.5 * x + arm * (-1 + 0.8 * x) + rng.normal(0, 1, n)  # effect at x is -1 + 0.8 x
    return x[:, None], arm, y


def test_maic_recovers_the_effect_in_the_target_population():
    X, arm, y = _effect_modifier_trial()
    res = run_maic(X, arm, "continuous", {"value": y}, [1.0], names=["x"])
    assert res.unadjusted == pytest.approx(-1.0, abs=0.12)  # IPD population has mean 0
    assert abs(res.estimate - (-0.2)) < 2.5 * res.se + 0.03  # target mean 1: -1 + 0.8
    assert res.ess < len(y) and res.se_method == "robust"


def test_survival_bootstrap_runs_and_ratio_scale():
    rng = np.random.default_rng(6)
    n = 300
    x = rng.normal(0, 1, n)
    arm = rng.integers(0, 2, n)
    t = rng.exponential(1 / (0.1 * np.exp(0.4 * x + arm * (-0.5))))
    c = rng.uniform(5, 20, n)
    time, event = np.minimum(t, c), (t <= c).astype(float)
    res = run_maic(x[:, None], arm, "survival", {"time": time, "event": event}, [0.3], n_boot=60)
    assert not res.errors and res.se_method == "bootstrap" and res.ratio
    lo, hi = res.interval
    assert lo < res.natural < hi and res.se > 0


def test_anchored_and_unanchored_formulas():
    est, se = anchored_indirect(-0.5, 0.2, -0.2, 0.15)
    assert est == pytest.approx(-0.3) and se == pytest.approx(0.25)
    events = np.array([1] * 30 + [0] * 70, float)
    est, se = unanchored_binary(events, np.ones(100), 40, 100)
    assert est == pytest.approx(math.log(0.3 / 0.7) - math.log(0.4 / 0.6))
    assert se == pytest.approx(math.sqrt(1 / (100 * 0.21) + 1 / (100 * 0.24)))


def test_population_adjustment_reaches_the_report():
    import io
    import zipfile

    import nma_editor_support as sup
    import nma_report
    from heor_tree_engine.nma_freq import run_nma_frequentist
    from heor_tree_engine.nma_mlnmr import AggArm, IPDArm, run_mlnmr
    from heor_tree_engine.nma_network import summarize_network
    from heor_tree_engine.nma_parser import resolve_nma

    X, arm, y = _effect_modifier_trial(n=600)
    effect = run_maic(X, arm, "continuous", {"value": y}, [1.0], names=["x"])
    frames = sup.maic_frames(effect)
    assert list(frames["effects"].iloc[:, 0]) == ["Unadjusted", "Adjusted (MAIC)"]
    rng = np.random.default_rng(2)
    ipd = [IPDArm("s", "A", X[:200], (rng.random(200) < 0.3).astype(float)), IPDArm("s", "B", X[:200], (rng.random(200) < 0.5).astype(float))]
    agg = [AggArm("t", "A", 35, 100, np.array([0.5]), np.array([1.0])), AggArm("t", "B", 50, 100, np.array([0.5]), np.array([1.0]))]
    fit = run_mlnmr(ipd, agg, ["x"], [False], ["x"], "A")
    assert not fit.errors
    document = sup.default_document()
    model = resolve_nma(document)
    bundle = nma_report.build_report(nma_report.ReportInput(
        document=document, model=model, summary=summarize_network(model), result=run_nma_frequentist(model), effects="random",
        title="P", popadj={"maic": frames, "mlnmr": sup.mlnmr_frames(fit, [0.5])}))
    with zipfile.ZipFile(io.BytesIO(bundle.docx_bytes)) as zf:
        text = zf.read("word/document.xml").decode("utf-8")
    assert "Table 10h" in text and "Table 10i" in text and "Signorini" in text
    import openpyxl

    assert {"MAIC balance", "ML-NMR effects"} <= set(openpyxl.load_workbook(io.BytesIO(bundle.xlsx_bytes)).sheetnames)
