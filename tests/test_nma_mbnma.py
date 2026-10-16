"""Dose-response MBNMA (C3): exact linear case by hand, binary logit closed form, Emax recovery and AIC model
choice, shared parameters within a class, and extrapolation warnings."""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from heor_tree_engine.nma_mbnma import DoseArm, dose_response, run_mbnma  # noqa: E402


def test_linear_slope_by_hand():
    # two studies, placebo and dose 10; differences 2.0 and 1.6, se 1 per arm -> contrast variance 2
    arms = [DoseArm("s1", "X", 0, mean=0.0, se=1.0), DoseArm("s1", "X", 10, mean=2.0, se=1.0),
            DoseArm("s2", "X", 0, mean=1.0, se=1.0), DoseArm("s2", "X", 10, mean=2.6, se=1.0)]
    res = run_mbnma(arms, "linear")
    assert not res.errors
    assert res.theta[0] == pytest.approx(0.18, abs=1e-8)
    assert res.se("X: theta0") == pytest.approx(0.1, abs=1e-6)
    assert res.df == 4 - 2 - 1
    f, lo, hi = res.curve("X", [0, 10])
    assert f[1] == pytest.approx(1.8) and lo[1] < f[1] < hi[1] and f[0] == 0


def test_binary_single_study_is_the_log_odds_ratio_per_unit_dose():
    arms = [DoseArm("s", "X", 0, events=20, n=100), DoseArm("s", "X", 5, events=40, n=100)]
    res = run_mbnma(arms, "linear")
    assert res.outcome == "binary" and not res.errors
    assert res.theta[0] == pytest.approx(math.log((40 / 60) / (20 / 80)) / 5, abs=1e-7)
    assert res.se("X: theta0") == pytest.approx(math.sqrt(1 / 20 + 1 / 80 + 1 / 40 + 1 / 60) / 5, rel=1e-5)


def _emax_network(seed=3, n_studies=14, sd=0.4):
    rng = np.random.default_rng(seed)
    truth = {"A": (-8.0, math.log(20.0)), "B": (-5.0, math.log(10.0))}
    arms = []
    for i in range(n_studies):
        agent = "A" if i % 2 == 0 else "B"
        base = rng.normal(50, 3)
        for dose in (0, 10, 30, 100) if agent == "A" else (0, 5, 20, 60):
            effect = 0.0 if dose == 0 else dose_response("emax", dose, truth[agent])
            arms.append(DoseArm(f"s{i}", agent, dose, mean=float(base + effect + rng.normal(0, sd)), se=sd))
    return arms, truth


def test_emax_is_recovered_and_preferred_to_linear():
    arms, truth = _emax_network()
    emax = run_mbnma(arms, "emax")
    linear = run_mbnma(arms, "linear")
    assert not emax.errors and not linear.errors
    assert emax.aic < linear.aic - 20
    for agent, (e0, l0) in truth.items():
        i = emax.parameter_names.index(f"{agent}: theta0")
        j = emax.parameter_names.index(f"{agent}: theta1")
        assert abs(emax.theta[i] - e0) < 3 * emax.se(f"{agent}: theta0") + 0.3
        assert abs(emax.theta[j] - l0) < 3 * emax.se(f"{agent}: theta1") + 0.3
    f, lo, hi = emax.curve("A", [0, 20, 100])
    assert f[1] == pytest.approx(dose_response("emax", 20, truth["A"]), abs=0.6)
    assert np.all(hi >= lo)


def test_shared_group_gives_one_curve_and_hill_runs():
    arms, _ = _emax_network()
    res = run_mbnma(arms, "emax", groups={"A": "class", "B": "class"})
    assert res.group_names == ["class"] and len(res.theta) == 2
    a = res.curve("A", [10, 50])[0]
    b = res.curve("B", [10, 50])[0]
    np.testing.assert_allclose(a, b)
    hill = run_mbnma(arms, "emax_hill")
    assert not hill.errors and math.isfinite(hill.aic)


def test_errors():
    assert run_mbnma([DoseArm("s", "X", 0, mean=1.0, se=1.0)], "linear").errors
    assert run_mbnma([DoseArm("s", "X", 0, mean=1.0, se=1.0), DoseArm("s", "X", 0, mean=1.0, se=1.0)], "linear").errors
    assert run_mbnma([DoseArm("s", "X", 0, mean=1.0, se=1.0), DoseArm("s", "X", 5, mean=2.0, se=1.0)], "cubic").errors
    mixed = [DoseArm("s", "X", 0, mean=1.0, se=1.0), DoseArm("s", "X", 5, events=3, n=10)]
    assert run_mbnma(mixed, "linear").errors


def test_dose_response_reaches_the_report():
    import io
    import zipfile

    import nma_editor_support as sup
    import nma_report
    from heor_tree_engine.nma_freq import run_nma_frequentist
    from heor_tree_engine.nma_network import summarize_network
    from heor_tree_engine.nma_parser import resolve_nma

    arms, _ = _emax_network()
    fits = {"Linear": run_mbnma(arms, "linear"), "Emax": run_mbnma(arms, "emax")}
    frame = sup.mbnma_models_frame(fits)
    assert list(frame["ΔAIC"]).count(0.0) == 1
    document = sup.default_document()
    model = resolve_nma(document)
    bundle = nma_report.build_report(nma_report.ReportInput(
        document=document, model=model, summary=summarize_network(model), result=run_nma_frequentist(model), effects="random",
        title="D", dose={"fits": fits, "selected": "Emax"}))
    with zipfile.ZipFile(io.BytesIO(bundle.docx_bytes)) as zf:
        assert "Table 10j" in zf.read("word/document.xml").decode("utf-8")
    import openpyxl

    assert "Dose-response models" in openpyxl.load_workbook(io.BytesIO(bundle.xlsx_bytes)).sheetnames
