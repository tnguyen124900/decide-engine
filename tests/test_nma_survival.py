"""Survival NMA (C1c): closed-form Poisson case, analytic cumulative hazard, parameter recovery and model choice
on simulated Weibull data."""

import math
import random
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from heor_tree_engine.nma_km_reconstruct import ReconstructedArm, interval_counts  # noqa: E402
from heor_tree_engine.nma_survival import (SurvivalArmData, SurvivalSettings, basis, cumulative_hazard,  # noqa: E402
                                            model_powers, run_survival_nma, survival_summary)


def _arm(study, treatment, events, py, t=(0.0, 10.0)):
    return SurvivalArmData(study, treatment, [t[0]], [t[1]], [events], [py])


def test_exponential_matches_the_poisson_closed_form():
    arms = [_arm("s1", "A", 100, 1000), _arm("s1", "B", 60, 1000), _arm("s2", "A", 50, 500), _arm("s2", "B", 30, 500)]
    res = run_survival_nma(arms, "A", SurvivalSettings(model="exponential", effects="fixed", chains=3, warmup=400,
                                                       draws_per_chain=1500, seed=4))
    assert not res.errors
    d = next(x for x in res.diagnostics if x.name.startswith("d[B]"))
    assert d.median == pytest.approx(math.log(90 / 150), abs=0.02)  # pooled MLE
    assert d.sd == pytest.approx(math.sqrt(1 / 150 + 1 / 90), rel=0.1)
    assert d.rhat < 1.05
    # survival of the reference: exp(-lambda t) with lambda = 150 / 1500 per unit time
    times = np.array([0.0, 5.0, 10.0])
    curves = survival_summary(res, times)
    assert curves["A"]["median"][2] == pytest.approx(math.exp(-0.1 * 10), abs=0.01)


def test_cumulative_hazard_is_analytic_for_weibull_and_gompertz():
    t = np.array([0.0, 5.0, 10.0, 20.0])
    scale = 20.0
    theta = np.array([[-1.2, 0.4]])
    h = cumulative_hazard(theta, t, [0.0], scale)[0]  # Weibull: h_s(s) = e^a s^b, H = e^a s^(b+1) / (b+1)
    s = t / scale
    np.testing.assert_allclose(h, math.exp(-1.2) * s ** 1.4 / 1.4, rtol=1e-4, atol=1e-12)
    theta = np.array([[-2.0, 1.5]])
    g = cumulative_hazard(theta, t, [1.0], scale)[0]  # Gompertz: h_s = e^(a + b s), H = e^a (e^(b s) - 1) / b
    np.testing.assert_allclose(g, math.exp(-2.0) * (np.exp(1.5 * s) - 1) / 1.5, rtol=1e-4, atol=1e-12)


def test_basis_fractional_polynomial_terms():
    t = np.array([0.5, 2.0])
    b = basis(t, [0.0, 0.0, 1.0])
    np.testing.assert_allclose(b[:, 1], np.log(t))
    np.testing.assert_allclose(b[:, 2], np.log(t) ** 2)
    np.testing.assert_allclose(b[:, 3], t)
    with pytest.raises(ValueError):
        model_powers("fp2", [0.3, 1.0])


def _simulate_network(seed=1):
    rng = random.Random(seed)
    k, lam_a, hr = 1.3, 0.02, 0.6
    breaks = [0, 3, 6, 9, 12, 18, 24, 36]
    arms = []

    def sim(n, lam):
        latent = [(-math.log(rng.random()) / lam) ** (1 / k) for _ in range(n)]
        censor = [rng.uniform(20, 40) for _ in range(n)]
        a = ReconstructedArm()
        a.times = [min(x, y) for x, y in zip(latent, censor)]
        a.events = [1 if x <= y else 0 for x, y in zip(latent, censor)]
        return a

    for s in range(4):
        for trt, lam in (("A", lam_a), ("B", lam_a * hr)):
            r = interval_counts(sim(250, lam), breaks)
            arms.append(SurvivalArmData(f"s{s}", trt, [x[0] for x in r], [x[1] for x in r],
                                        [x[2] for x in r], [x[3] for x in r]))
    return arms, math.log(hr)


def test_weibull_recovers_the_hazard_ratio_and_beats_the_exponential():
    arms, log_hr = _simulate_network()
    weibull = run_survival_nma(arms, "A", SurvivalSettings(model="weibull", effects="fixed", warmup=500, draws_per_chain=500, seed=2))
    expo = run_survival_nma(arms, "A", SurvivalSettings(model="exponential", effects="fixed", warmup=500, draws_per_chain=500, seed=2))
    assert not weibull.errors and weibull.dic < expo.dic - 20
    # hazard ratio at the middle of follow-up (scaled time 0.5) from the two coefficients
    d = weibull.d_draws[:, 1, :]
    log_hr_mid = d[:, 0] + d[:, 1] * math.log(0.5)
    assert abs(np.median(log_hr_mid) - log_hr) < 2 * float(np.std(log_hr_mid)) + 0.05
    assert all(x.rhat < 1.1 for x in weibull.diagnostics)
    curves = survival_summary(weibull, np.linspace(0, 36, 13))
    assert curves["B"]["median"][-1] > curves["A"]["median"][-1]  # B (HR 0.6) survives longer
    assert curves["A"]["lower"][6] < curves["A"]["median"][6] < curves["A"]["upper"][6]


def test_random_effects_run_and_errors():
    arms, _ = _simulate_network(seed=3)
    res = run_survival_nma(arms, "A", SurvivalSettings(model="weibull", effects="random", warmup=300, draws_per_chain=300, seed=1))
    assert not res.errors and res.tau_draws is not None and math.isfinite(res.dic)
    assert any(x.name.startswith("tau") for x in res.diagnostics)
    assert run_survival_nma(arms, "Z").errors  # unknown reference
    assert run_survival_nma(arms, "A", SurvivalSettings(model="fp2", powers=[0.3])).errors
    disconnected = arms + [_arm("x", "C", 5, 50), _arm("x", "D", 4, 50)]
    assert run_survival_nma(disconnected, "A", SurvivalSettings(model="exponential")).errors


def _psm_survival(model, row, t):
    if model == "exponential":
        return math.exp(-row["rate"] * t)
    if model == "weibull":
        return math.exp(-((t / row["scale"]) ** row["shape"]))
    return math.exp(-(row["rate"] / row["shape"]) * (math.exp(row["shape"] * t) - 1.0))


def test_psm_parameters_reproduce_the_fitted_curves():
    from heor_tree_engine.nma_survival import psa_survival_draws, psm_parameter_draws, survival_table

    arms, _ = _simulate_network()
    for model in ("exponential", "weibull", "gompertz"):
        res = run_survival_nma(arms, "A", SurvivalSettings(model=model, effects="fixed", warmup=300, draws_per_chain=200, seed=3))
        assert not res.errors
        times_months = np.array([0.0, 6.0, 12.0, 24.0, 36.0])
        params = psm_parameter_draws(res, time_factor=1 / 12, n_draws=50)
        curves = psa_survival_draws(res, times_months, n_draws=50)
        checked = 0
        for _, row in params[params["treatment"] == "B"].head(10).iterrows():
            sub = curves[(curves["treatment"] == "B") & (curves["draw"] == row["draw"])]
            for _, c in sub.iterrows():
                assert _psm_survival(model, row, c["time"] / 12) == pytest.approx(c["survival"], abs=1e-4)
                checked += 1
        assert checked >= 20
    table = survival_table(res, [0, 12, 24])
    assert list(table.columns) == ["time", "treatment", "survival", "lower", "upper"] and len(table) == 6
    fp = run_survival_nma(arms, "A", SurvivalSettings(model="fp2", powers=[0.0, 1.0], effects="fixed", warmup=300, draws_per_chain=200, seed=3))
    assert not fp.errors and math.isfinite(fp.dic) and len(fp.coefficient_names) == 3
    with pytest.raises(ValueError):
        psm_parameter_draws(fp)


def test_survival_reaches_the_report():
    import io
    import zipfile

    import nma_editor_support as sup
    import nma_report
    from heor_tree_engine.nma_freq import run_nma_frequentist
    from heor_tree_engine.nma_network import summarize_network
    from heor_tree_engine.nma_parser import resolve_nma

    arms, _ = _simulate_network()
    fits = {}
    for name in ("exponential", "weibull"):
        fits[name.capitalize()] = run_survival_nma(arms, "A", SurvivalSettings(model=name, effects="fixed", warmup=200, draws_per_chain=150, seed=1))
    frame = sup.survival_models_frame(fits)
    assert list(frame["ΔDIC"]).count(0.0) == 1 and len(frame) == 2
    document = sup.default_document()
    model = resolve_nma(document)
    bundle = nma_report.build_report(nma_report.ReportInput(
        document=document, model=model, summary=summarize_network(model), result=run_nma_frequentist(model), effects="random",
        title="S", survival={"fits": fits, "selected": "Weibull"}))
    with zipfile.ZipFile(io.BytesIO(bundle.docx_bytes)) as zf:
        assert "Table 10g" in zf.read("word/document.xml").decode("utf-8")
    import openpyxl

    assert "Survival models" in openpyxl.load_workbook(io.BytesIO(bundle.xlsx_bytes)).sheetnames
