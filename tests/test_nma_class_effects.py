"""Class effects in the exact-likelihood Bayesian NMA (B1).

* exchangeable class effects on a tiny fixed-effect network of two
  contrast-level studies have a closed form (Gaussian conditional on
  sigma_class, then a 1-D numerical integral over sigma_class);
* the common class effect equals an ordinary NMA of the network with the
  classes merged into single treatments;
* an exchangeable model with a vague sigma_class prior stays close to the model
  without classes, one with a very tight prior close to the common effect;
* classes with one treatment (or no classes) leave the model unchanged.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from heor_tree_engine.nma_bayes import BayesSettings  # noqa: E402
from heor_tree_engine.nma_bayes_exact import run_nma_bayesian_exact  # noqa: E402
from heor_tree_engine.nma_freq import run_nma_frequentist  # noqa: E402
from heor_tree_engine.nma_parser import resolve_nma  # noqa: E402


def _two_contrast_network(classes=("X", "X")):
    return resolve_nma({
        "spec": {"outcome": "continuous", "measure": "MD", "reference": "P"},
        "treatments": [{"name": "P"}, {"name": "A1", "class": classes[0]}, {"name": "A2", "class": classes[1]}],
        "studies": [
            {"id": "s1", "contrasts": [{"treatment": "A1", "versus": "P", "estimate": -1.0, "se": 0.35}]},
            {"id": "s2", "contrasts": [{"treatment": "A2", "versus": "P", "estimate": -2.2, "se": 0.40}]},
        ]})


def _analytic_exchangeable(y, s, prior_sd=100.0, scale=1.0):
    """Posterior mean / SD of (d1, d2) and mean of the class mean gamma when
    d_k ~ N(gamma, sigma^2), gamma ~ N(0, prior_sd^2), sigma ~ Half-Normal(0, scale),
    y_k ~ N(d_k, s_k^2) (fixed effect, one contrast per treatment)."""

    y, s = np.array(y), np.array(s)
    grid = np.linspace(1e-4, 25.0, 25000)
    logw, means, second, gam = [], [], [], []
    for sigma in grid:
        cov_d = sigma ** 2 * np.eye(2) + prior_sd ** 2 * np.ones((2, 2))
        cov_y = cov_d + np.diag(s ** 2)
        inv = np.linalg.inv(cov_y)
        logw.append(-0.5 * math.log(np.linalg.det(cov_y)) - 0.5 * y @ inv @ y - 0.5 * (sigma / scale) ** 2)
        mean = cov_d @ inv @ y
        cov_post = cov_d - cov_d @ inv @ cov_d
        means.append(mean)
        second.append(cov_post + np.outer(mean, mean))
        prec = 1.0 / prior_sd ** 2 + np.sum(1.0 / (sigma ** 2 + s ** 2))
        gam.append(np.sum(y / (sigma ** 2 + s ** 2)) / prec)
    w = np.exp(np.array(logw) - max(logw))
    w /= w.sum()
    mean = np.einsum("g,gi->i", w, np.array(means))
    second = np.einsum("g,gij->ij", w, np.array(second))
    sd = np.sqrt(np.diag(second) - mean ** 2)
    return mean, sd, float(w @ np.array(gam))


def test_exchangeable_class_effect_matches_the_closed_form() -> None:
    model = _two_contrast_network()
    settings = BayesSettings(effects="fixed", chains=4, draws_per_chain=3000, seed=17, class_model="exchangeable",
                             class_sd_scale=1.0)
    result = run_nma_bayesian_exact(model, settings, warmup=1000)
    assert result.errors == [] and result.class_model == "exchangeable"
    assert result.class_members == {"X": ["A1", "A2"]}
    mean, sd, gamma = _analytic_exchangeable([-1.0, -2.2], [0.35, 0.40])
    got = {e.treatment: e for e in result.fit.effects}
    assert got["A1"].estimate == pytest.approx(mean[0], abs=0.04)
    assert got["A2"].estimate == pytest.approx(mean[1], abs=0.04)
    assert got["A1"].se == pytest.approx(sd[0], rel=0.08)
    assert got["A2"].se == pytest.approx(sd[1], rel=0.08)
    assert result.class_effects["X"].estimate == pytest.approx(gamma, abs=0.05)
    assert result.class_sd is not None and 0.0 < result.class_sd.median < 3.0
    assert max(d.rhat for d in result.diagnostics) < 1.05
    # shrinkage: each treatment moves towards the class mean relative to the plain (no class) estimate
    plain = run_nma_bayesian_exact(model, BayesSettings(effects="fixed", chains=3, draws_per_chain=1500, seed=3),
                                   warmup=500)
    plain_a = {e.treatment: e.estimate for e in plain.fit.effects}
    assert abs(got["A1"].estimate - got["A2"].estimate) < abs(plain_a["A1"] - plain_a["A2"])


def _five_arm_network(rng_seed=5):
    """P, A1, A2 (class X), B, C (class Y); no study contains both members of a class."""

    rng = np.random.default_rng(rng_seed)
    true = {"P": 0.0, "A1": -1.5, "A2": -2.5, "B": -1.0, "C": -3.5}
    designs = [("P", "A1"), ("P", "A1"), ("P", "A2"), ("P", "A2"), ("A1", "B"), ("A2", "C"), ("P", "B"), ("P", "C"),
               ("B", "C"), ("P", "B", "A1")]
    studies = []
    for i, design in enumerate(designs):
        base = rng.normal(20, 3)
        studies.append({"id": f"s{i}", "arms": [{"treatment": t, "mean": base + true[t] + rng.normal(0, 0.6), "sd": 6.0,
                                                 "n": int(rng.integers(40, 90))} for t in design]})
    return studies


def _model(studies, classes):
    return resolve_nma({"spec": {"outcome": "continuous", "measure": "MD", "reference": "P"},
                        "treatments": [{"name": n, **({"class": classes[n]} if n in classes else {})}
                                       for n in ("P", "A1", "A2", "B", "C")], "studies": studies})


def test_common_class_effect_equals_the_network_with_the_classes_merged() -> None:
    classes = {"A1": "X", "A2": "X", "B": "Y", "C": "Y"}
    studies = _five_arm_network()
    # the multi-arm study (P, B, A1) is fine; merge X = {A1, A2}, Y = {B, C} for the reference analysis
    merged_studies = []
    for study in studies:
        names = [arm["treatment"] for arm in study["arms"]]
        if len({classes.get(n, n) for n in names}) < len(names):
            continue  # a study comparing two members of one class disappears when they are merged
        merged_studies.append({"id": study["id"], "arms": [{**arm, "treatment": classes.get(arm["treatment"], arm["treatment"])}
                                                           for arm in study["arms"]]})
    merged = resolve_nma({"spec": {"outcome": "continuous", "measure": "MD", "reference": "P"},
                          "treatments": [{"name": n} for n in ("P", "X", "Y")], "studies": merged_studies})
    dropped = [s["id"] for s in studies if s["id"] not in {m["id"] for m in merged_studies}]
    assert dropped == ["s8"]  # B vs C is a within-class comparison: no information under a common effect

    model = _model(studies, classes)
    settings = BayesSettings(effects="fixed", chains=4, draws_per_chain=1500, seed=23, class_model="common")
    result = run_nma_bayesian_exact(model, settings, warmup=600)
    assert result.errors == [] and result.class_model == "common"
    reference = run_nma_frequentist(merged)
    assert reference.errors == []
    freq = {e.treatment: e for e in reference.fixed.effects}
    got = {e.treatment: e for e in result.fit.effects}
    assert got["A1"].estimate == pytest.approx(got["A2"].estimate, abs=1e-12)  # same parameter
    assert got["B"].estimate == pytest.approx(got["C"].estimate, abs=1e-12)
    assert got["A1"].estimate == pytest.approx(freq["X"].estimate, abs=0.06)
    assert got["B"].estimate == pytest.approx(freq["Y"].estimate, abs=0.06)
    assert result.class_effects["X"].estimate == pytest.approx(freq["X"].estimate, abs=0.06)
    assert result.class_effects["Y"].se == pytest.approx(freq["Y"].se, rel=0.1)
    assert set(result.class_sucra) == {"X", "Y"}
    assert result.class_sucra["Y"] > result.class_sucra["X"] or result.class_sucra["X"] > result.class_sucra["Y"]
    # the merged network has fewer effective parameters than the plain model
    plain = run_nma_bayesian_exact(_model(studies, {}), BayesSettings(effects="fixed", chains=3, draws_per_chain=1000,
                                                                    seed=3), warmup=500)
    assert result.p_d < plain.p_d + 1.0


def test_exchangeable_interpolates_between_no_classes_and_common() -> None:
    classes = {"A1": "X", "A2": "X", "B": "Y", "C": "Y"}
    model = _model(_five_arm_network(), classes)
    base = dict(effects="fixed", chains=4, draws_per_chain=1500, seed=29)
    plain = {e.treatment: e.estimate for e in run_nma_bayesian_exact(
        _model(_five_arm_network(), {}), BayesSettings(**base), warmup=600).fit.effects}
    loose = {e.treatment: e.estimate for e in run_nma_bayesian_exact(
        model, BayesSettings(**base, class_model="exchangeable", class_sd_scale=100.0), warmup=600).fit.effects}
    tight_result = run_nma_bayesian_exact(
        model, BayesSettings(**base, class_model="exchangeable", class_sd_scale=0.01), warmup=600)
    tight = {e.treatment: e.estimate for e in tight_result.fit.effects}
    for name in plain:
        assert loose[name] == pytest.approx(plain[name], abs=0.25)  # vague sigma_class: little shrinkage
    assert abs(tight["A1"] - tight["A2"]) < 0.1 and abs(tight["B"] - tight["C"]) < 0.1  # tight: ~ common effect
    assert abs(plain["A1"] - plain["A2"]) > 0.5  # the data do separate them


def test_classes_with_a_single_treatment_change_nothing() -> None:
    model = _model(_five_arm_network(), {"A1": "X", "B": "Y"})  # one member each
    for mode in ("common", "exchangeable"):
        result = run_nma_bayesian_exact(model, BayesSettings(effects="fixed", chains=2, draws_per_chain=200, seed=1,
                                                             class_model=mode), warmup=100)
        assert result.errors == [] and result.class_model == "none" and result.class_effects == {}
        assert any("class" in w.lower() for w in result.warnings)
    # random-effects exchangeable model runs and reports sigma_class
    classes = {"A1": "X", "A2": "X"}
    result = run_nma_bayesian_exact(_model(_five_arm_network(), classes), BayesSettings(
        effects="random", chains=3, draws_per_chain=600, seed=2, class_model="exchangeable"), warmup=400)
    assert result.errors == [] and result.tau is not None and result.class_sd is not None
    assert any(d.name == "sigma_class" for d in result.diagnostics)
    assert result.dic is not None and math.isfinite(result.dic)


def test_class_effects_reach_the_tables_and_the_report() -> None:
    import io
    import zipfile

    import nma_editor_support as sup
    import nma_report
    from heor_tree_engine.nma_network import summarize_network

    document = sup.default_document()  # Class X = {Drug A, Drug B}, Class Y = {Drug C, Drug D}, Control = {Placebo}
    model = resolve_nma(document)
    settings = BayesSettings(effects="random", chains=3, draws_per_chain=500, seed=5, class_model="exchangeable")
    bayes = run_nma_bayesian_exact(model, settings, warmup=400)
    assert bayes.errors == [] and set(bayes.class_members) == {"Class X", "Class Y"}
    frame = sup.class_effects_frame(bayes, "OR", "CrI")
    assert list(frame["Class"]) == ["Class X", "Class Y"]
    assert "OR vs Placebo (CrI)" in frame.columns
    assert sum(float(v) for v in frame["SUCRA"]) == pytest.approx(1.0, abs=0.002)  # two classes: SUCRAs sum to 1

    frequentist = run_nma_frequentist(model)
    bundle = nma_report.build_report(nma_report.ReportInput(
        document=document, model=model, summary=summarize_network(model), result=frequentist, bayes=bayes,
        effects="random", title="Classes"))
    with zipfile.ZipFile(io.BytesIO(bundle.docx_bytes)) as zf:
        text = zf.read("word/document.xml").decode("utf-8")
    assert "Table 10b" in text and "exchangeable class effects" in text
    import openpyxl

    workbook = openpyxl.load_workbook(io.BytesIO(bundle.xlsx_bytes))
    assert "Class effects" in workbook.sheetnames
