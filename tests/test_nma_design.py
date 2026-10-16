"""Non-randomised evidence (B4): down-weighting equals inflating the variance, alpha -> 0 approaches the
RCT-only fit, and the scenario table is complete."""

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from heor_tree_engine.nma_design import run_design_sensitivity  # noqa: E402
from heor_tree_engine.nma_freq import run_nma_frequentist  # noqa: E402
from heor_tree_engine.nma_parser import resolve_nma  # noqa: E402


def _net(nrs_se=1.0):
    def c(study, t, v, est, se):
        return {"id": study, "contrasts": [{"treatment": t, "versus": v, "estimate": est, "se": se}]}
    return resolve_nma({
        "spec": {"outcome": "continuous", "measure": "MD", "reference": "A"},
        "treatments": [{"name": n} for n in "ABC"],
        "studies": [c("r1", "B", "A", 1.0, 0.5), c("r2", "C", "A", 2.0, 0.5), c("r3", "C", "B", 1.2, 0.6),
                    c("n1", "B", "A", 3.0, nrs_se)]})


def test_weight_is_variance_inflation():
    out = run_design_sensitivity(_net(), ["n1"], alphas=(0.25,), effects="fixed")
    weighted = out.scenarios[2].effects
    reference = run_nma_frequentist(_net(nrs_se=1.0 / math.sqrt(0.25))).fixed
    for e in reference.effects:
        assert weighted[e.treatment][0] == pytest.approx(e.natural, abs=1e-9)
        assert weighted[e.treatment][1] == pytest.approx(e.natural_lower, abs=1e-9)


def test_scenarios_and_limits():
    out = run_design_sensitivity(_net(), ["n1"], alphas=(1e-6,), effects="fixed")
    names = [s.name for s in out.scenarios]
    assert names[0] == "RCT only" and names[1].startswith("All studies") and len(names) == 3
    rct, tiny = out.scenarios[0].effects, out.scenarios[2].effects
    for t in rct:
        assert tiny[t][0] == pytest.approx(rct[t][0], abs=1e-3)
    assert out.max_shift["B"] > 0.1  # the NRS (3.0 vs ~1) pulls B strongly


def test_errors():
    assert run_design_sensitivity(_net(), []).errors
    assert run_design_sensitivity(_net(), ["r1", "r2", "r3", "n1"]).errors


def test_design_reaches_the_report():
    import io
    import zipfile

    import nma_editor_support as sup
    import nma_report
    from heor_tree_engine.nma_network import summarize_network

    model = _net()
    out = run_design_sensitivity(model, ["n1"], effects="random")
    frame = sup.design_frame(out)
    assert list(frame.columns)[1] == "RCT only"
    bundle = nma_report.build_report(nma_report.ReportInput(
        document={"name": "d", "spec": {"outcome": "continuous", "measure": "MD", "reference": "A"}}, model=model,
        summary=summarize_network(model), result=run_nma_frequentist(model), effects="random", title="D", design=out))
    with zipfile.ZipFile(io.BytesIO(bundle.docx_bytes)) as zf:
        assert "Table 10e" in zf.read("word/document.xml").decode("utf-8")
