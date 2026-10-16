"""Piecewise HR NMA and PH checks (C1a, C1b): hand-calculated Cochran Q, per-interval equivalence with the
standard NMA, input handling."""

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from heor_tree_engine.nma_freq import chi2_sf, run_nma_frequentist  # noqa: E402
from heor_tree_engine.nma_parser import resolve_nma  # noqa: E402
from heor_tree_engine.nma_timevarying import normalize_rows, run_piecewise_nma  # noqa: E402


def _rows():
    rows = []
    for t0, t1, log_hr in ((0, 6, 0.5), (6, 12, 1.0)):
        for study in ("s1", "s2"):
            rows.append({"study": study, "treatment": "B", "versus": "A", "t_from": t0, "t_to": t1,
                         "log_hr": log_hr, "se": 0.1})
    return rows


def test_hand_calculation_of_interval_estimates_and_q():
    out = run_piecewise_nma(_rows(), "A", "fixed")
    assert not out.errors and out.intervals == [(0, 6), (6, 12)]
    first, second = out.estimates
    assert math.log(first.hr) == pytest.approx(0.5) and first.se == pytest.approx(0.1 / math.sqrt(2))
    assert math.log(second.hr) == pytest.approx(1.0)
    net = next(r for r in out.ph if r.scope == "network")
    assert net.q == pytest.approx(25.0) and net.df == 1 and net.flagged
    assert net.p == pytest.approx(chi2_sf(25.0, 1))
    study = [r for r in out.ph if r.scope == "study"]
    assert len(study) == 2 and study[0].q == pytest.approx(0.25 / (2 * 0.01) )  # (0.5)^2 / (0.01+0.01)


def test_constant_hr_is_not_flagged():
    rows = [dict(r, log_hr=0.7) for r in _rows()]
    out = run_piecewise_nma(rows, "A", "fixed")
    assert not any(r.flagged for r in out.ph)


def test_interval_equals_standard_nma():
    rows = [{"study": "s1", "treatment": "B", "versus": "A", "t_from": 0, "t_to": 6, "log_hr": 0.4, "se": 0.2},
            {"study": "s2", "treatment": "C", "versus": "A", "t_from": 0, "t_to": 6, "log_hr": 0.9, "se": 0.3},
            {"study": "s3", "treatment": "C", "versus": "B", "t_from": 0, "t_to": 6, "log_hr": 0.3, "se": 0.25}]
    out = run_piecewise_nma(rows, "A", "random")
    model = resolve_nma({"spec": {"outcome": "contrast", "measure": "HR", "reference": "A"},
                         "treatments": [{"name": n} for n in "ABC"],
                         "studies": [{"id": r["study"], "contrasts": [{"treatment": r["treatment"], "versus": r["versus"],
                                                                        "estimate": r["log_hr"], "se": r["se"]}]} for r in rows]})
    std = {e.treatment: e for e in run_nma_frequentist(model).random.effects}
    for e in out.estimates:
        assert e.log_hr == pytest.approx(std[e.treatment].estimate) and e.se == pytest.approx(std[e.treatment].se)


def test_ci_input_and_bad_rows():
    rows, problems = normalize_rows([
        {"study": "s", "treatment": "B", "versus": "A", "t_from": 0, "t_to": 6, "hr": 0.5, "lower": 0.3, "upper": 0.83},
        {"study": "s", "treatment": "B", "versus": "A", "t_from": 6, "t_to": 6, "log_hr": 0.1, "se": 0.1},
        {"study": "", "treatment": "", "versus": "", "t_from": None, "t_to": None}])
    assert len(rows) == 1 and rows[0]["se"] == pytest.approx((math.log(0.83) - math.log(0.3)) / 3.92, rel=1e-3)
    assert len(problems) == 1
    assert run_piecewise_nma([]).errors


def test_piecewise_reaches_the_report():
    import io
    import zipfile

    import nma_editor_support as sup
    import nma_report
    from heor_tree_engine.nma_network import summarize_network

    out = run_piecewise_nma(_rows(), "A", "random")
    assert "0-6 months" in sup.piecewise_frame(out, "months").columns
    assert list(sup.ph_frame(out)["p"])  # non-empty
    document = sup.default_document()
    model = resolve_nma(document)
    bundle = nma_report.build_report(nma_report.ReportInput(
        document=document, model=model, summary=summarize_network(model), result=run_nma_frequentist(model),
        effects="random", title="T", piecewise=out))
    with zipfile.ZipFile(io.BytesIO(bundle.docx_bytes)) as zf:
        assert "Table 10f" in zf.read("word/document.xml").decode("utf-8")
    import openpyxl

    assert "Piecewise HR" in openpyxl.load_workbook(io.BytesIO(bundle.xlsx_bytes)).sheetnames
