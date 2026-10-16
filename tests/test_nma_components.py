"""Additive component NMA (B3): hand-calculated 3-contrast example, exact additive data,
and equivalence with the standard NMA when every treatment is its own component."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from heor_tree_engine.nma_components import parse_components, run_component_nma  # noqa: E402
from heor_tree_engine.nma_freq import run_nma_frequentist  # noqa: E402
from heor_tree_engine.nma_parser import resolve_nma  # noqa: E402

COMPONENTS = {"X": "X", "Y": "Y", "XY": "X + Y"}


def _net(contrasts):
    studies = {}
    for study, t, v, est in contrasts:
        studies.setdefault(study, []).append({"treatment": t, "versus": v, "estimate": est, "se": 1.0})
    return resolve_nma({
        "spec": {"outcome": "continuous", "measure": "MD", "reference": "P"},
        "treatments": [{"name": n} for n in ("P", "X", "Y", "XY")],
        "studies": [{"id": s, "contrasts": c} for s, c in studies.items()]})


def test_parse_components():
    assert parse_components("A + B, C;A") == ["A", "B", "C"] and parse_components(None) == []


def test_hand_calculation():
    out = run_component_nma(_net([("s1", "X", "P", 1.0), ("s2", "Y", "P", 2.0), ("s3", "XY", "P", 3.5)]), COMPONENTS, "fixed")
    assert not out.errors
    effects = {e.treatment: e.estimate for e in out.component_effects}
    assert effects["X"] == pytest.approx(3.5 / 3) and effects["Y"] == pytest.approx(6.5 / 3)
    assert out.q_additive == pytest.approx(1 / 12) and out.q_standard == pytest.approx(0.0, abs=1e-12)
    assert out.df_diff == 1 and out.q_diff == pytest.approx(1 / 12)
    xy = next(e for e in out.fit.effects if e.treatment == "XY")
    assert xy.estimate == pytest.approx(10 / 3)


def test_exactly_additive_data_gives_zero_q_diff():
    out = run_component_nma(_net([("s1", "X", "P", 1.0), ("s2", "Y", "P", 2.0), ("s3", "XY", "P", 3.0),
                                  ("s4", "XY", "X", 2.0)]), COMPONENTS, "fixed")
    assert out.q_diff == pytest.approx(0.0, abs=1e-9)
    xy = next(e for e in out.fit.effects if e.treatment == "XY")
    assert xy.estimate == pytest.approx(3.0)


def test_own_components_equal_standard_nma():
    model = _net([("s1", "X", "P", 1.0), ("s2", "Y", "P", 2.0), ("s3", "XY", "P", 3.5), ("s4", "XY", "X", 2.0)])
    out = run_component_nma(model, None, "random")
    std = run_nma_frequentist(model)
    for a, b in zip(out.fit.effects, std.random.effects):
        assert a.estimate == pytest.approx(b.estimate, abs=1e-9) and a.se == pytest.approx(b.se, abs=1e-9)
    assert out.q_diff == pytest.approx(0.0, abs=1e-9)


def test_unidentified_components_are_reported():
    model = resolve_nma({
        "spec": {"outcome": "continuous", "measure": "MD", "reference": "P"},
        "treatments": [{"name": n} for n in ("P", "XY", "Z")],
        "studies": [{"id": "s1", "contrasts": [{"treatment": "XY", "versus": "P", "estimate": 3.0, "se": 1.0}]},
                    {"id": "s2", "contrasts": [{"treatment": "Z", "versus": "P", "estimate": 1.0, "se": 1.0}]}]})
    out = run_component_nma(model, {"XY": "X + Y"}, "fixed")
    assert out.errors and out.errors[0].code == "components_not_identified"


def test_components_reach_the_report():
    import io
    import zipfile

    import nma_editor_support as sup
    import nma_report
    from heor_tree_engine.nma_network import summarize_network

    model = _net([("s1", "X", "P", 1.0), ("s2", "Y", "P", 2.0), ("s3", "XY", "P", 3.5), ("s4", "XY", "X", 2.0)])
    out = run_component_nma(model, COMPONENTS, "random")
    assert not out.errors
    document = {"name": "c", "spec": {"outcome": "continuous", "measure": "MD", "reference": "P"}}
    bundle = nma_report.build_report(nma_report.ReportInput(
        document=document, model=model, summary=summarize_network(model), result=out.standard, effects="random",
        title="Cmp", components=out))
    with zipfile.ZipFile(io.BytesIO(bundle.docx_bytes)) as zf:
        text = zf.read("word/document.xml").decode("utf-8")
    assert "Table 10d" in text and "Rücker" in text
    import openpyxl

    assert "Components" in openpyxl.load_workbook(io.BytesIO(bundle.xlsx_bytes)).sheetnames
