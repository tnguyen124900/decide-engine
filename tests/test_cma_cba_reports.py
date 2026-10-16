"""CMA/CBA sections in the tree and Markov technical reports."""
import io
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import markov_report as mkr
import tree_report as tr
from heor_tree_engine import cma_cba


def _text(bundle):
    with zipfile.ZipFile(io.BytesIO(bundle.docx_bytes)) as zf:
        return zf.read("word/document.xml").decode("utf-8")


def _tree_input(settings):
    return tr.TreeReportInput(
        model_name="T", document={"nodes": []}, parameters={}, paths=[],
        strategy_order=["Std", "New"],
        strategy_totals={"Std": {"cost": 100.0, "effect": 1.0}, "New": {"cost": 300.0, "effect": 1.5}},
        cost_symbol="$", effect_symbol="QALY", cost_decimals=2, effect_decimals=3,
        extra_outcome_labels={}, baseline_strategy="Std", maximize_effect=True,
        willingness_to_pay=50000.0, cost_discount_rate=0.0, effect_discount_rate=0.0,
        analysis_settings=settings,
    )


def test_tree_report_cba_and_cma():
    b = tr.build_tree_report(_tree_input({"type": "CBA", "cba_value_per_effect": 1000.0}))
    assert "Cost-benefit analysis" in _text(b)
    b = tr.build_tree_report(_tree_input({"type": "CMA", "cma_tolerance_pct": 1.0}))
    t = _text(b)
    assert "Cost-minimization analysis" in t and "WARNING" in t  # effects differ 40%
    b = tr.build_tree_report(_tree_input(None))
    assert "Cost-minimization" not in _text(b)


def test_report_section_values():
    df = cma_cba.report_section({"type": "CBA", "cba_value_per_effect": 1000.0},
                                ["Std", "New"], [100.0, 300.0], [1.0, 1.5], True, "Std")
    new = df[df["Strategy"] == "New"].iloc[0]
    assert new["Incremental benefit"] == 500.0 and new["Incremental cost"] == 200.0
    assert new["BCR"] == 2.5 and cma_cba.report_section({"type": "EV"}, ["a"], [1], [1], True, "a") is None
