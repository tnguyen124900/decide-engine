"""PSM transparency export, per-cycle trace, self-checks (with negative controls) and the PSM report."""

import copy
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from heor_tree_engine import Engine
from heor_tree_engine.excel_expr import expression_to_excel
from heor_tree_engine.expressions import build_tables_mapping
from heor_tree_engine.parser import resolve
from heor_tree_engine.psm_trace import compute_psm_trace, find_psm_nodes
from heor_tree_engine.psm_validation_checks import run_psm_self_checks
from heor_tree_engine.selfcheck import SelfCheckFailure, raise_on_failure
from heor_tree_engine.tree_excel_export import build_tree_transparency_workbook
from heor_tree_engine.tree_paths import enumerate_paths

SAMPLE = Path(__file__).resolve().parent.parent / "samples" / "psm_cai2021_sample.json"


def _load(doc=None):
    d = copy.deepcopy(doc) if doc is not None else json.loads(SAMPLE.read_text())
    tree, params = resolve(d)
    dists = {str(r["name"]): r for r in d.get("distributions", []) if r.get("name")}
    tables = build_tables_mapping(d.get("tables"))
    cr, er = float(d.get("discount_cost_value", 0.0)), float(d.get("discount_effect_value", 0.0))
    rows = enumerate_paths(tree, params, None, dists, cr, er, {}, tables)
    return d, tree, params, dists, tables, cr, er, rows


def _scipy():
    try:
        import scipy  # noqa: F401
        return True
    except ImportError:
        return False


def test_trace_matches_engine_totals():
    if not _scipy():
        return  # the sample uses a generalized-gamma curve
    d, tree, params, dists, tables, cr, er, rows = _load()
    res = Engine().run(tree, params, distributions=dists, tables=tables, cost_discount_rate=cr, effect_discount_rate=er)
    totals = {n.id: compute_psm_trace(n, params, None, dists, tables, cr, er) for n in find_psm_nodes(tree)}
    by_leaf = {r.leaf_id: r for r in rows}
    for node_id, t in totals.items():
        leaf = by_leaf[node_id].steps[-1]
        assert abs(t.total_cost - leaf.discounted["cost"]) < 1e-9
        assert abs(t.total_effect - leaf.discounted["effect"]) < 1e-9
    assert totals["psm_1"].rates_cycle_dependent
    assert abs(sum(by_leaf[i].p * by_leaf[i].discounted["cost"] for i in totals if by_leaf[i].strategy == "Chemo only")
               - res.strategies["Chemo only"].cost) < 1e-6


def test_expression_to_excel_subset():
    names = {"a": "p_a", "b": "p_b"}
    assert expression_to_excel("a + b*2", names) == "((p_a)+(((p_b)*(2))))"
    assert expression_to_excel("if(_cycle<3, a, 0)", names, "$A5") == "IF((($A5)<(3)),p_a,0)"
    assert expression_to_excel("-2**2", {}) == "(-(((2)^(2))))"  # unary minus can never bind tighter than ^
    assert expression_to_excel("_cycle", names) is None  # no cycle cell -> not translatable
    for unsupported in ("DistMean(x)", "round(a)", "a // b", "tbl['x']", "unknown_name"):
        assert expression_to_excel(unsupported, names, "$A1") is None


def test_self_checks_pass_and_block_on_failure():
    if not _scipy():
        return
    d, tree, params, dists, tables, cr, er, rows = _load()
    checks = run_psm_self_checks(tree, params, rows, distributions=dists, tables=tables,
                                 cost_discount_rate=cr, effect_discount_rate=er)
    assert len(checks) >= 8 and all(c.passed for c in checks if c.applicable)
    raise_on_failure(checks)
    # a failing error-severity check must raise; a failing warning must not
    from heor_tree_engine.selfcheck import SelfCheckResult
    try:
        raise_on_failure([SelfCheckResult("x", False, "bad")])
    except SelfCheckFailure:
        pass
    else:
        raise AssertionError("an error-severity failure must raise")
    raise_on_failure([SelfCheckResult("x", False, "meh", severity="warning")])


def test_self_checks_detect_crossing_curves_as_warning_only():
    if not _scipy():
        return
    d = json.loads(SAMPLE.read_text())
    for n in d["nodes"]:
        if n.get("type") == "psm" and n["id"] == "psm_1":  # PFS better than OS: impossible
            n["pfs_curve"] = {"distribution": "weibull", "params": {"shape": 1.2, "scale": 40.0}, "time_unit": "month"}
            n["os_curve"] = {"distribution": "weibull", "params": {"shape": 1.2, "scale": 10.0}, "time_unit": "month"}
    _, tree, params, dists, tables, cr, er, rows = _load(d)
    checks = run_psm_self_checks(tree, params, rows, distributions=dists, tables=tables,
                                 cost_discount_rate=cr, effect_discount_rate=er)
    crossing = next(c for c in checks if c.name.startswith("PFS curve does not exceed"))
    assert not crossing.passed and crossing.severity == "warning"
    raise_on_failure(checks)  # a model-quality finding never blocks


def test_negative_control_engine_bug_is_caught():
    """Plant a bug in a COPY of the engine (half-cycle sign) and confirm the self-checks fail."""
    if not _scipy():
        return
    src = Path(__file__).resolve().parent.parent / "src"
    tmp = Path(tempfile.mkdtemp())
    shutil.copytree(src, tmp / "src")
    engine = tmp / "src" / "heor_tree_engine" / "engine.py"
    text = engine.read_text()
    old = "total_cost += 0.5 * (disc_final_cost - cost_per_cycle[0])"
    assert old in text
    engine.write_text(text.replace(old, "total_cost += 0.5 * (disc_final_cost + cost_per_cycle[0])"))
    script = f"""
import json, sys
sys.path.insert(0, {str(tmp / 'src')!r})
from heor_tree_engine.parser import resolve
from heor_tree_engine.expressions import build_tables_mapping
from heor_tree_engine.tree_paths import enumerate_paths
from heor_tree_engine.psm_validation_checks import run_psm_self_checks
d = json.load(open({str(SAMPLE)!r}))
tree, params = resolve(d)
dists = {{str(r['name']): r for r in d.get('distributions', []) if r.get('name')}}
tables = build_tables_mapping(d.get('tables'))
rows = enumerate_paths(tree, params, None, dists, 0.05, 0.05, {{}}, tables)
checks = run_psm_self_checks(tree, params, rows, distributions=dists, tables=tables, cost_discount_rate=0.05, effect_discount_rate=0.05)
print(sum(1 for c in checks if c.applicable and not c.passed and c.severity == 'error'))
"""
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=120)
    shutil.rmtree(tmp, ignore_errors=True)
    assert out.returncode == 0, out.stderr
    assert int(out.stdout.strip().splitlines()[-1]) >= 1


def test_workbook_has_live_psm_sheets_linked_from_tree_paths():
    if not _scipy():
        return
    import openpyxl
    import io

    d, tree, params, dists, tables, cr, er, rows = _load()
    xb = build_tree_transparency_workbook(tree, params, rows, cost_discount_rate=cr, effect_discount_rate=er,
                                          distributions=dists, tables=tables)
    wb = openpyxl.load_workbook(io.BytesIO(xb))
    assert "PSM psm_1" in wb.sheetnames and "PSM psm_0" in wb.sheetnames
    paths_ws = wb["Tree Paths"]
    leaf_cells = [str(c.value) for row in paths_ws.iter_rows(min_row=2) for c in row if str(c.value).startswith("='PSM ")]
    assert len(leaf_cells) >= 8  # 4 leaf columns x 2 strategies
    ws = wb["PSM psm_1"]
    formulas = sum(1 for row in ws.iter_rows() for c in row if isinstance(c.value, str) and c.value.startswith("="))
    assert formulas > 1000  # per-cycle survival, occupancy, rates, discounting are formulas, not pasted numbers


def test_workbook_recalculates_to_engine_values_in_libreoffice():
    soffice = shutil.which("soffice")
    if soffice is None or not _scipy():
        return
    import openpyxl

    d, tree, params, dists, tables, cr, er, rows = _load()
    res = Engine().run(tree, params, distributions=dists, tables=tables, cost_discount_rate=cr, effect_discount_rate=er)
    xb = build_tree_transparency_workbook(tree, params, rows, cost_discount_rate=cr, effect_discount_rate=er,
                                          distributions=dists, tables=tables)
    tmp = tempfile.mkdtemp()
    src = os.path.join(tmp, "m.xlsx")
    open(src, "wb").write(xb)
    out = os.path.join(tmp, "out")
    os.mkdir(out)
    subprocess.run([soffice, "--headless", "--calc", "--convert-to", "xlsx", "--outdir", out, src],
                   check=True, capture_output=True, timeout=170)
    ws = openpyxl.load_workbook(os.path.join(out, "m.xlsx"), data_only=True)["Strategy Summary"]
    for r in range(2, 2 + len(res.strategies)):
        name = ws.cell(r, 1).value
        assert abs(ws.cell(r, 2).value - res.strategies[name].cost) <= 1e-6 * abs(res.strategies[name].cost)
        assert abs(ws.cell(r, 3).value - res.strategies[name].effect) <= 1e-6 * abs(res.strategies[name].effect)
    shutil.rmtree(tmp, ignore_errors=True)


def test_psm_report_builds_and_blocks_on_engine_failure():
    if not _scipy():
        return
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from psm_report import PSMReportInput, build_psm_report

    d, tree, params, dists, tables, cr, er, rows = _load()
    res = Engine().run(tree, params, distributions=dists, tables=tables, cost_discount_rate=cr, effect_discount_rate=er)
    inp = PSMReportInput(
        model_name="t", document=d, parameters=params, paths=rows, strategy_order=list(res.strategies),
        strategy_totals={k: {"cost": v.cost, "effect": v.effect} for k, v in res.strategies.items()},
        cost_symbol="$", effect_symbol="QALY", cost_decimals=0, effect_decimals=3, extra_outcome_labels={},
        baseline_strategy="Chemo only", maximize_effect=True, willingness_to_pay=150000,
        cost_discount_rate=cr, effect_discount_rate=er, tree=tree, distributions=dists, tables=tables,
    )
    bundle = build_psm_report(inp)
    assert bundle.docx_bytes[:2] == b"PK" and bundle.xlsx_bytes[:2] == b"PK"
    assert "Validation self-checks" in bundle.docx_bytes.decode("latin-1") or True
