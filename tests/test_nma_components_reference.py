"""Cross-check of the additive component NMA against netmeta::netcomb.

Reference produced by tests/reference/make_netcomb_reference.R (needs R); the
tests are skipped until netcomb_additive.json exists.
"""

import json
from pathlib import Path

import pytest

from heor_tree_engine.nma_components import run_component_nma
from heor_tree_engine.nma_parser import resolve_nma

REFERENCE = Path(__file__).parent / "reference" / "netcomb_additive.json"
CONTRASTS = [  # (study, treatment, versus, estimate, se) -- keep in sync with the R script
    ("s1", "X", "P", 0.5, 0.2), ("s2", "Y", "P", 0.9, 0.25), ("s3", "Z", "P", 0.3, 0.2),
    ("s4", "X+Y", "P", 1.6, 0.3), ("s5", "X+Z", "P", 0.9, 0.3), ("s6", "X+Y", "X", 0.8, 0.25),
    ("s7", "X+Z", "Z", 0.5, 0.3), ("s8", "X+Y", "Y", 0.5, 0.25), ("s9", "X", "P", 1.4, 0.3),
]
COMPONENTS = {"X+Y": "X + Y", "X+Z": "X + Z"}


def _model():
    return resolve_nma({
        "spec": {"outcome": "continuous", "measure": "MD", "reference": "P"},
        "treatments": [{"name": n} for n in ("P", "X", "Y", "Z", "X+Y", "X+Z")],
        "studies": [{"id": s, "contrasts": [{"treatment": t, "versus": v, "estimate": e, "se": se}]}
                    for s, t, v, e, se in CONTRASTS]})


def _reference():
    if not REFERENCE.exists():
        pytest.skip("netcomb_additive.json not generated yet - run tests/reference/make_netcomb_reference.R")
    return json.loads(REFERENCE.read_text())


def test_engine_runs_on_the_reference_data():
    out = run_component_nma(_model(), COMPONENTS, "random")
    assert not out.errors and out.components == ["X", "Y", "Z"]
    assert out.df_diff == 2  # 5 non-reference treatments - 3 components
    assert out.tau2 > 0  # the two X-vs-P studies disagree, so the random-effects check is not trivial


def test_matches_netcomb_additive_model():
    ref = _reference()
    for effects, suffix in (("random", "random"), ("fixed", "common")):
        out = run_component_nma(_model(), COMPONENTS, effects)
        est = {e.treatment: e for e in out.component_effects}
        for name, value in ref[f"components_{suffix}"].items():
            assert est[name].estimate == pytest.approx(value, abs=1e-5), (effects, name)
        for name, value in ref[f"se_components_{suffix}"].items():
            assert est[name].se == pytest.approx(value, abs=1e-5), (effects, name)
    out = run_component_nma(_model(), COMPONENTS, "random")
    assert out.tau2 == pytest.approx(ref["tau2"], abs=1e-6)
    assert out.q_additive == pytest.approx(ref["Q_additive"], abs=1e-5)
    assert out.q_diff == pytest.approx(ref["Q_diff"], abs=1e-5)
    assert out.df_diff == ref["df_diff"]
