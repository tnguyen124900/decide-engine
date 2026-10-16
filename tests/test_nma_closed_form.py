"""Hand-derived closed-form checks of the frequentist NMA (independent of netmeta).

Triangle A-B-C, three two-arm studies, all contrast SE = 1:
    s1: B vs A = 1,  s2: C vs B = 2,  s3: C vs A = 5.

Fixed effect.  B vs A: direct 1 (weight 1), indirect 5-2 = 3 (weight 1/2)
  -> d_BA = (1 + 1.5)/1.5 = 5/3, var = 1/1.5.   C vs A: direct 5 (1), indirect 1+2 = 3 (1/2)
  -> d_CA = (5 + 1.5)/1.5 = 13/3.   C vs B = 8/3.
Residuals (-2/3, -2/3, 2/3): Q = 3 * 4/9 = 4/3 on df = 3 - 2 = 1.
DerSimonian-Laird: all weights 1 so tr(P) = df = 1, tau2 = (4/3 - 1)/1 = 1/3.
Random effect: weights 1/(1+1/3) = 3/4 for direct, indirect variance 2*(4/3) -> weight 3/8; the
estimates are unchanged by symmetry and var(d_BA) = 1/(3/4 + 3/8) = 8/9.
"""

import math

import pytest

from heor_tree_engine.nma_freq import run_nma_frequentist
from heor_tree_engine.nma_parser import resolve_nma


def _run(estimates=(1.0, 2.0, 5.0)):
    pairs = [("B", "A"), ("C", "B"), ("C", "A")]
    studies = [{"id": f"s{i}", "contrasts": [{"treatment": t, "versus": v, "estimate": e, "se": 1.0}]}
               for i, ((t, v), e) in enumerate(zip(pairs, estimates), 1)]
    return run_nma_frequentist(resolve_nma({
        "spec": {"outcome": "contrast", "measure": "MD", "reference": "A"}, "studies": studies}))


def test_fixed_effect_triangle():
    result = _run()
    assert not result.errors
    est = {e.treatment: e for e in result.fixed.effects}
    assert est["B"].estimate == pytest.approx(5 / 3, abs=1e-12)
    assert est["C"].estimate == pytest.approx(13 / 3, abs=1e-12)
    assert est["B"].se == pytest.approx(math.sqrt(1 / 1.5), abs=1e-12)
    cb = result.fixed.league[("C", "B")]
    assert cb.estimate == pytest.approx(8 / 3, abs=1e-12)


def test_heterogeneity_and_random_effects_triangle():
    result = _run()
    assert result.heterogeneity.q == pytest.approx(4 / 3, abs=1e-12)
    assert result.heterogeneity.tau2 == pytest.approx(1 / 3, abs=1e-12)
    est = {e.treatment: e for e in result.random.effects}
    assert est["B"].estimate == pytest.approx(5 / 3, abs=1e-12)
    assert est["B"].se == pytest.approx(math.sqrt(8 / 9), abs=1e-9)


def test_consistent_triangle_has_zero_q_and_tau2():
    result = _run((1.0, 2.0, 3.0))
    assert result.heterogeneity.q == pytest.approx(0.0, abs=1e-12)
    assert result.heterogeneity.tau2 == 0.0
    est = {e.treatment: e.estimate for e in result.random.effects}
    assert est["B"] == pytest.approx(1.0, abs=1e-12)
    assert est["C"] == pytest.approx(3.0, abs=1e-12)
