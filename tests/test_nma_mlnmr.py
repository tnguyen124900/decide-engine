"""ML-NMR (C2-b): analytic derivatives, QMC integration, closed form without covariates, and recovery of the
effect modifier when aggregate studies are generated from patient-level data."""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from heor_tree_engine import nma_mlnmr  # noqa: E402
from heor_tree_engine.nma_mlnmr import AggArm, IPDArm, halton, run_mlnmr  # noqa: E402


def test_halton_integrates_a_smooth_function():
    from heor_tree_engine.nma_parser import _normal_quantile

    u = halton(512, 1)[:, 0]
    x = np.array([_normal_quantile(v) for v in u])
    got = float(np.mean(1 / (1 + np.exp(-(0.3 + 0.8 * x)))))
    nodes, weights = np.polynomial.hermite_e.hermegauss(60)
    exact = float(np.sum(weights / math.sqrt(2 * math.pi) / (1 + np.exp(-(0.3 + 0.8 * nodes)))))
    assert got == pytest.approx(exact, abs=1e-3)


def test_no_covariates_single_study_is_the_two_by_two_log_or():
    agg = [AggArm("s1", "A", 30, 100, np.zeros(0), np.zeros(0)), AggArm("s1", "B", 50, 110, np.zeros(0), np.zeros(0))]
    res = run_mlnmr([], agg, [], [], [], "A")
    assert not res.errors
    expected = math.log((50 / 60) / (30 / 70))
    assert res.theta[res.names.index("d[B]")] == pytest.approx(expected, abs=1e-6)
    assert res.se("d[B]") == pytest.approx(math.sqrt(1 / 30 + 1 / 70 + 1 / 50 + 1 / 60), rel=1e-4)


def _simulate_patients(rng, study, n, mean_x, arms, beta=0.5, gamma=0.6, d=None, mu=-0.4):
    d = d or {"A": 0.0, "B": -0.7}
    out = {}
    for arm in arms:
        x = rng.normal(mean_x, 1.0, n)
        eta = mu + beta * x + (d[arm] + gamma * x if arm != "A" else 0.0)
        out[arm] = (x[:, None], (rng.random(n) < 1 / (1 + np.exp(-eta))).astype(float))
    return out


def test_recovers_the_effect_modifier_from_ipd_and_aggregate_studies():
    rng = np.random.default_rng(11)
    ipd, agg = [], []
    for study, mean_x in (("ipd1", 0.0), ("ipd2", 0.6)):
        for arm, (x, y) in _simulate_patients(rng, study, 3000, mean_x, ("A", "B")).items():
            ipd.append(IPDArm(study, arm, x, y))
    for study, mean_x in (("agg1", 1.0), ("agg2", -0.4)):
        for arm, (x, y) in _simulate_patients(rng, study, 3000, mean_x, ("A", "B")).items():
            agg.append(AggArm(study, arm, float(y.sum()), float(len(y)), np.array([x.mean()]), np.array([x.std(ddof=1)])))
    res = run_mlnmr(ipd, agg, ["x"], [False], ["x"], "A", n_points=512)
    assert not res.errors and not res.warnings
    for name, truth in (("beta[x]", 0.5), ("gamma[x]", 0.6)):
        i = res.names.index(name)
        assert abs(res.theta[i] - truth) < 3 * res.se(name) + 0.02, (name, res.theta[i], res.se(name))
    # effect at x = 1: d + gamma * 1 = -0.1; at x = 0 it is -0.7
    at0, se0 = res.effect("B", [0.0])
    at1, se1 = res.effect("B", [1.0])
    assert abs(at0 - (-0.7)) < 3 * se0 + 0.02 and abs(at1 - (-0.1)) < 3 * se1 + 0.02
    assert res.effect("A", [0.0]) == (0.0, 0.0)


def test_analytic_gradient_matches_finite_differences():
    rng = np.random.default_rng(3)
    agg = []
    for study, m in (("a", 0.2), ("b", 0.8)):
        for arm, e in (("A", 40), ("B", 55)):
            agg.append(AggArm(study, arm, e, 120, np.array([m]), np.array([1.0])))
    x = rng.normal(0, 1, (400, 1))
    ipd = [IPDArm("c", "A", x, (rng.random(400) < 0.4).astype(float)), IPDArm("c", "B", x + 0.1, (rng.random(400) < 0.5).astype(float))]
    res = run_mlnmr(ipd, agg, ["x"], [False], ["x"], "A")
    assert not res.errors
    ev = res._evaluate
    theta = res.theta + 0.05 * np.random.default_rng(1).standard_normal(len(res.theta))
    ll, grad, hess = ev(theta)
    eps = 1e-6
    for i in range(len(theta)):
        up, dn = theta.copy(), theta.copy()
        up[i] += eps
        dn[i] -= eps
        fu, gu, _ = ev(up)
        fd, gd, _ = ev(dn)
        assert grad[i] == pytest.approx((fu - fd) / (2 * eps), rel=1e-4, abs=1e-4)
        np.testing.assert_allclose(hess[:, i], (gu - gd) / (2 * eps), rtol=1e-4, atol=1e-3)
    assert np.max(np.abs(ev(res.theta)[1])) < 1e-4  # stationary at the fit


def test_binary_covariate_and_errors():
    rng = np.random.default_rng(8)
    x = rng.binomial(1, 0.4, (500, 1)).astype(float)
    ipd = [IPDArm("s", "A", x, (rng.random(500) < 0.3).astype(float)), IPDArm("s", "B", x, (rng.random(500) < 0.45).astype(float))]
    agg = [AggArm("t", "A", 35, 100, np.array([0.5]), np.array([0.5])), AggArm("t", "B", 55, 100, np.array([0.5]), np.array([0.5]))]
    res = run_mlnmr(ipd, agg, ["male"], [True], ["male"], "A")
    assert not res.errors and res.effect("B", [1.0])[1] > 0
    assert run_mlnmr(ipd, agg, ["male"], [True], ["age"], "A").errors
    assert run_mlnmr(ipd, agg, ["male"], [True], ["male"], "Z").errors
