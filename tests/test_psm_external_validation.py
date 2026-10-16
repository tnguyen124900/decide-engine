"""External-method validation of the PSM node.

The reference here is the method published in the hesim "Partitioned survival models" vignette (state probabilities
= S1, S2 - S1, 1 - S2; QALY/cost = discounted integral of state probability x state value) and used by NICE DSU
TSD 19: a *continuous-time* integral.  DECIDE sums per cycle (trapezoid when the half-cycle correction is on), so
the engine must (a) match the closed form for exponential curves, (b) converge to the continuous integral as the
cycle length shrinks, at second order, for every supported closed-form family.  The references below are
written independently of the engine (own formulas, own integrator) and only use numpy/math.

Survival parameterisations are the flexsurv ones (weibull shape/scale, lnorm meanlog/sdlog, llogis shape/scale,
gompertz shape/rate), the same as hesim uses, so numbers carry over to R without conversion.
Not covered: generalized gamma (needs scipy for an independent reference) and the R-generated numbers
themselves (R is not available in the build environment) -- see tests/reference/README.md.
"""
import math

import numpy as np

from heor_tree_engine import Branch, TerminalNode, expected_cost, expected_utility
from heor_tree_engine.models import PSMNode, SurvivalCurve

COST = {"pre_progression": 1200.0, "post_progression": 2500.0, "dead": 0.0}  # per YEAR
UTIL = {"pre_progression": 0.80, "post_progression": 0.55, "dead": 0.0}  # QALY weight per year
HORIZON_YEARS = 10


def _ref_surv(dist, p, t):
    t = np.asarray(t, dtype=float)
    if dist == "exponential":
        return np.exp(-p["rate"] * t)
    if dist == "weibull":
        return np.exp(-((t / p["scale"]) ** p["shape"]))
    if dist == "log_normal":
        z = (np.log(np.maximum(t, 1e-300)) - p["meanlog"]) / (p["sdlog"] * math.sqrt(2))
        out = 0.5 * np.array([math.erfc(v) for v in np.atleast_1d(z)])
        return np.where(t <= 0, 1.0, out)
    if dist == "log_logistic":
        return 1.0 / (1.0 + (t / p["scale"]) ** p["shape"])
    if dist == "gompertz":
        return np.exp(-(p["rate"] / p["shape"]) * np.expm1(p["shape"] * t))
    raise ValueError(dist)


def _node(dist, pfs, os_, cpy, discounted_node=True):
    cycles = HORIZON_YEARS * cpy
    mk = lambda i, role: Branch(
        label=role, role=role,
        child=TerminalNode(id=f"{role}_{i}", cost=COST[role] / cpy, effect=UTIL[role] / cpy),
    )
    return PSMNode(
        id="psm", pfs_curve=SurvivalCurve(dist, pfs), os_curve=SurvivalCurve(dist, os_),
        cycles=cycles, cycles_per_year=float(cpy), half_cycle_correction=True,
        branches=[mk(0, "pre_progression"), mk(1, "post_progression"), mk(2, "dead")],
    )


def _ref_totals(dist, pfs, os_, rate_c, rate_e, n=200001):
    """Continuous-time reference: Simpson on a very fine grid, continuous-in-years discounting (1+r)^-t."""
    t = np.linspace(0.0, HORIZON_YEARS, n)
    s1, s2 = _ref_surv(dist, pfs, t), _ref_surv(dist, os_, t)
    occ = {"pre_progression": s1, "post_progression": np.maximum(0.0, s2 - s1), "dead": 1.0 - s2}
    h = t[1] - t[0]
    w = np.ones(n); w[1:-1:2] = 4; w[2:-1:2] = 2
    simpson = lambda y: h / 3.0 * float(np.sum(w * y))
    cost = simpson(sum(occ[r] * COST[r] for r in occ) * (1 + rate_c) ** (-t))
    qaly = simpson(sum(occ[r] * UTIL[r] for r in occ) * (1 + rate_e) ** (-t))
    return cost, qaly


def _engine(node, rate_c, rate_e):
    kw = dict(cost_discount_rate=rate_c, effect_discount_rate=rate_e)
    return expected_cost(node, **kw), expected_utility(node, **kw)


CASES = [
    ("exponential", {"rate": 0.35}, {"rate": 0.15}),
    ("weibull", {"shape": 1.4, "scale": 3.0}, {"shape": 1.3, "scale": 7.0}),
    ("log_normal", {"meanlog": 0.6, "sdlog": 0.7}, {"meanlog": 1.4, "sdlog": 0.8}),
    ("log_logistic", {"shape": 2.2, "scale": 2.5}, {"shape": 2.0, "scale": 6.0}),
    ("gompertz", {"shape": 0.15, "rate": 0.20}, {"shape": 0.12, "rate": 0.07}),
]


def test_exponential_matches_closed_form() -> None:
    a, b, r = 0.35, 0.15, 0.035
    d = math.log(1 + r)
    H = HORIZON_YEARS
    I = lambda k: (1 - math.exp(-(k + d) * H)) / (k + d)  # integral of exp(-(k+d)t)
    pre, post = I(a), I(b) - I(a)
    exact_cost = COST["pre_progression"] * pre + COST["post_progression"] * post
    exact_qaly = UTIL["pre_progression"] * pre + UTIL["post_progression"] * post
    cost, qaly = _engine(_node("exponential", {"rate": a}, {"rate": b}, 365), r, r)
    assert abs(cost - exact_cost) / exact_cost < 1e-6
    assert abs(qaly - exact_qaly) / exact_qaly < 1e-6


def test_engine_matches_continuous_reference_for_each_distribution() -> None:
    for dist, pfs, os_ in CASES:
        ref_c, ref_q = _ref_totals(dist, pfs, os_, 0.035, 0.035)
        cost, qaly = _engine(_node(dist, pfs, os_, 365), 0.035, 0.035)
        assert abs(cost - ref_c) / ref_c < 2e-4, (dist, cost, ref_c)
        assert abs(qaly - ref_q) / ref_q < 2e-4, (dist, qaly, ref_q)


def test_undiscounted_matches_reference() -> None:
    for dist, pfs, os_ in CASES:
        ref_c, ref_q = _ref_totals(dist, pfs, os_, 0.0, 0.0)
        cost, qaly = _engine(_node(dist, pfs, os_, 365), 0.0, 0.0)
        assert abs(cost - ref_c) / ref_c < 2e-4, dist
        assert abs(qaly - ref_q) / ref_q < 2e-4, dist


def test_error_shrinks_at_second_order_with_cycle_length() -> None:
    """Trapezoid rule: halving the cycle length should cut the error ~4x."""
    for dist, pfs, os_ in CASES:
        ref_c, _ = _ref_totals(dist, pfs, os_, 0.035, 0.035)
        err = {cpy: abs(_engine(_node(dist, pfs, os_, cpy), 0.035, 0.035)[0] - ref_c) for cpy in (4, 8, 16)}
        assert err[8] < err[4] and err[16] < err[8], (dist, err)
        assert 2.5 < err[4] / err[8] < 6.0, (dist, err)
        assert 2.5 < err[8] / err[16] < 6.0, (dist, err)


def test_state_times_add_up_to_os_restricted_mean() -> None:
    """RMST identity quoted in the psm3mkv vignette: PF + PD = OS (restricted mean times, undiscounted)."""
    dist, pfs, os_ = CASES[1]
    t = np.linspace(0, HORIZON_YEARS, 200001)
    s1, s2 = _ref_surv(dist, pfs, t), _ref_surv(dist, os_, t)
    h = t[1] - t[0]
    trap = lambda y: h * (float(np.sum(y)) - 0.5 * (y[0] + y[-1]))
    pf, pd, osm = trap(s1), trap(np.maximum(0, s2 - s1)), trap(s2)
    assert abs(pf + pd - osm) < 1e-9
    # and the engine reproduces PF + PD time through a unit-reward model
    node = _node(dist, pfs, os_, 365)
    for br in node.branches:
        br.child.cost = 0.0
        br.child.effect = (1.0 / 365) if br.role != "dead" else 0.0
    _, qaly = _engine(node, 0.0, 0.0)
    assert abs(qaly - osm) / osm < 2e-4
