import math

import numpy as np

from heor_tree_engine.psm_km_fit import (
    fit_all,
    fit_distribution,
    parse_patient_data,
    reconstruct_from_km,
    survival,
    km_steps,
)


def _simulate(dist, params, n=1500, censor_rate=0.05, seed=1):
    rng = np.random.default_rng(seed)
    u = rng.uniform(size=n)
    if dist == "weibull":
        t = params["scale"] * (-np.log(u)) ** (1 / params["shape"])
    elif dist == "exponential":
        t = -np.log(u) / params["rate"]
    elif dist == "log_normal":
        t = np.exp(params["meanlog"] + params["sdlog"] * rng.standard_normal(n))
    elif dist == "log_logistic":
        t = params["scale"] * ((1 - u) / u) ** (1 / params["shape"])
    elif dist == "gompertz":
        t = np.log(1 - params["shape"] * np.log(u) / params["rate"]) / params["shape"]
    c = rng.exponential(1 / censor_rate, size=n)
    time = np.minimum(t, c)
    event = (t <= c).astype(float)
    return "\n".join(f"{a:.6f},{int(b)}" for a, b in zip(time, event))


def _close(est, true, tol):
    for k, v in true.items():
        assert abs(est[k] - v) <= tol * abs(v), (k, est[k], v)


def test_recovers_weibull():
    true = {"shape": 1.6, "scale": 8.0}
    fit = fit_distribution("weibull", parse_patient_data(_simulate("weibull", true)))
    assert fit.converged
    _close(fit.params, true, 0.06)


def test_recovers_each_closed_form_distribution():
    cases = {
        "exponential": {"rate": 0.2},
        "log_normal": {"meanlog": 1.5, "sdlog": 0.7},
        "log_logistic": {"shape": 2.2, "scale": 6.0},
        "gompertz": {"shape": 0.15, "rate": 0.05},
    }
    for dist, true in cases.items():
        fit = fit_distribution(dist, parse_patient_data(_simulate(dist, true, seed=7)))
        _close(fit.params, true, 0.12)
        assert fit.max_gap < 0.03, (dist, fit.max_gap)


def test_aic_prefers_true_model_over_exponential():
    data = parse_patient_data(_simulate("weibull", {"shape": 2.5, "scale": 6.0}))
    fits = {f.distribution: f for f in fit_all(data, ("exponential", "weibull"))}
    assert fits["weibull"].aic < fits["exponential"].aic - 20
    assert fit_all(data, ("exponential", "weibull"))[0].distribution == "weibull"


def test_matches_independent_loglik():
    # exponential MLE has a closed form: rate = events / total time
    data = parse_patient_data(_simulate("exponential", {"rate": 0.3}))
    fit = fit_distribution("exponential", data)
    assert abs(fit.params["rate"] - data.events.sum() / data.times.sum()) < 1e-5


def test_survival_formulas_match_engine():
    from heor_tree_engine.validation import _psm_survival_unchecked
    from heor_tree_engine.models import SurvivalCurve

    cases = [
        ("weibull", {"shape": 1.5, "scale": 8.0}),
        ("log_normal", {"meanlog": 1.2, "sdlog": 0.8}),
        ("log_logistic", {"shape": 2.0, "scale": 5.0}),
        ("gompertz", {"shape": 0.2, "rate": 0.05}),
        ("exponential", {"rate": 0.1}),
    ]
    for dist, p in cases:
        for t in (0.5, 3.0, 12.0):
            mine = float(survival(dist, p, np.array([t]))[0])
            ref = _psm_survival_unchecked(SurvivalCurve(dist, dict(p)), t, {})
            assert abs(mine - ref) < 1e-12, (dist, t, mine, ref)


def test_reconstruction_roundtrip_fits_close():
    true = {"shape": 1.5, "scale": 10.0}
    times = np.linspace(0, 24, 25)
    curve = "\n".join(f"{t},{math.exp(-((t / 10.0) ** 1.5)):.5f}" for t in times)
    risk = "0,300\n6,189\n12,80\n18,27\n24,7"
    data = reconstruct_from_km(curve, risk)
    fit = fit_distribution("weibull", data)
    _close(fit.params, true, 0.15)


def test_bad_input_rejected():
    for bad in ("1,1\n2,0", "t,e\n1,2\n2,1\n3,1\n4,1\n5,1"):
        try:
            parse_patient_data(bad)
        except ValueError:
            continue
        raise AssertionError("should reject")


def test_generalized_gamma_if_scipy():
    try:
        import scipy  # noqa: F401
    except ImportError:
        return
    data = parse_patient_data(_simulate("log_normal", {"meanlog": 1.5, "sdlog": 0.7}, seed=3))
    ln = fit_distribution("log_normal", data)
    gg = fit_distribution("generalized_gamma", data)
    assert gg.loglik >= ln.loglik - 1e-3  # GG nests log-normal
    assert gg.max_gap < 0.03
