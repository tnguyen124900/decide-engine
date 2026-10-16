import math

import numpy as np

from heor_tree_engine import psm_extrapolation as ex
from heor_tree_engine import psm_km_fit as kmf


def _weibull_data(n=400, shape=1.3, scale=2.0, censor=3.0, seed=7, unit_factor=1.0):
    rng = np.random.default_rng(seed)
    t = scale * rng.weibull(shape, n)
    c = rng.uniform(0.5 * censor, censor, n)
    time = np.minimum(t, c) * unit_factor
    ev = (t <= c).astype(float)
    return kmf.KMData(time, ev, n, int(ev.sum()), "patient-level")


def test_exponential_landmarks_median_and_rmst_closed_form() -> None:
    p = {"rate": 0.4}
    s = ex.survival_years("exponential", p, "year", [1, 5])
    assert abs(s[0] - math.exp(-0.4)) < 1e-12 and abs(s[1] - math.exp(-2.0)) < 1e-12
    assert abs(ex.median_years("exponential", p, "year") - math.log(2) / 0.4) < 1e-3
    assert abs(ex.restricted_mean_years("exponential", p, "year", 10) - (1 - math.exp(-4.0)) / 0.4) < 1e-8
    # same curve typed in months: rate per month = 0.4/12
    pm = {"rate": 0.4 / 12}
    assert abs(ex.restricted_mean_years("exponential", pm, "month", 10) - (1 - math.exp(-4.0)) / 0.4) < 1e-8


def test_weibull_hazard_matches_analytic() -> None:
    p = {"shape": 1.5, "scale": 4.0}
    t = np.array([0.5, 2.0, 6.0])
    exact = (p["shape"] / p["scale"]) * (t / p["scale"]) ** (p["shape"] - 1)
    assert np.allclose(ex.hazard_curve("weibull", p, "year", t), exact, rtol=1e-4)


def test_extrapolate_table_on_simulated_weibull() -> None:
    data = _weibull_data()
    fits = kmf.fit_all(data, ("exponential", "weibull", "log_normal"))
    summary = ex.extrapolate(data, fits, "year", 10.0)
    assert [r.distribution for r in summary.rows] == [f.distribution for f in fits if f.params]
    best = summary.rows[0]
    assert best.distribution == "weibull"
    # Weibull true mean to 10y by fine integration
    x = np.linspace(0, 10, 100001)
    truth = float(np.trapezoid(np.exp(-((x / 2.0) ** 1.3)), x)) if hasattr(np, "trapezoid") else float(np.trapz(np.exp(-((x / 2.0) ** 1.3)), x))
    assert abs(best.rmst_horizon_years - truth) / truth < 0.08
    assert 0.0 <= best.extrapolation_share <= 1.0
    assert best.rmst_followup_years <= best.rmst_horizon_years
    assert abs(summary.km_rmst_followup_years - best.rmst_followup_years) / best.rmst_followup_years < 0.1
    assert set(best.survival_at) == set(ex.DEFAULT_LANDMARKS_YEARS)


def test_unit_conversion_gives_same_years() -> None:
    d_year = _weibull_data(unit_factor=1.0)
    d_month = _weibull_data(unit_factor=12.0)
    r_year = ex.extrapolate(d_year, kmf.fit_all(d_year, ("weibull",)), "year", 10.0).rows[0]
    r_month = ex.extrapolate(d_month, kmf.fit_all(d_month, ("weibull",)), "month", 10.0).rows[0]
    assert abs(r_year.rmst_horizon_years - r_month.rmst_horizon_years) < 5e-3
    assert abs(r_year.survival_at[5.0] - r_month.survival_at[5.0]) < 1e-3


def test_life_table_parse_and_background_survival() -> None:
    bg = ex.parse_life_table("age,qx\n60,0.02\n70,0.05\n80,0.12")
    s = ex.background_survival(bg, 60.0, [0.0, 5.0, 10.0])
    assert abs(s[0] - 1.0) < 1e-12
    assert abs(s[1] - (1 - 0.02) ** 5) < 1e-4
    assert abs(s[2] - (1 - 0.02) ** 10) < 2e-3  # step at age 70 begins right at year 10


def test_life_table_errors() -> None:
    for bad in ("60,0.02", "60,0.02\n55,0.03", "60,2\n70,0.1"):
        try:
            ex.parse_life_table(bad)
        except ValueError:
            continue
        raise AssertionError(bad)


def test_background_floor_lowers_mean_and_never_raises_survival() -> None:
    bg = ex.BackgroundMortality([60.0, 100.0], [0.05, 0.05])
    p = {"rate": 0.01}  # disease hazard below the background hazard
    t, s = ex.background_adjusted_survival("exponential", p, "year", bg, 60.0, 20.0, cutoff_years=2.0)
    s_unadj = ex.survival_years("exponential", p, "year", t)
    assert np.all(s <= s_unadj + 1e-12)
    expected_tail = math.exp(-0.01 * 2.0) * (1 - 0.05) ** (20.0 - 2.0)
    assert abs(s[-1] - expected_tail) < 2e-3
    assert ex.background_adjusted_mean_years("exponential", p, "year", bg, 60.0, 20.0, 2.0) < ex.restricted_mean_years("exponential", p, "year", 20.0)


def test_plausibility_flags() -> None:
    data = _weibull_data()
    fits = kmf.fit_all(data, ("exponential", "weibull"))
    summary = ex.extrapolate(data, fits, "year", 30.0)
    bg = ex.BackgroundMortality([50.0, 100.0], [0.3, 0.3])
    issues = ex.plausibility_checks(summary, fits, bg=bg, start_age=50.0, max_extrapolation_share=0.02)
    assert any(i.code == "extrapolation_share" for i in issues)
    # a curve with a very low hazard outlives a 30%/yr background mortality
    low = ex.ExtrapolationSummary("year", 3.0, 30.0, 1.0, 0.5, [ex.ExtrapolationRow("exponential", {"rate": 0.01}, 0, 0, survival_at_horizon=0.7)])
    issues2 = ex.plausibility_checks(low, [kmf.FitResult("exponential", {"rate": 0.01}, converged=True)], bg=bg, start_age=50.0)
    codes = {i.code for i in issues2}
    assert "above_general_population" in codes and "alive_at_horizon" in codes
    # PFS above OS
    issues3 = ex.plausibility_checks(low, [kmf.FitResult("exponential", {"rate": 0.01}, converged=True)], pfs_params=("exponential", {"rate": 0.001}, "year"))
    assert any(i.code == "pfs_above_os" for i in issues3)


def test_library_add_select_remove() -> None:
    curve = {"distribution": "weibull", "params": {"shape": 1.0, "scale": 2.0}, "time_unit": "year", "fit_info": {"x": 1}}
    ex.library_add(curve, "Weibull base", {"distribution": "weibull", "params": {"shape": 1.3, "scale": 2.1}, "aic": 100.0, "bic": 105.0}, "month", "best AIC")
    ex.library_add(curve, "Exp", {"distribution": "exponential", "params": {"rate": 0.3}}, "year")
    ex.library_add(curve, "Exp", {"distribution": "exponential", "params": {"rate": 0.25}}, "year", "updated")
    assert [e["name"] for e in curve["library"]] == ["Weibull base", "Exp"] and curve["library"][1]["params"]["rate"] == 0.25
    ex.library_select(curve, "Weibull base")
    assert curve["distribution"] == "weibull" and curve["time_unit"] == "month" and curve["library_selected"] == "Weibull base"
    assert "fit_info" not in curve
    assert ex.library_remove(curve, "Weibull base") and "library_selected" not in curve
    for fn in (lambda: ex.library_select(curve, "nope"), lambda: ex.library_add(curve, " ", {"distribution": "weibull", "params": {"shape": 1}}, "year"),
               lambda: ex.library_add(curve, "x", {"distribution": "weibull", "params": {}}, "year")):
        try:
            fn()
        except ValueError:
            continue
        raise AssertionError


def test_library_seeded_from_fit_info() -> None:
    curve = {"time_unit": "year", "fit_info": {"time_unit": "month", "candidates": [
        {"distribution": "weibull", "params": {"shape": 1.2, "scale": 20.0}, "aic": 1.0, "bic": 2.0},
        {"distribution": "exponential", "params": {}, "aic": 3.0, "bic": 4.0}]}}
    assert ex.library_from_fit_info(curve) == 1
    assert ex.library_from_fit_info(curve) == 0
    assert curve["library"][0]["time_unit"] == "month"


def test_km_steps_rmst_years_and_candidates_with_mixed_units() -> None:
    steps = [[6.0, 0.8], [12.0, 0.5], [24.0, 0.2]]  # months
    follow, area, s_end = ex.km_steps_rmst_years(steps, "month")
    assert abs(follow - 2.0) < 1e-12 and abs(s_end - 0.2) < 1e-12
    assert abs(area - (6 * 1.0 + 6 * 0.8 + 12 * 0.5) / 12.0) < 1e-12
    cands = [
        ex.Candidate("Exp (year)", "exponential", {"rate": 0.4}, "year"),
        ex.Candidate("Exp (month)", "exponential", {"rate": 0.4 / 12}, "month", converged=False),
    ]
    summary = ex.extrapolate_candidates(cands, 10.0, follow, area, s_end)
    a, b = summary.rows
    assert abs(a.rmst_horizon_years - b.rmst_horizon_years) < 1e-9 and a.name == "Exp (year)"
    issues = ex.plausibility_checks(summary, [])
    assert [i.code for i in issues if i.code == "not_converged"] == ["not_converged"]
    assert issues[0].message.startswith("Exp (month)") or any(i.message.startswith("Exp (month)") for i in issues)
