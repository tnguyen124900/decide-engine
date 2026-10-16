import math
import pytest
from types import SimpleNamespace as NS
from heor_tree_engine.risk_preference import analyze_paths, utility, inverse_utility


def row(s, p, cost, eff, leaf="terminal"):
    d = {"cost": cost, "effect": eff}
    return NS(strategy=s, p=p, discounted=d, undiscounted=d, leaf_type=leaf)


ROWS = [row("Safe", 1.0, 100, 1.0),                      # NMB(wtp=1000) = 900 sure
        row("Gamble", 0.5, 0, 1.8), row("Gamble", 0.5, 0, 0.0)]  # 1800 or 0, E = 900


def test_same_mean_risk_averse_prefers_safe():
    res = {r.strategy: r for r in analyze_paths(ROWS, 1000, "exponential", 500)}
    assert res["Safe"].expected_nmb == pytest.approx(900)
    assert res["Gamble"].expected_nmb == pytest.approx(900)
    assert res["Safe"].rank == 1 and res["Gamble"].rank == 2
    assert res["Safe"].risk_premium == pytest.approx(0, abs=1e-9)
    eu = 0.5 * (1 - math.exp(-1800 / 500)) + 0.5 * 0.0
    assert res["Gamble"].expected_utility == pytest.approx(eu)
    assert res["Gamble"].certainty_equivalent == pytest.approx(-500 * math.log(1 - eu))
    assert res["Gamble"].risk_premium > 0


def test_risk_seeking_prefers_gamble_and_neutral_ties():
    res = {r.strategy: r for r in analyze_paths(ROWS, 1000, "risk_seeking", 500)}
    assert res["Gamble"].rank == 1 and res["Gamble"].risk_premium < 0
    n = {r.strategy: r for r in analyze_paths(ROWS, 1000, "neutral")}
    assert n["Safe"].expected_utility == pytest.approx(n["Gamble"].expected_utility)


def test_large_R_approaches_risk_neutral():
    res = {r.strategy: r for r in analyze_paths(ROWS, 1000, "exponential", 1e9)}
    assert res["Gamble"].risk_premium == pytest.approx(0, abs=1e-2)


def test_inverse_roundtrip_and_errors():
    for k in ("exponential", "risk_seeking"):
        assert inverse_utility(utility(123.0, k, 400), k, 400) == pytest.approx(123.0)
    with pytest.raises(ValueError, match="R must be"):
        utility(1, "exponential", 0)
    with pytest.raises(ValueError, match="subtree"):
        analyze_paths([row("X", 1, 0, 0, leaf="psm")], 1000)
