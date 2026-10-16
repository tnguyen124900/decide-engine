"""Risk preference (utility) analysis for decision trees.

TreeAge's "Risk Preference Function" replaces expected value by expected
utility.  Here the outcome of every root-to-leaf path is its net monetary
benefit ``NMB = WTP * effect - cost`` (discounted), and a strategy is ranked
by ``EU = sum_i p_i * u(NMB_i)``.

Utility families (``risk_tolerance`` R > 0 is in the same money unit as NMB):

* ``exponential``  u(x) = 1 - exp(-x / R)   (constant absolute risk aversion)
* ``risk_seeking`` u(x) = exp(x / R) - 1    (the mirror image)
* ``neutral``      u(x) = x                  (expected value, for comparison)

The certainty equivalent CE = u^-1(EU) is the sure amount worth the same as the
gamble; the risk premium is E[NMB] - CE (positive for risk-averse, negative for
risk-seeking).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

UTILITY_KINDS = ("exponential", "risk_seeking", "neutral")


def utility(x: float, kind: str, risk_tolerance: float = 1.0) -> float:
    if kind == "neutral":
        return float(x)
    if risk_tolerance is None or not risk_tolerance > 0:
        raise ValueError("Risk tolerance R must be > 0")
    if kind == "exponential":
        return 1.0 - math.exp(-x / risk_tolerance)
    if kind == "risk_seeking":
        return math.exp(x / risk_tolerance) - 1.0
    raise ValueError(f"Unknown utility kind '{kind}'")


def inverse_utility(u: float, kind: str, risk_tolerance: float = 1.0) -> float:
    if kind == "neutral":
        return float(u)
    if risk_tolerance is None or not risk_tolerance > 0:
        raise ValueError("Risk tolerance R must be > 0")
    if kind == "exponential":
        if u >= 1.0:
            return math.inf
        return -risk_tolerance * math.log(1.0 - u)
    if kind == "risk_seeking":
        if u <= -1.0:
            return -math.inf
        return risk_tolerance * math.log(1.0 + u)
    raise ValueError(f"Unknown utility kind '{kind}'")


@dataclass
class RiskResult:
    strategy: str
    expected_nmb: float
    expected_utility: float
    certainty_equivalent: float
    risk_premium: float
    worst_nmb: float
    best_nmb: float
    p_total: float
    rank: int = 0


def analyze_paths(
    rows: Iterable[Any],
    wtp: float,
    kind: str = "exponential",
    risk_tolerance: float = 1.0,
    discounted: bool = True,
) -> list[RiskResult]:
    """Expected utility per strategy from ``tree_paths.PathRow`` objects.

    Raises ``ValueError`` when a path ends in a subtree (PSM/DES/Markov leaf):
    those return only an expected payoff, not a distribution, so a utility
    over them would silently equal the risk-neutral value.
    """
    by_strategy: dict[str, list[Any]] = {}
    for row in rows:
        if getattr(row, "leaf_type", "terminal") != "terminal":
            raise ValueError(
                f"Strategy '{row.strategy}' ends in a {row.leaf_type} subtree: a risk "
                "preference needs the outcome of every path, which a subtree does not give."
            )
        by_strategy.setdefault(row.strategy, []).append(row)

    results: list[RiskResult] = []
    for name, paths in by_strategy.items():
        total_p = sum(r.p for r in paths)
        if total_p <= 0:
            raise ValueError(f"Strategy '{name}' has zero total probability")
        nmbs, eu, enmb = [], 0.0, 0.0
        for r in paths:
            totals = r.discounted if discounted else r.undiscounted
            nmb = wtp * totals.get("effect", 0.0) - totals.get("cost", 0.0)
            w = r.p / total_p
            nmbs.append(nmb)
            enmb += w * nmb
            eu += w * utility(nmb, kind, risk_tolerance)
        ce = inverse_utility(eu, kind, risk_tolerance)
        results.append(RiskResult(name, enmb, eu, ce, enmb - ce, min(nmbs), max(nmbs), total_p))
    order = sorted(results, key=lambda x: -x.expected_utility)
    for i, res in enumerate(order, 1):
        res.rank = i
    return results


def results_to_records(results: list[RiskResult]) -> list[dict[str, Any]]:
    return [
        {
            "Rank": r.rank,
            "Strategy": r.strategy,
            "E[NMB]": r.expected_nmb,
            "Expected utility": r.expected_utility,
            "Certainty equivalent": r.certainty_equivalent,
            "Risk premium": r.risk_premium,
            "Worst path NMB": r.worst_nmb,
            "Best path NMB": r.best_nmb,
        }
        for r in sorted(results, key=lambda x: x.rank)
    ]
