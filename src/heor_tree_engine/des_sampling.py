"""Time-to-event sampling for DES models.

Every parametric distribution is sampled by inverting its survival
function, S(x) = u, with a seeded numpy Generator -- closed forms with
plain ``math`` (no scipy needed) for everything but the generalized gamma
(lazy scipy import, same convention as engine._curve_survival).

Conditional sampling (TreeAge ``DistSampleCond(dist; lower)``, help 43.9):
to draw X given X > lower, solve S(x) = u * S(lower). This is exact
inverse-transform sampling from the truncated distribution -- no rejection
loop, so it stays fast even when ``lower`` is deep in the tail (e.g. a
90-year-old sampled from a mortality curve).
"""

from __future__ import annotations

import math
from bisect import bisect_left, bisect_right
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from statistics import NormalDist
from typing import Any

from .des_models import TABLE_TO_TIME_TYPES
from .distributions import conditional_event_time, conditional_survival

_STD_NORMAL = NormalDist()


def stacy_to_prentice_params(params: Mapping[str, Any]) -> dict[str, float | str] | None:
    """Convert legacy Stacy a/c/scale values or expressions to Prentice mu/sigma/Q."""

    try:
        a, c, scale = (params[name] for name in ("a", "c", "scale"))
    except KeyError:
        return None
    if any(isinstance(value, bool) or not isinstance(value, (int, float, str)) for value in (a, c, scale)):
        return None

    if all(isinstance(value, (int, float)) for value in (a, c, scale)):
        a, c, scale = float(a), float(c), float(scale)
        if a <= 0 or c == 0 or scale <= 0:
            return None
        q = math.copysign(1.0 / math.sqrt(a), c)
        sigma = 1.0 / (abs(c) * math.sqrt(a))
        mu = math.log(scale) - (2.0 / c) * math.log(abs(q))
        return {"mu": mu, "sigma": sigma, "Q": q}

    a_expr, c_expr, scale_expr = (f"({value})" for value in (a, c, scale))
    q_expr = f"(sign({c_expr}) / sqrt({a_expr}))"
    sigma_expr = f"(1 / (abs({c_expr}) * sqrt({a_expr})))"
    mu_expr = f"(log({scale_expr}) - (2 / {c_expr}) * log(abs({q_expr})))"
    return {"mu": mu_expr, "sigma": sigma_expr, "Q": q_expr}


def survival(distribution: str, params: Mapping[str, float], x: float) -> float:
    """S(x) = P(X > x); supports both PSM Prentice and DES Stacy gamma params."""

    if x <= 0:
        return 1.0
    if distribution == "exponential":
        return math.exp(-params["rate"] * x)
    if distribution == "weibull":
        return math.exp(-((x / params["scale"]) ** params["shape"]))
    if distribution == "log_normal":
        return 0.5 * math.erfc((math.log(x) - params["meanlog"]) / (params["sdlog"] * math.sqrt(2)))
    if distribution == "log_logistic":
        return 1.0 / (1.0 + (x / params["scale"]) ** params["shape"])
    if distribution == "gompertz":
        shape, rate = params["shape"], params["rate"]
        if shape == 0:
            return math.exp(-rate * x)
        return math.exp(-(rate / shape) * math.expm1(shape * x))
    if distribution == "generalized_gamma":
        if "Q" in params and params["Q"] == 0:
            return 0.5 * math.erfc((math.log(x) - params["mu"]) / (params["sigma"] * math.sqrt(2)))
        return float(_gengamma(params).sf(x))
    raise ValueError(f"Unsupported time-to-event distribution '{distribution}'")


def inverse_survival(distribution: str, params: Mapping[str, float], u: float) -> float:
    """x such that S(x) = u, for u in (0, 1]. May return math.inf (e.g. a
    Gompertz with negative shape has a 'cure' plateau: some patients never
    experience the event)."""

    if u >= 1.0:
        return 0.0
    if u <= 0.0:
        return math.inf
    if distribution == "exponential":
        return -math.log(u) / params["rate"]
    if distribution == "weibull":
        return params["scale"] * (-math.log(u)) ** (1.0 / params["shape"])
    if distribution == "log_normal":
        return math.exp(params["meanlog"] + params["sdlog"] * _STD_NORMAL.inv_cdf(1.0 - u))
    if distribution == "log_logistic":
        return params["scale"] * ((1.0 - u) / u) ** (1.0 / params["shape"])
    if distribution == "gompertz":
        shape, rate = params["shape"], params["rate"]
        if shape == 0:
            return -math.log(u) / rate
        inner = 1.0 - shape * math.log(u) / rate
        return math.log(inner) / shape if inner > 0 else math.inf
    if distribution == "generalized_gamma":
        if "Q" in params and params["Q"] == 0:
            return math.exp(params["mu"] + params["sigma"] * _STD_NORMAL.inv_cdf(1.0 - u))
        return float(_gengamma(params).isf(u))
    raise ValueError(f"Unsupported time-to-event distribution '{distribution}'")


def _gengamma(params: Mapping[str, float]):
    try:
        from scipy import stats
    except ImportError as exc:
        raise ValueError(
            "Generalized Gamma time-to-event requires scipy (not installed in this environment)"
        ) from exc
    if "Q" in params:
        mu, sigma, q = params["mu"], params["sigma"], params["Q"]
        if sigma <= 0 or q == 0:
            raise ValueError("Prentice generalized gamma requires sigma > 0 and Q != 0 in the Stacy conversion")
        a = 1.0 / (q * q)
        c = q / sigma
        log_scale = mu + (2.0 * sigma / q) * math.log(abs(q))
        return stats.gengamma(a, c, scale=math.exp(log_scale))
    return stats.gengamma(params["a"], params["c"], scale=params["scale"])


def conditional_event_probability(
    distribution: str,
    params: Mapping[str, float],
    start: float,
    duration: float,
) -> float:
    """P(event in (start, start + duration] | no event by start) -- exactly
    DistToProb, computed from the same shared distributions.conditional_survival()."""

    if survival(distribution, params, start) <= 0:
        return 0.0
    return 1.0 - conditional_survival(lambda x: survival(distribution, params, x), start, duration)


def sample_parametric(
    distribution: str,
    params: Mapping[str, float],
    uniform: float,
    lower: float = 0.0,
) -> float:
    """Draw X | X > lower (lower = 0 -> unconditional) from one uniform
    number in (0, 1]. Returns the value on the distribution's own axis
    (NOT lower-subtracted).

    Same formula as DistToProb, used the other way round: DistToProb gives
    1 - conditional_survival(S, lower, x - lower) for a known x; here x is
    the unknown, found from conditional_survival(S, lower, x - lower) = u,
    i.e. S(x) = u * S(lower). So P(X - lower <= d) equals
    DistToProb(dist, lower, d) exactly (tested)."""

    remaining = conditional_event_time(
        lambda x: survival(distribution, params, x),
        lambda u: inverse_survival(distribution, params, u),
        lower,
        uniform,
    )
    return lower + remaining


# --------------------------------------------------------------------------
# TableToTime (TreeAge "TableProb", help 43.8)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TableCDF:
    """A table turned into a CDF over its index column (e.g. age).

    xs/cdf are the knots; with ``interpolate`` the CDF is linear between
    knots (a continuous draw -- uniform within each interval, the usual
    'uniform distribution of deaths' assumption for life tables); without
    it the draw is discrete: one of the index values themselves."""

    xs: tuple[float, ...]
    cdf: tuple[float, ...]
    interpolate: bool

    def value_at(self, x: float) -> float:
        xs, cdf = self.xs, self.cdf
        if x <= xs[0]:
            return cdf[0] if x == xs[0] else 0.0
        if x >= xs[-1]:
            return 1.0
        if not self.interpolate:
            return cdf[bisect_right(xs, x) - 1]
        i = bisect_right(xs, x) - 1
        span = xs[i + 1] - xs[i]
        return cdf[i] + (cdf[i + 1] - cdf[i]) * (x - xs[i]) / span

    def sample(self, uniform: float, lower: float = -math.inf) -> float:
        """X | X > lower (continuous) or X >= lower (discrete)."""

        xs, cdf = self.xs, self.cdf
        if self.interpolate:
            f_lower = self.value_at(lower) if lower > xs[0] else 0.0
            target = f_lower + (1.0 - uniform) * (1.0 - f_lower)
            target = min(max(target, f_lower), 1.0)
            i = bisect_left(cdf, target)
            if i <= 0:
                return max(xs[0], lower)
            if i >= len(xs):
                return xs[-1]
            lo_f, hi_f = cdf[i - 1], cdf[i]
            x = xs[i] if hi_f == lo_f else xs[i - 1] + (xs[i] - xs[i - 1]) * (target - lo_f) / (hi_f - lo_f)
            return max(x, lower)
        # discrete
        start = bisect_left(xs, lower)
        if start >= len(xs):
            return xs[-1]
        f_before = cdf[start - 1] if start > 0 else 0.0
        target = f_before + (1.0 - uniform) * (1.0 - f_before)
        i = bisect_left(cdf, target, lo=start)
        return xs[min(i, len(xs) - 1)]


def build_table_cdf(
    rows: Sequence[Mapping[str, Any]],
    index_column: str,
    value_column: str,
    prob_type: str,
    interpolate: bool,
) -> TableCDF:
    """prob_type:
      * "probability" -- TreeAge's natural form: probability the event
        happens AT/within each row (first row 0, all rows sum to 1);
      * "cumulative"  -- cumulative probability (first row 0, last row 1);
      * "conditional" -- a life table's q(x): probability of the event in
        [x, next x) given no event before x (DECIDE extension -- the form
        most published mortality tables use). The last row's interval is
        closed with everyone experiencing the event by the next knot.
    """

    if prob_type not in TABLE_TO_TIME_TYPES:
        raise ValueError(f"prob_type must be one of {TABLE_TO_TIME_TYPES}")
    pairs: list[tuple[float, float]] = []
    for row in rows:
        if index_column not in row or value_column not in row:
            raise ValueError(f"Table rows need columns '{index_column}' and '{value_column}'")
        pairs.append((float(row[index_column]), float(row[value_column])))
    if len(pairs) < 2:
        raise ValueError("A TableToTime table needs at least 2 rows")
    pairs.sort()
    xs = [x for x, _ in pairs]
    if len(set(xs)) != len(xs):
        raise ValueError("TableToTime index values must be unique")
    values = [v for _, v in pairs]
    if any(v < 0 or v > 1 for v in values):
        raise ValueError("TableToTime values must be probabilities between 0 and 1")

    if prob_type == "conditional":
        knots = list(xs) + [xs[-1] + (xs[-1] - xs[-2])]
        cdf = [0.0]
        surviving = 1.0
        for q in values:
            surviving *= 1.0 - q
            cdf.append(1.0 - surviving)
        cdf[-1] = 1.0
        return TableCDF(tuple(knots), tuple(cdf), interpolate)

    if prob_type == "probability":
        cdf, running = [], 0.0
        for v in values:
            running += v
            cdf.append(running)
    else:
        cdf = list(values)
    if any(b < a - 1e-12 for a, b in zip(cdf, cdf[1:])):
        raise ValueError("TableToTime cumulative probabilities must be non-decreasing")
    if abs(cdf[-1] - 1.0) > 1e-6:
        raise ValueError(
            f"TableToTime probabilities must reach 1 at the last row (got {cdf[-1]:.6g})"
        )
    cdf[-1] = 1.0
    return TableCDF(tuple(xs), tuple(min(max(c, 0.0), 1.0) for c in cdf), interpolate)


def table_to_prob(cdf: TableCDF, start: float, duration: float) -> float:
    """TableToProb: P(event in (start, start + duration] | no event by start)
    from a table -- the SAME conditional-survival formula as DistToProb
    (distributions.conditional_survival), with S(x) = 1 - table CDF(x).
    For a q(x) life table at a whole age, TableToProb(t, age, 1) = q(age)."""

    def table_survival(x: float) -> float:
        return 1.0 - cdf.value_at(x)

    if table_survival(start) <= 0:
        return 0.0
    return 1.0 - conditional_survival(table_survival, start, duration)


def table_to_time(cdf: TableCDF, start: float, uniform: float) -> float:
    """TableToTime: random remaining time (from `start`) until the event for
    someone event-free at `start` -- solves S(start + d) = u * S(start), the
    same formula TableToProb uses (so P(d <= t2) = TableToProb(start, t2)).
    Also what the DES engine's TableToTime time-to-event draws."""

    return max(0.0, cdf.sample(uniform, lower=start) - start)


# --------------------------------------------------------------------------
# Discounting (TreeAge help 43.13 "Discounting in DES models")
# --------------------------------------------------------------------------


def discount_factor(rate: float, time: float) -> float:
    """Event rewards: accrued at one moment -> the standard factor."""

    return 1.0 if not rate or time <= 0 else (1.0 + rate) ** (-time)


def discounted_duration(rate: float, start: float, end: float) -> float:
    """Time rewards: integral of (1+rate)^-s ds over [start, end] -- the
    continuous-discounting counterpart of TreeAge's DiscountContDES. Multiply
    by a per-time-unit reward to get its discounted accumulated value."""

    if end <= start:
        return 0.0
    if not rate:
        return end - start
    delta = math.log1p(rate)
    return (math.exp(-delta * start) - math.exp(-delta * end)) / delta
