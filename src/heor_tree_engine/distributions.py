"""Runtime adapters for DECIDE distribution functions."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from typing import Any


class DistributionError(ValueError):
    """Raised when a DECIDE distribution cannot be evaluated."""


def reciprocal_parameter_value(value: Any) -> Any:
    """1 / value for a stored parameter value that may be a number, a numeric
    string or an expression (a reference to a Parameter / Variable): numbers
    give a number, anything else is wrapped as ``1/(expression)`` so its
    meaning is preserved."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        text = str(value).strip()
        return f"1/({text})" if text else value
    if number == 0:
        return value
    reciprocal = 1.0 / number
    rounded = float(f"{reciprocal:.12g}")
    if abs(rounded * number - 1.0) < 1e-12:
        return f"{reciprocal:.12g}"
    return f"1/{value}"


def migrate_legacy_gamma_parameters(parameters: dict[str, Any]) -> None:
    """Rewrite a Gamma definition saved with k (shape) and theta (SCALE) to the
    flexsurv convention (shape, RATE), in place: k -> shape, theta -> rate =
    1 / theta.  Numbers and numeric strings give a number, a Parameter or
    expression reference is wrapped as 1/(expression) so its meaning is kept.
    Does nothing when the new keys are already present."""

    if "k" in parameters and "shape" not in parameters:
        parameters["shape"] = parameters.pop("k")
    if "theta" in parameters and "rate" not in parameters:
        parameters["rate"] = reciprocal_parameter_value(parameters.pop("theta"))


def _scipy_distribution_uncached(definition: Mapping[str, Any], resolve: Callable[[Any], float]):
    try:
        from scipy import stats
    except ImportError as exc:
        raise DistributionError("Distribution functions require scipy") from exc

    distribution_type = str(definition.get("type", "")).strip()
    parameters = definition.get("parameters", {})
    values = {str(name): resolve(value) for name, value in parameters.items()}
    try:
        if distribution_type == "Beta":
            return stats.beta(values["a"], values["b"])
        if distribution_type == "Bernoulli":
            return stats.bernoulli(values["p"])
        if distribution_type == "Binomial":
            return stats.binom(int(values["n"]), values["p"])
        if distribution_type == "Categorical":
            # p: a list of event probabilities summing to 1 (enter as a
            # list literal, e.g. {0.2, 0.3, 0.5}) -- the distribution's
            # "value" is which category was drawn (0-based index). scipy
            # has no dedicated Categorical distribution object; a discrete
            # distribution with those exact point masses is the same thing.
            p = values["p"]
            if not isinstance(p, (list, tuple)) or not p:
                raise DistributionError("Categorical: 'p' must be a list of probabilities, e.g. {0.2, 0.3, 0.5}")
            return stats.rv_discrete(name="categorical", values=(range(len(p)), p))
        if distribution_type == "Cauchy":
            return stats.cauchy(loc=values["x0"], scale=values["gamma"])
        if distribution_type == "Chi-Square":
            return stats.chi2(values["k"])
        if distribution_type == "Discrete Uniform":
            return stats.randint(int(values["a"]), int(values["b"]) + 1)
        if distribution_type == "Exponential":
            return stats.expon(scale=1.0 / values["rate"])
        if distribution_type == "Gamma":
            # flexsurv convention (R flexsurv `dgamma`): shape and RATE, so
            # mean = shape / rate and scale = 1 / rate.  Documents saved
            # before this change stored k (shape) and theta (SCALE); honour
            # those when no `rate` is present (the editor also rewrites
            # them to shape/rate the first time the row is loaded).
            if "rate" in values or "shape" in values:
                return stats.gamma(values["shape"], scale=1.0 / values["rate"])
            return stats.gamma(values["k"], scale=values["theta"])
        if distribution_type == "Generalized Gamma":
            # Prentice (flexsurv) mu/sigma/Q reparametrisation -- matches
            # the PSM curve picker's own Generalized Gamma params, so a
            # paper's published fitted values (almost always reported as
            # mu/sigma/Q, e.g. Cai et al. 2021) can be typed straight in
            # here too. Q == 0 is the log-normal special case (undefined
            # a/c below -- same convention as des_sampling.survival's own
            # Q==0 branch). Any row saved before this change still has the
            # old scipy-native Stacy a/c/scale keys instead; detect and
            # honor those rather than breaking existing documents (same
            # "legacy_gengamma" convention as validation.py/psm_validation.py
            # for PSM curves).
            if "mu" in values:
                mu, sigma, q = values["mu"], values["sigma"], values["Q"]
                if q == 0:
                    return stats.lognorm(s=sigma, scale=math.exp(mu))
                a = 1.0 / (q * q)
                c = q / sigma
                log_scale = mu + (2.0 * sigma / q) * math.log(abs(q))
                return stats.gengamma(a, c, scale=math.exp(log_scale))
            return stats.gengamma(values["a"], values["c"], scale=values["scale"])
        if distribution_type == "Geometric":
            return stats.geom(values["p"])
        if distribution_type == "Gompertz":
            # Standard HEOR/flexsurv parametrisation: hazard h(t) = rate *
            # exp(shape * t). scipy.stats.gompertz(c, scale) has survival
            # exp(-c*(exp(x/scale)-1)); matching term-by-term gives
            # scale = 1/shape, c = rate/shape. Only shape > 0 (increasing
            # hazard -- the common case: mortality/progression risk rising
            # over time) maps to a valid positive scipy scale; shape <= 0
            # falls through to scipy's/Python's own ValueError or
            # ZeroDivisionError below, caught and re-raised the same way
            # every other distribution's bad input already is.
            shape = values["shape"]
            rate = values["rate"]
            return stats.gompertz(rate / shape, scale=1.0 / shape)
        if distribution_type == "Gumbel":
            return stats.gumbel_r(loc=values["mu"], scale=values["beta"])
        if distribution_type == "Half-Cauchy":
            return stats.halfcauchy(scale=values["gamma"])
        if distribution_type == "Half-Normal":
            return stats.halfnorm(scale=values["sigma"])
        if distribution_type == "Hypergeometric":
            return stats.hypergeom(int(values["w"] + values["b"]), int(values["w"]), int(values["n"]))
        if distribution_type == "Laplace":
            return stats.laplace(loc=values["mu"], scale=values["b"])
        if distribution_type == "Logistic":
            return stats.logistic(loc=values["mu"], scale=values["s"])
        if distribution_type == "Log-logistic":
            # scipy's "fisk" distribution IS the log-logistic distribution:
            # survival 1/(1+(x/scale)^c) matches the standard shape/scale
            # log-logistic parametrisation term-for-term (c = shape).
            return stats.fisk(values["shape"], scale=values["scale"])
        if distribution_type == "Lognormal":
            return stats.lognorm(s=values["sdlog"], scale=math.exp(values["meanlog"]))
        if distribution_type == "Negative Binomial":
            return stats.nbinom(int(values["r"]), values["p"])
        if distribution_type == "Normal":
            return stats.norm(loc=values["mu"], scale=values["sigma"])
        if distribution_type == "Pareto":
            return stats.pareto(values["alpha"], scale=values["k"])
        if distribution_type == "PERT Distribution (Program Evaluation and Review Technique)":
            # a=minimum, b=most likely (mode), c=maximum -- the standard
            # PERT-Beta reparametrisation (shape1/shape2 chosen so the mode
            # is exactly b): a 4x-weighted Beta, matching TreeAge's/Amua's
            # own PERT, without needing a distribution family scipy doesn't
            # have natively.
            a, b, c = values["a"], values["b"], values["c"]
            if not (a <= b <= c) or a == c:
                raise DistributionError("PERT: requires minimum <= most likely <= maximum, with minimum < maximum")
            shape1 = 1.0 + 4.0 * (b - a) / (c - a)
            shape2 = 1.0 + 4.0 * (c - b) / (c - a)
            return stats.beta(shape1, shape2, loc=a, scale=c - a)
        if distribution_type == "Poisson":
            return stats.poisson(values["lambda"])
        if distribution_type == "Student's t-distribution":
            return stats.t(values["nu"])
        if distribution_type == "Triangular":
            lower, mode, upper = values["a"], values["b"], values["c"]
            return stats.triang((mode - lower) / (upper - lower), loc=lower, scale=upper - lower)
        if distribution_type == "Truncated Normal":
            # mu/sigma describe the UNTRUNCATED normal; a/b are the
            # (untruncated-scale) bounds -- scipy.stats.truncnorm instead
            # wants them standardized ((bound - mu) / sigma).
            mu, sigma, a, b = values["mu"], values["sigma"], values["a"], values["b"]
            return stats.truncnorm((a - mu) / sigma, (b - mu) / sigma, loc=mu, scale=sigma)
        if distribution_type == "Uniform":
            return stats.uniform(loc=values["a"], scale=values["b"] - values["a"])
        if distribution_type == "Weibull":
            return stats.weibull_min(values["shape"], scale=values["scale"])
        if distribution_type == "Zipf":
            # s: exponent (>0), n: number of elements (finite support
            # 1..n) -- scipy's finite-support "zipfian" (its plain "zipf"
            # is the infinite-support version, which Zipf's declared "n"
            # parameter wouldn't do anything for).
            return stats.zipfian(values["s"], int(values["n"]))
    except (KeyError, ValueError, ZeroDivisionError) as exc:
        raise DistributionError(f"Invalid parameters for distribution '{distribution_type}'") from exc
    raise DistributionError(f"Unsupported distribution '{distribution_type}'")


_FROZEN_DISTRIBUTION_CACHE: dict[Any, Any] = {}


def _scipy_distribution(definition: Mapping[str, Any], resolve: Callable[[Any], float]):
    """Memoized wrapper: building a frozen scipy distribution is slow
    (scipy re-formats its docstrings on every construction -- ~30% of a
    long Markov run in profiling) and DistToProb/DistSurvival rebuild the
    identical object on every call. Key on the distribution type plus the
    RESOLVED parameter values so a changed parameter (PSA/SA) never reuses
    a stale object."""

    try:
        distribution_type = str(definition.get("type", "")).strip()
        parameters = definition.get("parameters", {})
        key = (distribution_type, tuple(sorted((str(n), float(resolve(v))) for n, v in parameters.items())))
    except Exception:
        return _scipy_distribution_uncached(definition, resolve)
    cached = _FROZEN_DISTRIBUTION_CACHE.get(key)
    if cached is None:
        cached = _scipy_distribution_uncached(definition, resolve)
        if len(_FROZEN_DISTRIBUTION_CACHE) > 4096:
            _FROZEN_DISTRIBUTION_CACHE.clear()
        _FROZEN_DISTRIBUTION_CACHE[key] = cached
    return cached


_FAST_SF_CACHE: dict[Any, Any] = {}
_FAST_SF_TYPES = ("Exponential", "Weibull", "Lognormal", "Log-logistic", "Gompertz")


def _fast_survival(definition: Mapping[str, Any], resolve: Callable[[Any], float]):
    """Scalar survival function S(x) for the five parametric time-to-event families used by Markov/PSA formulas
    (Exponential, Weibull, Lognormal, Log-logistic, Gompertz), written with plain `math` instead of a frozen scipy
    object: a scipy `.sf()` call costs ~100 us, a `math.exp` ~0.1 us, and a Markov run calls it thousands of times
    (PSA multiplies that by the iteration count).  Same formulas as scipy's, so results agree to ~1e-15.  Returns
    None (-> the caller keeps using scipy, including its error messages) for any other family or for parameters
    scipy would reject (non-positive or non-finite)."""

    distribution_type = str(definition.get("type", "")).strip()
    if distribution_type not in _FAST_SF_TYPES:
        return None
    try:
        parameters = definition.get("parameters", {})
        values = {str(n): float(resolve(v)) for n, v in parameters.items()}
    except Exception:  # noqa: BLE001 - e.g. a list-valued parameter: let scipy deal with it
        return None
    key = (distribution_type, tuple(sorted(values.items())))
    if key in _FAST_SF_CACHE:
        return _FAST_SF_CACHE[key]
    fn = None
    try:
        if distribution_type == "Exponential":
            rate = values["rate"]
            if rate > 0 and math.isfinite(rate):
                fn = lambda x: 1.0 if x <= 0 else math.exp(-rate * x)  # noqa: E731
        elif distribution_type == "Weibull":
            shape, scale = values["shape"], values["scale"]
            if shape > 0 and scale > 0 and math.isfinite(shape) and math.isfinite(scale):
                fn = lambda x: 1.0 if x <= 0 else math.exp(-((x / scale) ** shape))  # noqa: E731
        elif distribution_type == "Lognormal":
            meanlog, sdlog = values["meanlog"], values["sdlog"]
            if sdlog > 0 and math.isfinite(sdlog) and math.isfinite(meanlog):
                denom = sdlog * math.sqrt(2.0)
                fn = lambda x: 1.0 if x <= 0 else 0.5 * math.erfc((math.log(x) - meanlog) / denom)  # noqa: E731
        elif distribution_type == "Log-logistic":
            shape, scale = values["shape"], values["scale"]
            if shape > 0 and scale > 0 and math.isfinite(shape) and math.isfinite(scale):
                fn = lambda x: 1.0 if x <= 0 else 1.0 / (1.0 + (x / scale) ** shape)  # noqa: E731
        elif distribution_type == "Gompertz":
            shape, rate = values["shape"], values["rate"]
            if shape > 0 and rate > 0 and math.isfinite(shape) and math.isfinite(rate):
                ratio = rate / shape
                fn = lambda x: 1.0 if x <= 0 else math.exp(-ratio * math.expm1(min(shape * x, 700.0)))  # noqa: E731
    except (KeyError, OverflowError, ValueError, ZeroDivisionError):
        fn = None
    if len(_FAST_SF_CACHE) > 4096:
        _FAST_SF_CACHE.clear()
    _FAST_SF_CACHE[key] = fn
    return fn


def _distribution(definition: Mapping[str, Any], resolve: Callable[[Any], float]):
    return _scipy_distribution(definition, resolve)


def distribution_mean(definition: Mapping[str, Any], resolve: Callable[[Any], float]) -> float:
    return float(_distribution(definition, resolve).mean())


def distribution_sample(
    definition: Mapping[str, Any],
    resolve: Callable[[Any], float],
    rng: Any | None = None,
) -> float:
    """Draw one sample. `rng` is optional (a numpy.random.Generator) so
    existing DistSample() calls in Tree/Markov formulas keep using scipy's
    own unseeded global random state by default; microsim_engine passes a
    seeded Generator so N-patient runs are reproducible from
    model.random_seed."""

    distribution = _distribution(definition, resolve)
    if rng is not None:
        return float(distribution.rvs(random_state=rng))
    return float(distribution.rvs())


def distribution_cdf(definition: Mapping[str, Any], value: float, resolve: Callable[[Any], float]) -> float:
    return float(_distribution(definition, resolve).cdf(value))


def distribution_ppf(definition: Mapping[str, Any], probability: float, resolve: Callable[[Any], float]) -> float:
    return float(_distribution(definition, resolve).ppf(probability))


def distribution_sample_cond(definition: Mapping[str, Any], lower: float, upper: float, resolve: Callable[[Any], float]) -> float:
    distribution = _distribution(definition, resolve)
    lower_probability = distribution.cdf(lower)
    upper_probability = distribution.cdf(upper)
    if not lower_probability < upper_probability:
        raise DistributionError("Distribution interval has no probability mass")
    return float(distribution.ppf(lower_probability + (upper_probability - lower_probability) * distribution.rvs()))


def distribution_trim(definition: Mapping[str, Any], lower: float, upper: float, resolve: Callable[[Any], float]) -> float:
    distribution = _distribution(definition, resolve)
    sample = float(distribution.rvs())
    for _ in range(9):
        if lower <= sample <= upper:
            return sample
        sample = float(distribution.rvs())
    return min(max(sample, lower), upper)


def conditional_survival(survival: Callable[[float], float], start: float, duration: float) -> float:
    """S(start + duration) / S(start): the probability that an event which
    has not happened by `start` still has not happened `duration` later.

    This is the ONE definition shared by every place DECIDE turns a
    survival curve into a conditional quantity (same math ->
    one formula):
      * DistToProb(dist, t1, t2) = 1 - conditional_survival(S, t1, t2)
        -- a per-cycle transition PROBABILITY for Markov/Tree formulas;
      * the DES engine's conditional time-to-event sampling (Patient clock
        / TreeAge DistSampleCond) draws the event TIME x with
        conditional_survival(S, now, x - now) = u, u ~ U(0, 1]
        (des_sampling.sample_parametric).
    Returns 0.0 when S(start) <= 0 (no survivors left to condition on)."""

    survival_at_start = float(survival(start))
    if survival_at_start <= 0:
        return 0.0
    return float(survival(start + duration)) / survival_at_start


def conditional_event_time(
    survival: Callable[[float], float],
    inverse_survival: Callable[[float], float],
    start: float,
    uniform: float,
) -> float:
    """The time-returning use of conditional_survival(): the remaining time
    d >= 0 until the event for someone event-free at `start`, found from
        conditional_survival(S, start, d) = uniform,  i.e.  S(start + d) = uniform * S(start),
    with `uniform` in (0, 1]. Drawing `uniform` at random makes
    P(d <= t2) = DistToProb(dist, start, t2) exactly -- same formula,
    solved for the time instead of the probability. Shared by
    DistToTime() and the DES engine (des_sampling.sample_parametric)."""

    survival_at_start = float(survival(start)) if start > 0 else 1.0
    if survival_at_start <= 0:
        return 0.0
    x = float(inverse_survival(uniform * survival_at_start))
    return max(0.0, x - start)


def distribution_to_time(
    definition: Mapping[str, Any],
    start: float,
    resolve: Callable[[Any], float],
    rng: Any | None = None,
) -> float:
    """DistToTime(distribution, t1): a random remaining time until the event
    for someone still event-free at t1 (t1 defaults to 0 = unconditional).
    Returns time measured FROM t1, like DistToProb's t2. `rng` (a numpy
    Generator) makes it reproducible; the DES engine passes its patient's
    own stream."""

    distribution = _distribution(definition, resolve)
    if rng is None:
        import numpy as _np

        rng = _np.random.default_rng()
    uniform = 1.0 - float(rng.random())  # (0, 1]
    return conditional_event_time(
        lambda x: float(distribution.sf(x)), lambda u: float(distribution.isf(u)), start, uniform,
    )


def distribution_to_prob(definition: Mapping[str, Any], start: float, duration: float, resolve: Callable[[Any], float]) -> float:
    """DistToProb: P(start < T <= start + duration | T > start)
    = 1 - conditional_survival(S, start, duration)."""

    fast = _fast_survival(definition, resolve)
    if fast is not None:
        if fast(start) <= 0:
            return 0.0
        return 1.0 - conditional_survival(fast, start, duration)
    distribution = _distribution(definition, resolve)
    if float(distribution.sf(start)) <= 0:
        return 0.0
    return 1.0 - conditional_survival(lambda x: float(distribution.sf(x)), start, duration)


def distribution_survival(definition: Mapping[str, Any], value: float, resolve: Callable[[Any], float]) -> float:
    fast = _fast_survival(definition, resolve)
    if fast is not None:
        return float(fast(value))
    return float(_distribution(definition, resolve).sf(value))


def distribution_hazard(definition: Mapping[str, Any], value: float, resolve: Callable[[Any], float]) -> float:
    distribution = _distribution(definition, resolve)
    survival = float(distribution.sf(value))
    return math.inf if survival <= 0 else float(distribution.pdf(value) / survival)
