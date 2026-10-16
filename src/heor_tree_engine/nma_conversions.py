"""Helpers to turn what trials report into what the NMA needs.

- Mean and SD from the median with the range and/or IQR:
  mean - Luo et al. 2018 (Stat Methods Med Res 27:1785);
  SD   - Wan et al. 2014 (BMC Med Res Methodol 14:135).
- SD from the SE or the CI of a mean (Cochrane Handbook 6.5.2.2; t with n-1 df).
- SE of an effect from its CI or from its (two-sided) p-value (Cochrane
  Handbook 6.3.1-6.3.2); log scale for ratio measures.
- Combining two arms of the same treatment (Cochrane Handbook Table 6.5.a).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .nma_freq import t_quantile
from .nma_parser import _normal_quantile


@dataclass
class MeanSD:
    mean: float
    sd: float
    method: str


def _xi(n: float) -> float:
    return 2.0 * _normal_quantile((n - 0.375) / (n + 0.25))


def _eta(n: float) -> float:
    return 2.0 * _normal_quantile((0.75 * n - 0.125) / (n + 0.25))


def mean_sd_from_median(n: float, median: float, minimum: float | None = None, maximum: float | None = None,
                        q1: float | None = None, q3: float | None = None) -> MeanSD:
    """Scenario S1 (min, median, max), S2 (Q1, median, Q3) or S3 (all five)."""

    if n is None or n < 2:
        raise ValueError("n must be at least 2")
    has_range = minimum is not None and maximum is not None
    has_iqr = q1 is not None and q3 is not None
    if has_range and maximum < minimum or has_iqr and q3 < q1:
        raise ValueError("Upper limit is below the lower limit")
    if has_range and has_iqr:
        w1 = 2.2 / (2.2 + n ** 0.75)
        w2 = 0.7 - 0.72 / n ** 0.55
        mean = w1 * (minimum + maximum) / 2 + w2 * (q1 + q3) / 2 + (1 - w1 - w2) * median
        sd = (maximum - minimum) / (2 * _xi(n)) + (q3 - q1) / (2 * _eta(n))
        return MeanSD(mean, sd, "S3: min, Q1, median, Q3, max (Luo 2018; Wan 2014)")
    if has_iqr:
        w = 0.7 + 0.39 / n
        mean = w * (q1 + q3) / 2 + (1 - w) * median
        return MeanSD(mean, (q3 - q1) / _eta(n), "S2: Q1, median, Q3 (Luo 2018; Wan 2014)")
    if has_range:
        w = 4.0 / (4.0 + n ** 0.75)
        mean = w * (minimum + maximum) / 2 + (1 - w) * median
        return MeanSD(mean, (maximum - minimum) / _xi(n), "S1: min, median, max (Luo 2018; Wan 2014)")
    raise ValueError("Give the range (min, max), the IQR (Q1, Q3) or both")


def sd_from_se(se: float, n: float) -> float:
    return se * math.sqrt(n)


def sd_from_ci(lower: float, upper: float, n: float, level: float = 0.95) -> float:
    """SD of one group from the CI of its mean (t with n-1 df)."""

    if n < 2 or upper <= lower:
        raise ValueError("Need n >= 2 and upper > lower")
    return math.sqrt(n) * (upper - lower) / (2 * t_quantile(0.5 + level / 2, n - 1))


def se_from_ci(lower: float, upper: float, ratio: bool, level: float = 0.95) -> float:
    if upper <= lower or ratio and lower <= 0:
        raise ValueError("Need upper > lower (> 0 for ratios)")
    width = math.log(upper) - math.log(lower) if ratio else upper - lower
    return width / (2 * _normal_quantile(0.5 + level / 2))


def se_from_p(estimate: float, p_value: float, ratio: bool) -> float:
    """SE from a two-sided p-value of a z-test; `estimate` on the natural scale."""

    if not 0 < p_value < 1:
        raise ValueError("p must be between 0 and 1 (exclusive)")
    value = math.log(estimate) if ratio else estimate
    if value == 0:
        raise ValueError("The estimate equals the null value: the SE cannot be recovered from p")
    return abs(value) / _normal_quantile(1 - p_value / 2)


def combine_arms(n1: float, mean1: float, sd1: float, n2: float, mean2: float, sd2: float) -> tuple[float, float, float]:
    """Pool two groups of a continuous outcome: (n, mean, sd)."""

    n = n1 + n2
    mean = (n1 * mean1 + n2 * mean2) / n
    variance = ((n1 - 1) * sd1 ** 2 + (n2 - 1) * sd2 ** 2 + n1 * n2 / n * (mean1 - mean2) ** 2) / (n - 1)
    return n, mean, math.sqrt(variance)
