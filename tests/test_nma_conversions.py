import math
import random
from statistics import NormalDist

import pytest

from heor_tree_engine.nma_conversions import (combine_arms, mean_sd_from_median, sd_from_ci, sd_from_se,
                                              se_from_ci, se_from_p)


def test_median_conversions_match_published_formulas() -> None:
    # Wan 2014: xi(n) = 2 Phi^-1((n - .375)/(n + .25)); eta(n) = 2 Phi^-1((.75n - .125)/(n + .25)).
    s2 = mean_sd_from_median(50, median=10, q1=8, q3=13)
    w = 0.7 + 0.39 / 50
    assert s2.mean == pytest.approx(w * 10.5 + (1 - w) * 10)
    eta = 2 * NormalDist().inv_cdf((0.75 * 50 - 0.125) / 50.25)  # 1.3101
    xi = 2 * NormalDist().inv_cdf((50 - 0.375) / 50.25)  # 4.4867
    assert s2.sd == pytest.approx(5 / eta, rel=1e-6)

    s1 = mean_sd_from_median(50, median=10, minimum=2, maximum=25)
    assert s1.sd == pytest.approx(23 / xi, rel=1e-6)
    assert s1.mean == pytest.approx(4 / (4 + 50 ** 0.75) * 13.5 + (1 - 4 / (4 + 50 ** 0.75)) * 10)
    assert "S1" in s1.method
    assert "S3" in mean_sd_from_median(50, 10, 2, 25, 8, 13).method
    with pytest.raises(ValueError):
        mean_sd_from_median(50, 10)


def test_median_conversion_recovers_normal_samples() -> None:
    rng = random.Random(7)
    sample = sorted(rng.gauss(20, 4) for _ in range(400))
    n = len(sample)
    q1, median, q3 = sample[n // 4], sample[n // 2], sample[3 * n // 4]
    result = mean_sd_from_median(n, median, sample[0], sample[-1], q1, q3)
    assert result.mean == pytest.approx(sum(sample) / n, abs=0.4)
    assert result.sd == pytest.approx(4, abs=0.5)


def test_standard_errors_and_sds() -> None:
    assert se_from_ci(0.5, 0.9, ratio=True) == pytest.approx((math.log(0.9) - math.log(0.5)) / (2 * 1.959964), rel=1e-5)
    assert se_from_ci(-2, 1, ratio=False) == pytest.approx(3 / (2 * 1.959964), rel=1e-5)
    assert se_from_p(0.7, 0.05, ratio=True) == pytest.approx(abs(math.log(0.7)) / 1.959964, rel=1e-5)
    assert sd_from_se(0.5, 100) == pytest.approx(5)
    # Large n: t ~ z.
    assert sd_from_ci(9.0, 11.0, 10_000) == pytest.approx(100 * 2 / (2 * 1.96), rel=1e-3)
    with pytest.raises(ValueError):
        se_from_p(1.0, 0.03, ratio=True)


def test_combining_two_arms_equals_pooling_the_raw_data() -> None:
    a = [1.0, 2.0, 3.0, 4.0]
    b = [6.0, 8.0, 10.0]

    def summary(values):
        m = sum(values) / len(values)
        return len(values), m, math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))

    n, mean, sd = combine_arms(*summary(a), *summary(b))
    assert (n, mean, sd) == pytest.approx(summary(a + b))
