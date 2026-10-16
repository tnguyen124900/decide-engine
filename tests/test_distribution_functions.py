import math

import pytest

from heor_tree_engine import Param, evaluate_expression


NORMAL = {
    "normal_dist": {
        "name": "normal_dist",
        "type": "Normal",
        "parameters": {"mu": 0, "sigma": 1},
    }
}


def test_distribution_mean_cdf_and_ppf() -> None:
    parameters = {}
    assert evaluate_expression("DistMean(normal_dist)", parameters, distributions=NORMAL) == pytest.approx(0.0)
    assert evaluate_expression("DistCDF(normal_dist, 0)", parameters, distributions=NORMAL) == pytest.approx(0.5)
    assert evaluate_expression("DistPPF(normal_dist, 0.5)", parameters, distributions=NORMAL) == pytest.approx(0.0)


def test_distribution_survival_and_hazard() -> None:
    parameters = {}
    assert evaluate_expression("DistSurv(normal_dist, 0)", parameters, distributions=NORMAL) == pytest.approx(0.5)
    assert evaluate_expression("DistHazard(normal_dist, 0)", parameters, distributions=NORMAL) == pytest.approx(
        math.sqrt(2 / math.pi)
    )


def test_distribution_to_prob_is_conditional_transition_probability() -> None:
    parameters = {}
    result = evaluate_expression("DistToProb(normal_dist, 0, 1)", parameters, distributions=NORMAL)
    expected = (0.8413447460685429 - 0.5) / 0.5
    assert result == pytest.approx(expected)


def test_distribution_parameters_can_reference_parameters() -> None:
    parameters = {"mu_value": Param(name="mu_value", base=2.0)}
    distribution = {
        "normal_dist": {
            "name": "normal_dist",
            "type": "Normal",
            "parameters": {"mu": "mu_value", "sigma": 1},
        }
    }
    assert evaluate_expression("DistMean(normal_dist)", parameters, distributions=distribution) == pytest.approx(2.0)


WEIBULL = {
    "prog": {"name": "prog", "type": "Weibull", "parameters": {"shape": 1.3, "scale": 6}},
}


def test_dist_to_time_is_the_time_counterpart_of_dist_to_prob() -> None:
    """DistToTime(d, t1) draws the remaining time for someone event-free at
    t1; the share of draws <= t2 equals DistToProb(d, t1, t2) -- one formula
    (distributions.conditional_survival), used for a time vs a probability."""
    import numpy as np

    rng = np.random.default_rng(1)
    probability = evaluate_expression("DistToProb(prog, 2, 1)", {}, distributions=WEIBULL)
    draws = [
        evaluate_expression("DistToTime(prog, 2)", {}, {"__rng__": rng}, distributions=WEIBULL)
        for _ in range(20000)
    ]
    assert min(draws) >= 0
    assert sum(d <= 1 for d in draws) / len(draws) == pytest.approx(probability, abs=0.008)
    # t1 omitted = unconditional: mean of Weibull(1.3, 6)
    unconditional = [
        evaluate_expression("DistToTime(prog)", {}, {"__rng__": rng}, distributions=WEIBULL) for _ in range(20000)
    ]
    assert sum(unconditional) / len(unconditional) == pytest.approx(6 * math.gamma(1 + 1 / 1.3), rel=0.02)


def test_dist_to_time_is_reproducible_with_a_seeded_stream() -> None:
    import numpy as np

    first = evaluate_expression("DistToTime(prog, 2)", {}, {"__rng__": np.random.default_rng(7)}, distributions=WEIBULL)
    second = evaluate_expression("DistToTime(prog, 2)", {}, {"__rng__": np.random.default_rng(7)}, distributions=WEIBULL)
    assert first == second
