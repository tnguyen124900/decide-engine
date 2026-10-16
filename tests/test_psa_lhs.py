"""Latin Hypercube sampling in run_psa (sampling="lhs")."""
import numpy as np
import pytest

from heor_tree_engine.psa import run_psa
from heor_tree_engine.sensitivity import evaluate_tree_scenario


def _doc():
    return {
        "root_id": "d",
        "parameters": [{"name": "p", "base": 0.3}, {"name": "c", "base": 100.0}, {"name": "u", "base": 0.7}],
        "nodes": [
            {"id": "d", "type": "decision", "branches": [
                {"label": "A", "child_id": "ev"}, {"label": "B", "child_id": "t_b"}]},
            {"id": "ev", "type": "chance", "branches": [
                {"label": "event", "child_id": "t_ev", "probability": "p", "cost_increment": "c"},
                {"label": "none", "child_id": "t_no", "probability": "#"}]},
            {"id": "t_ev", "type": "terminal", "cost": 0.0, "effect": 0.0, "year": 0.0, "branches": []},
            {"id": "t_no", "type": "terminal", "cost": 0.0, "effect": "u", "year": 0.0, "branches": []},
            {"id": "t_b", "type": "terminal", "cost": 10.0, "effect": 0.5, "year": 0.0, "branches": []},
        ],
    }


ROWS = [
    {"target_id": "param:p", "psa_type": "Beta", "values": {"mean": 0.3, "se": 0.05}},
    {"target_id": "param:c", "psa_type": "Gamma", "values": {"mean": 100.0, "se": 20.0}},
    {"target_id": "param:u", "psa_type": "Uniform", "values": {"min": 0.5, "max": 0.9}},
]


def _run(sampling, seed=1, n=200, rows=ROWS, **kw):
    return run_psa(_doc(), rows, iterations=n, evaluate_scenario=evaluate_tree_scenario,
                   seed=seed, sampling=sampling, **kw)


def test_each_column_has_one_draw_per_stratum() -> None:
    n = 200
    res = _run("lhs", n=n)
    u = res.samples[:, 2]  # Uniform(0.5, 0.9): stratum index must be a permutation of 0..n-1
    strata = np.floor((u - 0.5) / 0.4 * n).astype(int)
    assert sorted(strata.tolist()) == list(range(n))
    # Beta column: its CDF values are also one-per-stratum
    from scipy import stats
    a, b = 0.3 * (0.3 * 0.7 / 0.05**2 - 1), 0.7 * (0.3 * 0.7 / 0.05**2 - 1)
    cdf = stats.beta.cdf(res.samples[:, 0], a, b)
    assert sorted(np.floor(cdf * n).astype(int).tolist()) == list(range(n))


def test_random_sampling_is_unchanged_and_default() -> None:
    a = _run("random", seed=7, n=100)
    b = run_psa(_doc(), ROWS, iterations=100, evaluate_scenario=evaluate_tree_scenario, seed=7)
    assert np.array_equal(a.samples, b.samples)
    assert np.array_equal(a.costs, b.costs)


def test_lhs_is_reproducible_and_differs_from_random() -> None:
    a, b = _run("lhs", seed=3, n=100), _run("lhs", seed=3, n=100)
    assert np.array_equal(a.samples, b.samples)
    assert not np.array_equal(a.samples, _run("random", seed=3, n=100).samples)


def test_lhs_estimates_the_mean_more_tightly_than_random() -> None:
    def spread(method):
        means = [float(np.mean(_run(method, seed=s, n=50).costs[:, 0])) for s in range(40)]
        return float(np.std(means))
    assert spread("lhs") < 0.6 * spread("random")


def test_lhs_with_correlation_keeps_marginals_and_induces_rank_correlation() -> None:
    corr = [{"target_a": "param:p", "target_b": "param:c", "rho": 0.8}]
    n = 400
    res = _run("lhs", n=n, correlations=corr)
    u = res.samples[:, 2]
    assert sorted(np.floor((u - 0.5) / 0.4 * n).astype(int).tolist()) == list(range(n))
    from scipy import stats
    assert stats.spearmanr(res.samples[:, 0], res.samples[:, 1])[0] > 0.6


def test_lhs_parallel_matches_sequential_parameters() -> None:
    seq = _run("lhs", seed=5, n=60)
    par = _run("lhs", seed=5, n=60, workers=2)
    assert np.allclose(seq.samples, par.samples)


def test_unknown_sampling_is_rejected() -> None:
    from heor_tree_engine.psa import PSAInputError
    with pytest.raises(PSAInputError):
        _run("sobol")
