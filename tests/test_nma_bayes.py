import math

import numpy as np
import pytest

from heor_tree_engine.nma_bayes import (
    BayesSettings,
    _Spectral,
    _sample_tau,
    _split_rhat,
    run_nma_bayesian,
    tau_grid_posterior,
)
from heor_tree_engine.nma_freq import _assemble, build_study_blocks, run_nma_frequentist
from heor_tree_engine.nma_models import NMAResult
from heor_tree_engine.nma_parser import resolve_nma


def _document():
    def arms(*rows):
        return [{"treatment": t, "events": r, "n": n} for t, r, n in rows]

    return {
        "spec": {"outcome": "binary", "reference": "Placebo"},
        "studies": [
            {"id": "S1", "arms": arms(("Placebo", 18, 120), ("Drug A", 11, 118))},
            {"id": "S2", "arms": arms(("Placebo", 25, 200), ("Drug A", 14, 196))},
            {"id": "S3", "arms": arms(("Placebo", 15, 100), ("Drug A", 10, 98))},
            {"id": "S4", "arms": arms(("Placebo", 30, 150), ("Drug B", 20, 148))},
            {"id": "S5", "arms": arms(("Placebo", 22, 130), ("Drug B", 8, 131))},
            {"id": "S6", "arms": arms(("Placebo", 16, 90), ("Drug C", 13, 92))},
            {"id": "S7", "arms": arms(("Drug A", 12, 90), ("Drug B", 15, 92), ("Drug C", 17, 91))},
        ],
    }


def _matrices(model):
    blocks = build_study_blocks(model, NMAResult())
    parameters = [t for t in model.treatment_names if t != model.reference]
    return _assemble(blocks, parameters)


def test_fixed_effect_posterior_matches_gls_under_vague_prior() -> None:
    model = resolve_nma(_document())
    freq = run_nma_frequentist(model)
    bayes = run_nma_bayesian(model, BayesSettings(effects="fixed", chains=4, draws_per_chain=10000, seed=1))

    assert bayes.errors == []
    draws = bayes.draws
    for column, effect in enumerate(freq.fixed.effects):
        mc_error = effect.se / math.sqrt(draws.shape[0])
        assert draws[:, column].mean() == pytest.approx(effect.estimate, abs=5 * mc_error)
        assert draws[:, column].std() == pytest.approx(effect.se, rel=0.02)
    # pD of a fixed-effect normal model = number of parameters.
    assert bayes.p_d == pytest.approx(3.0, abs=0.15)
    assert bayes.tau is None and bayes.fit.model == "bayesian-fixed"


def test_tau_marginal_likelihood_matches_direct_multivariate_normal() -> None:
    y, X, V, C = _matrices(resolve_nma(_document()))
    prior_sd = 10.0
    spectral = _Spectral(y, X, V, C, prior_sd)

    def direct(tau):
        cov = V + tau * tau * C + prior_sd ** 2 * X @ X.T
        sign, log_det = np.linalg.slogdet(cov)
        return -0.5 * (log_det + y @ np.linalg.solve(cov, y))

    taus = [0.0, 0.1, 0.4, 1.3]
    ours = [spectral.conditional(t)[3] for t in taus]
    theirs = [direct(t) for t in taus]
    for i in range(1, len(taus)):
        assert ours[i] - ours[0] == pytest.approx(theirs[i] - theirs[0], abs=1e-8)


def test_tau_sampler_reproduces_grid_density() -> None:
    model = resolve_nma(_document())
    y, X, V, C = _matrices(model)
    settings = BayesSettings()
    grid, density = tau_grid_posterior(y, X, V, C, settings)
    draws = _sample_tau(grid, density, 200000, np.random.default_rng(3))

    step = grid[1] - grid[0]
    mean = float(np.sum(0.5 * (grid[:-1] * density[:-1] + grid[1:] * density[1:]) * step))
    assert draws.mean() == pytest.approx(mean, rel=0.01)
    assert draws.min() >= 0 and draws.max() <= settings.tau_max


def test_rank_probabilities_and_sucra_are_consistent() -> None:
    result = run_nma_bayesian(resolve_nma(_document()), BayesSettings(seed=5))
    n = len(result.treatments)
    matrix = np.array([result.rank_probabilities[t] for t in result.treatments])

    assert np.allclose(matrix.sum(axis=1), 1.0) and np.allclose(matrix.sum(axis=0), 1.0)
    assert sum(result.sucra.values()) / n == pytest.approx(0.5, abs=1e-12)
    for name in result.treatments:
        assert result.sucra[name] == pytest.approx((n - result.mean_rank[name]) / (n - 1), abs=1e-12)
    assert result.fit.p_scores == result.sucra
    best = max(result.sucra, key=result.sucra.get)
    assert best != "Placebo"


def test_random_effects_run_is_reproducible_and_diagnostics_are_clean() -> None:
    model = resolve_nma(_document())
    first = run_nma_bayesian(model, BayesSettings(seed=11))
    second = run_nma_bayesian(model, BayesSettings(seed=11))

    assert np.array_equal(first.draws, second.draws)
    assert first.tau is not None and 0 <= first.tau.lower < first.tau.median < first.tau.upper <= 5
    for diag in first.diagnostics:
        assert diag.rhat < 1.01
        assert diag.ess > 0.5 * 4000
    assert first.dic == pytest.approx(first.residual_deviance + first.p_d)
    assert first.n_data_points == 8
    league = first.fit.league[("Drug A", "Drug B")]
    assert league.natural_lower < league.natural < league.natural_upper


def test_random_effects_posterior_centres_near_frequentist_estimate() -> None:
    model = resolve_nma(_document())
    freq = run_nma_frequentist(model).random
    bayes = run_nma_bayesian(model, BayesSettings(seed=2, draws_per_chain=4000))
    for effect, freq_effect in zip(bayes.fit.effects, freq.effects):
        # Bayesian RE integrates over tau, so it is close to but wider than DL.
        assert effect.estimate == pytest.approx(freq_effect.estimate, abs=0.5 * freq_effect.se)
        assert effect.natural_upper - effect.natural_lower >= 0.9 * (freq_effect.natural_upper - freq_effect.natural_lower)


def test_half_normal_prior_and_bad_settings() -> None:
    model = resolve_nma(_document())
    result = run_nma_bayesian(model, BayesSettings(tau_prior="half-normal", tau_scale=0.3))
    assert result.errors == [] and result.tau.upper <= 1.5
    assert any(e.code == "unknown_tau_prior" for e in run_nma_bayesian(model, BayesSettings(tau_prior="gamma")).errors)


def test_split_rhat_flags_disagreeing_chains() -> None:
    rng = np.random.default_rng(0)
    good = rng.normal(size=(4, 1000))
    bad = good + np.array([[0.0], [0.0], [0.0], [3.0]])
    assert _split_rhat(good) < 1.01
    assert _split_rhat(bad) > 1.1
