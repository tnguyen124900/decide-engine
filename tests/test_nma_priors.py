"""Informative tau^2 priors (Turner 2015 / Rhodes 2015): lookup, fallback,
quantiles, the density used by the samplers and the effect on a small
network in both the normal-approximation and the exact sampler."""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import nma_editor_support as sup  # noqa: E402
from heor_tree_engine import nma_priors  # noqa: E402
from heor_tree_engine.nma_bayes import BayesSettings, _log_tau_prior, run_nma_bayesian, tau_prior_upper  # noqa: E402
from heor_tree_engine.nma_bayes_exact import run_nma_bayesian_exact  # noqa: E402
from heor_tree_engine.nma_parser import resolve_nma  # noqa: E402


def test_lookup_and_fallback() -> None:
    general = nma_priors.lookup("binary")
    assert general.specific and general.df is None and general.source.startswith("Turner")
    cell = nma_priors.lookup("binary", nma_priors.BINARY_OUTCOMES[0], nma_priors.COMPARISONS[0])
    assert cell.specific and (cell.mu, cell.sigma) != (general.mu, general.sigma)  # Turner cell shipped
    base = nma_priors.lookup("continuous")
    missing = nma_priors.lookup("continuous", "Pain (not a Rhodes outcome group)", nma_priors.CONTINUOUS_COMPARISONS[0])
    assert (missing.mu, missing.sigma, missing.df) == (base.mu, base.sigma, base.df)  # unknown cell -> general
    assert missing.specific is False
    cont = nma_priors.lookup("continuous")
    assert cont.df == 5 and "log-t" in cont.family
    with pytest.raises(ValueError):
        nma_priors.lookup("rate")
    assert nma_priors.scale_for("binary", "OR") == ("binary", "")
    assert nma_priors.scale_for("continuous", "SMD") == ("continuous", "")
    assert nma_priors.scale_for("binary", "RR")[0] == "binary" and "approximation" in nma_priors.scale_for("binary", "RR")[1]
    assert nma_priors.scale_for("continuous", "MD")[0] == "continuous"
    assert nma_priors.scale_for("binary", "RD")[0] is None


def test_quantiles_and_density() -> None:
    prior = nma_priors.lookup("binary")
    # log tau^2 ~ N(mu, sigma^2): median tau = exp(mu / 2)
    assert prior.tau_quantile(0.5) == pytest.approx(math.exp(prior.mu / 2), rel=1e-6)
    assert prior.tau_quantile(0.975) == pytest.approx(math.exp((prior.mu + 1.959964 * prior.sigma) / 2), rel=1e-5)
    assert "τ median" in prior.describe()
    # density used by the samplers: log p(tau) = -(2 log tau - mu)^2 / (2 sigma^2)
    settings = BayesSettings(tau_prior="lognormal", tau_mu=prior.mu, tau_sigma=prior.sigma)
    taus = np.array([0.05, 0.2, 1.0])
    expected = -0.5 * ((2 * np.log(taus) - prior.mu) / prior.sigma) ** 2
    assert np.allclose(_log_tau_prior(taus, settings), expected)
    assert _log_tau_prior(np.array([0.0]), settings)[0] == -math.inf
    for t in taus:
        assert nma_priors.log_prior_tau(float(t), prior.mu, prior.sigma, None) == pytest.approx(
            -0.5 * ((2 * math.log(t) - prior.mu) / prior.sigma) ** 2)
    assert nma_priors.log_prior_tau(0.2, -3.44, 2.59, 5) == pytest.approx(
        -3.0 * math.log1p(((2 * math.log(0.2) + 3.44) / 2.59) ** 2 / 5))
    assert 1.0 < tau_prior_upper(settings) < 50.0
    assert nma_priors._normal_quantile(0.975) == pytest.approx(1.959964, abs=1e-6)


def _small_network():
    document = sup.default_document()
    document["studies"] = document["studies"][:4]
    used = {arm["treatment"] for study in document["studies"] for arm in study["arms"]}
    document["treatments"] = [t for t in document["treatments"] if t["name"] in used]
    return resolve_nma(document)


def test_informative_prior_tightens_tau_in_both_samplers() -> None:
    model = _small_network()
    vague = BayesSettings(effects="random", tau_prior="uniform", chains=4, draws_per_chain=1200, seed=1)
    informative = BayesSettings(effects="random", tau_prior="lognormal", tau_mu=-2.56, tau_sigma=1.74,
                                chains=4, draws_per_chain=1200, seed=1)
    normal_v, normal_i = run_nma_bayesian(model, vague), run_nma_bayesian(model, informative)
    exact_i = run_nma_bayesian_exact(model, informative, warmup=600)
    assert normal_i.errors == [] and exact_i.errors == []
    assert normal_i.tau.upper < normal_v.tau.upper * 0.6  # the prior cuts the long upper tail
    assert normal_i.tau.lower > normal_v.tau.lower
    assert exact_i.tau.median == pytest.approx(normal_i.tau.median, abs=0.12)
    assert exact_i.tau.upper == pytest.approx(normal_i.tau.upper, rel=0.35)
    assert max(d.rhat for d in exact_i.diagnostics) < 1.05
    logt = BayesSettings(effects="random", tau_prior="lognormal", tau_mu=-3.44, tau_sigma=2.59, tau_df=5,
                         chains=4, draws_per_chain=1200, seed=1)
    assert run_nma_bayesian(model, logt).tau.median == pytest.approx(
        run_nma_bayesian_exact(model, logt, warmup=600).tau.median, abs=0.12)


def test_pooled_sd_for_md_rescaling() -> None:
    document = {"studies": [{"id": "a", "arms": [{"treatment": "P", "mean": 1, "sd": 2.0, "n": 11},
                                                  {"treatment": "A", "mean": 2, "sd": 4.0, "n": 21}]}]}
    assert sup.pooled_sd(document) == pytest.approx(math.sqrt((10 * 4 + 20 * 16) / 30))
    assert sup.pooled_sd({"studies": []}) is None


def test_turner_table_is_complete_and_has_the_additive_structure():
    """All 80 Turner 2015 cells are shipped (16 outcome types x 5 comparison types) and,
    as in the fitted model (outcome effect + comparison effect on log tau^2), the
    difference between two outcome rows is the same in every comparison column
    (within the rounding of the printed table)."""

    from heor_tree_engine import nma_priors as P

    cells = {(o, c): P.lookup("binary", o, c) for o in P.BINARY_OUTCOMES for c in P.COMPARISONS}
    assert len(cells) == 80
    assert all(cell.specific and cell.df is None for cell in cells.values())
    reference = P.BINARY_OUTCOMES[0]
    for outcome in P.BINARY_OUTCOMES[1:]:
        offsets = [cells[(outcome, c)].mu - cells[(reference, c)].mu for c in P.COMPARISONS]
        assert max(offsets) - min(offsets) <= 0.021, (outcome, offsets)
    # spot checks against the printed table
    assert cells[(P.BINARY_OUTCOMES[0], P.COMPARISONS[0])].mu == -3.95
    assert cells[(P.BINARY_OUTCOMES[0], P.COMPARISONS[0])].sigma == 1.34
    assert cells[(P.BINARY_OUTCOMES[10], P.COMPARISONS[3])].mu == -1.03
    assert cells[(P.BINARY_OUTCOMES[15], P.COMPARISONS[4])].sigma == 1.45


def test_rhodes_table_is_complete_and_has_the_additive_structure() -> None:
    """All 24 Rhodes 2015 cells (8 outcome groups x 3 comparison columns, log-t with 5 df) are
    shipped. In every row pharmacologic-vs-pharmacologic is 0.27-0.28 below and nonpharmacologic
    0.14 above pharmacologic-vs-placebo (the printed 3.85 for mental health / nonpharmacologic is
    -2.85: printed median 0.058 = exp(-2.85))."""

    import math

    from heor_tree_engine import nma_priors as P

    cells = {(o, c): P.lookup("continuous", o, c) for o in P.CONTINUOUS_OUTCOMES for c in P.CONTINUOUS_COMPARISONS}
    assert len(cells) == 24 and all(cell.specific and cell.df == 5 for cell in cells.values())
    for outcome in P.CONTINUOUS_OUTCOMES:
        placebo, pharma, non = (cells[(outcome, c)].mu for c in P.CONTINUOUS_COMPARISONS)
        assert abs((pharma - placebo) + 0.275) <= 0.011, outcome
        assert abs((non - placebo) - 0.14) <= 0.011, outcome
    mental = cells[("Mental health outcome", "Nonpharmacologic (any)")]
    assert mental.mu == -2.85 and mental.sigma == 1.93
    assert abs(math.exp(mental.mu) - 0.058) < 0.001  # printed median
    obstetric = cells[("Obstetric outcome", "Pharmacologic vs placebo / control")]
    assert (obstetric.mu, obstetric.sigma) == (-4.13, 2.34) and abs(math.exp(obstetric.mu) - 0.016) < 0.001
    general = P.lookup("continuous")
    assert (general.mu, general.sigma, general.df) == (-3.44, 2.59, 5)
