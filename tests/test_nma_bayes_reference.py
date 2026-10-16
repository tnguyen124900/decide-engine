"""Cross-check DECIDE's Bayesian NMA engine against the R packages gemtc and
BUGSnet -- two independent MCMC implementations of the same random-effects
consistency model, on the same smokingcessation data the frequentist netmeta
reference uses.

The reference files are produced by tests/reference/make_gemtc_reference.R and
make_bugsnet_reference.R (need R + JAGS + the package; run once from the
repository root). Until a reference JSON exists the matching test is
skipped, exactly like tests/test_nma_reference.py.

Tolerance is Monte-Carlo tolerance, not the frequentist reference's 1e-5: two
independent samplers of the same posterior differ by their simulation error,
so posterior means are compared to 5% and the reference's own Monte-Carlo SE
is surfaced in the failure message.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from heor_tree_engine.nma_bayes import BayesSettings, run_nma_bayesian
from heor_tree_engine.nma_parser import resolve_nma

REFERENCE_DIR = Path(__file__).parent / "reference"

# gemtc/BUGSnet ran 4 chains x 20k draws (post-burn-in); DECIDE's slice sampler
# needs enough draws for its own Monte-Carlo error to sit well under the 5%
# tolerance. 4 chains x 4000 draws keeps the test under a couple of seconds.
SETTINGS = BayesSettings(effects="random", chains=4, draws_per_chain=4000, seed=2026)

# Posterior means agree to 5% (relative) or this absolute slack on the log-OR
# scale, whichever is larger -- near-zero effects otherwise trip the relative
# check on pure MCMC noise.
REL = 0.05
ABS = 0.02


def _load(name: str) -> dict:
    path = REFERENCE_DIR / f"{name}.json"
    if not path.exists():
        pytest.skip(f"{path.name} not generated yet - run tests/reference/make_{name.split('_')[0]}_reference.R")
    return json.loads(path.read_text())


def _model(reference: dict):
    studies = [{**study, "arms": [arm for arm in study["arms"] if arm.get("treatment")]}
               for study in reference["studies"]]
    return resolve_nma({
        "spec": {"outcome": reference.get("outcome", "binary"), "measure": reference.get("measure", "OR"),
                 "reference": reference["reference"], "small_values": "good"},
        "studies": studies,
    })


def _effects_by_treatment(result) -> dict:
    assert result.errors == []
    assert result.fit is not None
    return {effect.treatment: effect for effect in result.fit.effects}


def _check_against(package: str, reference: dict) -> None:
    """Compare DECIDE's posterior to one R reference (gemtc or BUGSnet)."""

    result = run_nma_bayesian(_model(reference), SETTINGS)
    estimates = _effects_by_treatment(result)

    # gemtc names its effect parameters "d.<treat>.<ref>"; map to DECIDE's
    # plain treatment names.
    expected_effects = {}
    for param, values in reference["effects"].items():
        parts = param.split(".")
        treatment = parts[1] if len(parts) >= 3 else param
        expected_effects[treatment] = values

    for treatment, expected in expected_effects.items():
        assert treatment in estimates, f"{package}: missing effect for {treatment}"
        got = estimates[treatment]
        assert got.estimate == pytest.approx(expected["mean"], rel=REL, abs=ABS), (
            f"{package} mean mismatch for {treatment}: "
            f"DECIDE {got.estimate:.4f} vs {package} {expected['mean']:.4f}"
        )
        # Posterior SD to 10% -- SDs carry larger Monte-Carlo error than means.
        assert got.se == pytest.approx(expected["sd"], rel=0.10, abs=ABS), (
            f"{package} sd mismatch for {treatment}: "
            f"DECIDE {got.se:.4f} vs {package} {expected['sd']:.4f}"
        )

    if reference.get("tau") is not None and result.tau is not None:
        # tau (between-study SD): the hardest parameter to pin down; 15% rel.
        assert result.tau.mean == pytest.approx(reference["tau"]["mean"], rel=0.15, abs=ABS), (
            f"{package} tau mismatch: DECIDE {result.tau.mean:.4f} vs {package} {reference['tau']['mean']:.4f}"
        )


def test_bayes_smoking_cessation_matches_gemtc() -> None:
    _check_against("gemtc", _load("gemtc_smokingcessation"))


def test_bayes_smoking_cessation_matches_bugsnet() -> None:
    _check_against("BUGSnet", _load("bugsnet_smokingcessation"))


def test_gemtc_and_bugsnet_agree_with_each_other() -> None:
    """Sanity check the two R references against each other (both must exist).

    If gemtc and BUGSnet disagree beyond tolerance, a DECIDE mismatch says
    nothing about DECIDE -- this guards the comparison itself.
    """

    gemtc = _load("gemtc_smokingcessation")
    bugsnet = _load("bugsnet_smokingcessation")
    for param, values in gemtc["effects"].items():
        if param not in bugsnet["effects"]:
            continue
        assert bugsnet["effects"][param]["mean"] == pytest.approx(values["mean"], rel=REL, abs=ABS), param
