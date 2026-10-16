"""Value of Information (VOI) analysis from a Bayesian NMA posterior.

A decision between treatments based on an NMA is uncertain because the
network's treatment effects are estimated with error. VOI quantifies how much
that uncertainty costs, and how much a further study that shrinks it would be
worth -- the EVPI/EVPPI framework of Claxton & Sculpher, applied directly to
the NMA posterior instead of a decision model's PSA sample.

The key difference from psa.py's VOI: there is no cost/QALY model here. The
"decision" is which treatment in the network is best, and the payoff is the
treatment effect itself (or the effect converted to a net benefit at a
willingness-to-pay for a unit of effect). Two natural payoff scales:

- **effect scale** (default): the best treatment is the one with the most
  favourable effect; EVPI is in effect units (e.g. log-OR, mean difference).
  Useful when the NMA's job is purely "which treatment wins".
- **net-benefit scale**: each treatment's effect is turned into a net
  monetary benefit as `wtp * effect - cost`, where `cost` comes from an
  optional per-treatment cost table; EVPI is then in money, directly
  comparable to a decision model's EVPI.

EVPI over the network posterior answers "how much is ALL remaining parameter
uncertainty in this network worth removing?"; EVPPI per treatment effect (or
per component, in a CNMA) answers "which specific comparison's uncertainty
drives the decision risk" -- the comparison a new trial should target.

Method: EVPI is the exact two-term form E[max_t payoff_t] - max_t E[payoff_t]
over the posterior draws. EVPPI per parameter uses the same
regression-method stand-in as psa.evppi_by_parameter (a cross-validated
polynomial regression of payoff on each parameter's draws), so numbers are
directly comparable to the decision-model EVPPI.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from .nma_bayes import BayesResult
from .nma_models import NetworkMetaAnalysis, is_ratio_measure
from .psa import _polynomial_design_matrix, _select_degree_by_cv

__all__ = ["NMAVOIResult", "run_nma_voi"]


@dataclass(frozen=True)
class NMAEVPPIRow:
    """EVPPI for one posterior parameter (a treatment effect vs reference)."""

    parameter: str
    evppi: float
    percent_of_evpi: float | None
    degree_used: int


@dataclass
class NMAVOIResult:
    """VOI over an NMA posterior, on the chosen payoff scale."""

    payoff_scale: str = "effect"  # "effect" | "net_benefit"
    reference: str | None = None
    treatments: list[str] = field(default_factory=list)
    # Per-patient EVPI on the payoff scale (effect units, or money when
    # payoff_scale == "net_benefit").
    evpi: float = 0.0
    # The treatment with the highest expected payoff (the current decision).
    optimal_treatment: str | None = None
    # Per-parameter EVPPI: which treatment effect's uncertainty drives the
    # decision risk (i.e. which comparison a new trial should target).
    evppi: list[NMAEVPPIRow] = field(default_factory=list)
    n_draws: int = 0
    warnings: list[str] = field(default_factory=list)


def _treatment_payoffs(
    result: BayesResult,
    wtp: float | None,
    costs: Mapping[str, float] | None,
) -> tuple[np.ndarray, list[str]]:
    """The (n_draws, n_treatments) payoff matrix over the posterior draws.

    Each treatment's payoff per draw is its effect versus the network
    reference (0 for the reference itself by construction), exponentiated
    back to the natural scale for a ratio measure when the analysis is on a
    log scale (the decision-relevant "effect" is the natural-scale one).

    - effect scale: payoff = effect.
    - net-benefit scale: payoff = wtp * effect - cost[treatment]; requires
      `wtp` and a cost per treatment (missing cost treated as 0).
    """

    if result.draws is None:
        return np.zeros((0, 0)), []
    reference = result.reference
    treatments = list(result.treatments)
    index = {name: i for i, name in enumerate(treatments)}
    parameters = [name for name in treatments if name != reference]
    n_draws = result.draws.shape[0]

    # Full effect matrix vs reference: 0 for the reference column.
    effects = np.zeros((n_draws, len(treatments)))
    ratio = is_ratio_measure(result.measure)
    for column, name in enumerate(parameters):
        values = result.draws[:, column]
        effects[:, index[name]] = np.exp(values) if ratio else values

    if wtp is None:
        return effects, treatments
    cost_vector = np.array([float((costs or {}).get(name, 0.0)) for name in treatments])
    nmb = wtp * effects - cost_vector[None, :]
    return nmb, treatments


def run_nma_voi(
    model: NetworkMetaAnalysis,
    result: BayesResult,
    *,
    willingness_to_pay: float | None = None,
    costs: Mapping[str, float] | None = None,
    candidate_degrees: Sequence[int] = (1, 2, 3),
) -> NMAVOIResult:
    """EVPI and per-parameter EVPPI over the network's posterior.

    `willingness_to_pay` + `costs` switch the payoff to net monetary benefit
    (money); leave `willingness_to_pay=None` for the pure effect scale.

    EVPPI is reported per treatment-effect parameter (d_k vs reference) --
    the comparison whose uncertainty most drives the decision risk, i.e. the
    comparison a new trial should target.
    """

    out = NMAVOIResult(
        payoff_scale="net_benefit" if willingness_to_pay is not None else "effect",
        reference=result.reference,
    )
    if result.draws is None or result.draws.shape[0] == 0:
        out.warnings.append("No posterior draws -- run the Bayesian NMA first.")
        return out
    if willingness_to_pay is not None and willingness_to_pay <= 0:
        out.warnings.append("Willingness-to-pay must be positive for net-benefit VOI.")
        return out

    payoff, treatments = _treatment_payoffs(result, willingness_to_pay, costs)
    out.treatments = treatments
    out.n_draws = payoff.shape[0]

    # EVPI = E[max_t payoff_t] - max_t E[payoff_t].
    expected_max = float(np.mean(np.max(payoff, axis=1)))
    mean_per_treatment = np.mean(payoff, axis=0)
    max_expected = float(np.max(mean_per_treatment))
    out.evpi = max(0.0, expected_max - max_expected)
    out.optimal_treatment = treatments[int(np.argmax(mean_per_treatment))]

    # EVPPI per posterior parameter (each d_k vs reference), regression
    # method: how much of the EVPI does knowing this one parameter resolve?
    reference = result.reference
    parameters = [name for name in treatments if name != reference]
    if out.evpi > 0 and parameters:
        rows: list[NMAEVPPIRow] = []
        # Standardise draws for the polynomial fit (same as psa.py).
        for column, name in enumerate(parameters):
            theta = result.draws[:, column]
            sd = float(theta.std())
            if sd <= 0:
                rows.append(NMAEVPPIRow(parameter=name, evppi=0.0, percent_of_evpi=0.0, degree_used=0))
                continue
            theta_std = (theta - float(theta.mean())) / sd
            # Fit payoff of the best strategy against theta: EVPPI measures
            # how knowing theta changes E[max payoff | theta] (regression
            # method -- the same cross-validated polynomial psa.py uses).
            best = np.max(payoff, axis=1)
            degree = _select_degree_by_cv(theta_std.reshape(-1, 1), best.reshape(-1, 1), candidate_degrees)
            design = _polynomial_design_matrix(theta_std.reshape(-1, 1), degree)
            coefficients, *_ = np.linalg.lstsq(design, best.reshape(-1, 1), rcond=None)
            fitted = (design @ coefficients).ravel()
            resolved = float(np.mean(fitted))
            evppi_value = max(0.0, resolved - max_expected)
            rows.append(NMAEVPPIRow(
                parameter=name,
                evppi=evppi_value,
                percent_of_evpi=100.0 * evppi_value / out.evpi if out.evpi > 0 else None,
                degree_used=degree,
            ))
        rows.sort(key=lambda row: row.evppi, reverse=True)
        out.evppi = rows

    return out
