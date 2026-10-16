"""NMA -> economic evaluation: turn relative effects into model parameters.

An "evidence set" is the relative effect of each chosen treatment versus an
anchor (normally the NMA reference, e.g. placebo / standard of care) with:

- a base value (point estimate, natural scale) and low/high (CI) for
  deterministic / tornado analysis;
- a PSA distribution that reproduces the NMA uncertainty exactly: the
  analysis-scale estimate is normal (mu, se), so a ratio effect is
  lognormal - passed to DECIDE's PSA as Lognormal(mean, se) on the natural
  scale with mean = exp(mu + se^2 / 2), se = mean * sqrt(exp(se^2) - 1),
  which DECIDE's method-of-moments converts straight back to (mu, se);
  a difference (MD) is Normal(mu, se);
- the correlations between effects (they share the anchor's evidence), as
  PSA correlation rows. DECIDE's PSA applies them with Iman-Conover on
  normal scores, i.e. a Gaussian copula - which for lognormal/normal
  marginals recovers the NMA's joint distribution.

Anything else about how the effect is applied (HR on a transition
probability, OR on baseline odds, ...) stays the modeller's formula in the
EE model; `APPLY_FORMULAS` gives the standard ones as ready-to-paste text.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .nma_models import NMAFit, NMAResult, is_ratio_measure

# How each measure is normally applied to a baseline in an EE model
# (DECIDE expression syntax; `{p}` = baseline probability parameter,
# `{x}` = the NMA parameter, `{b}` = baseline value for differences).
APPLY_FORMULAS = {
    "HR": "1 - (1 - {p}) ** {x}",
    "RATE_RATIO": "1 - (1 - {p}) ** {x}",
    "OR": "logis(logit({p}) + ln({x}))",
    "RR": "min(1, {p} * {x})",
    "MD": "{b} + {x}",
    "SMD": "{b} + {x}",
    "RD": "min(1, max(0, {p} + {x}))",
}


@dataclass
class EvidenceParameter:
    name: str
    treatment: str
    versus: str
    measure: str
    base: float  # natural scale point estimate
    low: float  # natural scale CI
    high: float
    mu: float  # analysis scale
    se: float
    psa_type: str  # "Lognormal" | "Normal"
    psa_values: dict[str, float] = field(default_factory=dict)
    # Posterior draws on the natural scale (Bayesian engine), thinned; the
    # EE model's PSA draws these jointly (one index for all NMA parameters).
    samples: list[float] | None = None

    def as_parameter_row(self, model_label: str) -> dict[str, Any]:
        return {
            "name": self.name,
            "base": self.base,
            "low": self.low,
            "high": self.high,
            "source": f"NMA ({model_label} effects): {self.measure} {self.treatment} vs {self.versus}",
        }

    def as_psa_row(self) -> dict[str, Any]:
        return {"target_id": f"param:{self.name}", "psa_type": self.psa_type, "values": dict(self.psa_values)}


@dataclass
class EvidenceSet:
    measure: str
    anchor: str
    model: str  # "fixed" | "random"
    parameters: list[EvidenceParameter] = field(default_factory=list)
    correlations: list[tuple[str, str, float]] = field(default_factory=list)

    @property
    def apply_formula(self) -> str:
        return APPLY_FORMULAS.get(self.measure, "{x}")

    def to_json(self) -> dict[str, Any]:
        return {
            "measure": self.measure,
            "anchor": self.anchor,
            "model": self.model,
            "parameters": [
                {"name": p.name, "treatment": p.treatment, "versus": p.versus, "base": p.base, "low": p.low,
                 "high": p.high, "mu": p.mu, "se": p.se, "psa_type": p.psa_type, "psa_values": p.psa_values,
                 **({"samples": p.samples} if p.samples else {})}
                for p in self.parameters
            ],
            "correlations": [{"a": a, "b": b, "rho": rho} for a, b, rho in self.correlations],
            "apply_formula": self.apply_formula,
        }


def parameter_name(measure: str, treatment: str, versus: str, prefix: str = "") -> str:
    """e.g. ("HR", "Drug A", "Placebo") -> "hr_drug_a_vs_placebo" (valid
    DECIDE expression identifier: letters, digits, underscores)."""

    def slug(text: str) -> str:
        return re.sub(r"[^0-9a-zA-Z]+", "_", text).strip("_").lower() or "x"

    name = f"{prefix}{measure.lower()}_{slug(treatment)}_vs_{slug(versus)}"
    if name[0].isdigit():
        name = f"p_{name}"
    return name


def _full_effects(fit: NMAFit, treatments: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Effects of every treatment vs the NMA reference (reference = 0) and
    their full covariance, in `treatments` order."""

    index = {name: i for i, name in enumerate(treatments)}
    effects = np.zeros(len(treatments))
    covariance = np.zeros((len(treatments), len(treatments)))
    positions = [index[name] for name in fit.parameter_names]
    by_name = {effect.treatment: effect.estimate for effect in fit.effects}
    for name in fit.parameter_names:
        effects[index[name]] = by_name[name]
    covariance[np.ix_(positions, positions)] = np.array(fit.covariance)
    return effects, covariance


def build_evidence_set(result: NMAResult, model: str = "random", anchor: str | None = None,
                       treatments: list[str] | None = None, prefix: str = "",
                       draws: np.ndarray | None = None, max_samples: int = 2000) -> EvidenceSet:
    """`draws` (Bayesian engine): posterior draws of the effects vs the NMA
    reference, columns in `fit.parameter_names` order; thinned to at most
    `max_samples` joint draws and exported as empirical PSA samples."""
    fit = result.random if model == "random" else result.fixed
    if fit is None:
        raise ValueError("Run the NMA first")
    anchor = anchor or result.reference
    if anchor not in result.treatments:
        raise ValueError(f"Unknown anchor treatment '{anchor}'")
    chosen = [name for name in (treatments or result.treatments) if name != anchor]
    unknown = [name for name in chosen if name not in result.treatments]
    if unknown:
        raise ValueError(f"Unknown treatment(s): {', '.join(unknown)}")

    effects, covariance = _full_effects(fit, result.treatments)
    index = {name: i for i, name in enumerate(result.treatments)}
    # Contrast matrix: row k = e_{t_k} - e_{anchor}.
    contrast = np.zeros((len(chosen), len(result.treatments)))
    for row, name in enumerate(chosen):
        contrast[row, index[name]] += 1.0
        contrast[row, index[anchor]] -= 1.0
    mu = contrast @ effects
    cov = contrast @ covariance @ contrast.T
    sample_matrix = None
    if draws is not None and len(draws):
        draws = np.asarray(draws, dtype=float)
        full = np.zeros((draws.shape[0], len(result.treatments)))
        for column, name in enumerate(fit.parameter_names):
            full[:, index[name]] = draws[:, column]
        if draws.shape[0] > max_samples:
            keep = np.linspace(0, draws.shape[0] - 1, max_samples).round().astype(int)
            full = full[keep]
        sample_matrix = full @ contrast.T  # (n_samples, len(chosen)), analysis scale

    ratio = is_ratio_measure(result.measure)
    z = _z_from_fit(fit)
    evidence = EvidenceSet(measure=result.measure, anchor=anchor, model=fit.model)
    for row, name in enumerate(chosen):
        m = float(mu[row])
        se = math.sqrt(max(float(cov[row, row]), 0.0))
        if ratio:
            base, low, high = math.exp(m), math.exp(m - z * se), math.exp(m + z * se)
            mean = math.exp(m + se * se / 2.0)
            psa_type, psa_values = "Lognormal", {"mean": mean, "se": mean * math.sqrt(math.expm1(se * se))}
        else:
            base, low, high = m, m - z * se, m + z * se
            psa_type, psa_values = "Normal", {"mean": m, "se": se}
        samples = None
        if sample_matrix is not None:
            column = sample_matrix[:, row]
            samples = [float(v) for v in (np.exp(column) if ratio else column)]
        evidence.parameters.append(EvidenceParameter(
            name=parameter_name(result.measure, name, anchor, prefix), treatment=name, versus=anchor,
            measure=result.measure, base=base, low=low, high=high, mu=m, se=se,
            psa_type=psa_type, psa_values=psa_values, samples=samples,
        ))
    for i in range(len(chosen)):
        for j in range(i + 1, len(chosen)):
            denominator = math.sqrt(cov[i, i] * cov[j, j])
            if denominator > 0:
                rho = float(cov[i, j] / denominator)
                if abs(rho) > 1e-9:
                    evidence.correlations.append(
                        (evidence.parameters[i].name, evidence.parameters[j].name, max(-1.0, min(1.0, rho))))
    return evidence


def _z_from_fit(fit: NMAFit) -> float:
    """Recover the CI multiplier the fit used (same for every effect)."""

    for effect in fit.effects:
        if effect.se > 0:
            return (effect.upper - effect.estimate) / effect.se
    return 1.959963984540054


def merge_into_document(document: dict[str, Any], evidence: EvidenceSet, add_psa: bool = True) -> dict[str, list[str]]:
    """Add or update the evidence parameters in an EE model document
    (Tree / Markov / PSM / Microsim all share `parameters`,
    `psa_parameters`, `psa_correlations`). Existing rows with the same
    names are replaced in place; nothing else is touched. Returns the
    names added and updated."""

    parameters = document.setdefault("parameters", [])
    position = {str(row.get("name")): i for i, row in enumerate(parameters)}
    added: list[str] = []
    updated: list[str] = []
    for parameter in evidence.parameters:
        row = parameter.as_parameter_row(evidence.model)
        if parameter.name in position:
            parameters[position[parameter.name]] = row
            updated.append(parameter.name)
        else:
            position[parameter.name] = len(parameters)
            parameters.append(row)
            added.append(parameter.name)

    if add_psa:
        names = {parameter.name for parameter in evidence.parameters}
        target_ids = {f"param:{name}" for name in names}
        psa_rows = [row for row in document.get("psa_parameters", []) if row.get("target_id") not in target_ids]
        psa_rows.extend(parameter.as_psa_row() for parameter in evidence.parameters)
        document["psa_parameters"] = psa_rows
        correlations = [
            row for row in document.get("psa_correlations", [])
            if row.get("target_a") not in target_ids and row.get("target_b") not in target_ids
        ]
        correlations.extend({"target_a": f"param:{a}", "target_b": f"param:{b}", "rho": round(rho, 6)}
                            for a, b, rho in evidence.correlations)
        document["psa_correlations"] = correlations
        empirical = {key: value for key, value in (document.get("psa_empirical") or {}).items()
                     if key not in target_ids}
        if any(parameter.samples for parameter in evidence.parameters):
            joint = "nma:" + ",".join(sorted(names))
            for parameter in evidence.parameters:
                if parameter.samples:
                    empirical[f"param:{parameter.name}"] = {
                        "samples": parameter.samples, "joint": joint,
                        "signature": {"psa_type": parameter.psa_type, "values": dict(parameter.psa_values)},
                        "source": f"NMA posterior ({evidence.model} effects), {len(parameter.samples)} draws",
                    }
        if empirical:
            document["psa_empirical"] = empirical
        else:
            document.pop("psa_empirical", None)
    return {"added": added, "updated": updated}
