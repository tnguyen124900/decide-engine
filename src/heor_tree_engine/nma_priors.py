"""Informative predictive distributions for the between-study variance tau^2.

Turner et al. 2015 (Stat Med 34:984-98; update of Turner 2012, Int J
Epidemiol 41:818-27) fitted log-normal distributions to tau^2 of binary
(log odds ratio) meta-analyses in the Cochrane Database, by outcome type
and intervention-comparison type. Rhodes et al. 2015 (J Clin Epidemiol
68:52-60) did the same for continuous outcomes on the standardised mean
difference scale with log-t distributions.

The prior is placed on tau^2 (log tau^2 ~ Normal(mu, sigma^2) or
t_df(mu, sigma)). The samplers work with tau, so the density used is
p(tau) = p_{tau^2}(tau^2) * 2 tau; for the log-normal case this gives
log p(tau) = -(2 log tau - mu)^2 / (2 sigma^2) + const.

`TABLE` holds the published cells (Turner 2015 Table IV: 80 binary cells; Rhodes 2015 Table 3: 24 continuous cells, medical areas other than cancer and respiratory diseases);
a (outcome, comparison) pair that is not in the table falls back to the
general distribution of that scale (`general_entry`) and `lookup` says so
(`specific=False`). Add cells as {"scale": {"outcome|comparison": [mu,
sigma, df]}} in `priors/tau_priors.json` next to the engine (loaded on
import) - the file wins over the built-in table.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

BINARY_OUTCOMES = [
    "All-cause mortality",
    "Semi-objective: obstetric outcomes",
    "Semi-objective: cause-specific mortality / major morbidity event / composite",
    "Semi-objective: resource use / hospital stay / process",
    "Semi-objective: surgical / device-related success or failure",
    "Semi-objective: withdrawals / drop-outs",
    "Semi-objective: internal / external structure-related outcome",
    "Subjective: general physical health indicators",
    "Subjective: adverse events",
    "Subjective: infection / onset of new disease",
    "Subjective: signs / symptoms reflecting continuation or end of condition",
    "Subjective: pain",
    "Subjective: quality of life / functioning",
    "Subjective: mental health indicators",
    "Subjective: biological markers",
    "Subjective: various subjectively measured outcomes",
]
CONTINUOUS_OUTCOMES = [  # Rhodes 2015 Table 3 (grouped outcome types)
    "Obstetric outcome",
    "Resource use and hospital stay / process",
    "Internal and external structure-related outcome",
    "General physical health, adverse event, pain and quality of life / functioning",
    "Signs / symptoms reflecting continuation or end of condition and infection / onset of new acute / chronic disease",
    "Mental health outcome",
    "Biological marker",
    "Various subjectively measured outcomes",
]
CONTINUOUS_COMPARISONS = [  # Rhodes 2015 Table 3 columns (non-pharmacologic is not split further)
    "Pharmacologic vs placebo / control",
    "Pharmacologic vs pharmacologic",
    "Nonpharmacologic (any)",
]
COMPARISONS = [
    "Pharmacological vs placebo / control",
    "Pharmacological vs pharmacological",
    "Non-pharmacological vs placebo / control",
    "Non-pharmacological vs pharmacological",
    "Non-pharmacological vs non-pharmacological",
]
GENERAL = "General (all outcome and comparison types)"
SCALES = ("binary", "continuous")
SOURCES = {"binary": "Turner et al. 2015", "continuous": "Rhodes et al. 2015"}


@dataclass(frozen=True)
class TauPrior:
    scale: str  # "binary" (log OR) or "continuous" (SMD)
    outcome: str
    comparison: str
    mu: float  # location of log(tau^2)
    sigma: float  # scale of log(tau^2)
    df: float | None  # None: log-normal; else log-t with this many df
    specific: bool  # False: general distribution used instead of the requested cell
    source: str

    @property
    def family(self) -> str:
        return "log-normal" if self.df is None else f"log-t ({self.df:g} df)"

    def tau_quantile(self, q: float) -> float:
        """Quantile of tau under the prior (log-normal: exact; log-t: t quantile)."""

        from .nma_freq import t_quantile

        z = _normal_quantile(q) if self.df is None else t_quantile(q, int(self.df))
        return math.exp(0.5 * (self.mu + self.sigma * z))

    def describe(self) -> str:
        lo, hi = self.tau_quantile(0.025), self.tau_quantile(0.975)
        return (f"log τ² ~ {self.family}(μ = {self.mu:g}, σ = {self.sigma:g}); "
                f"τ median {self.tau_quantile(0.5):.3f}, 95% range {lo:.3f}–{hi:.2f}")


def _normal_quantile(q: float) -> float:
    # Acklam's rational approximation (|error| < 1.2e-9)
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00)
    if q <= 0.02425:
        t = math.sqrt(-2 * math.log(q))
        return (((((c[0] * t + c[1]) * t + c[2]) * t + c[3]) * t + c[4]) * t + c[5]) / ((((d[0] * t + d[1]) * t + d[2]) * t + d[3]) * t + 1)
    if q >= 1 - 0.02425:
        return -_normal_quantile(1 - q)
    t = q - 0.5
    r = t * t
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * t / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


# Built-in cells: (mu, sigma, df). Only the general distributions are
# shipped; the outcome x comparison cells are loaded from priors/tau_priors.json
# once transcribed from the papers (Turner 2015 Table 4; Rhodes 2015 Table 3).
TABLE: dict[str, dict[str, tuple[float, float, float | None]]] = {
    "binary": {f"{GENERAL}|{GENERAL}": (-2.56, 1.74, None)},
    "continuous": {f"{GENERAL}|{GENERAL}": (-3.44, 2.59, 5.0)},
}
TABLE_FILE = Path(__file__).resolve().parents[2] / "priors" / "tau_priors.json"


def load_table(path: Path | None = None) -> None:
    """Merge the JSON table (if present) into TABLE."""

    path = path or TABLE_FILE
    if not path.exists():
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    for scale, cells in data.items():
        if scale not in SCALES or not isinstance(cells, dict):
            continue
        for key, values in cells.items():
            if isinstance(values, (list, tuple)) and len(values) >= 2:
                mu, sigma = float(values[0]), float(values[1])
                df = float(values[2]) if len(values) > 2 and values[2] is not None else None
                TABLE[scale][key] = (mu, sigma, df)


load_table()


def general_entry(scale: str) -> TauPrior:
    mu, sigma, df = TABLE[scale][f"{GENERAL}|{GENERAL}"]
    return TauPrior(scale, GENERAL, GENERAL, mu, sigma, df, True, SOURCES[scale])


def lookup(scale: str, outcome: str = GENERAL, comparison: str = GENERAL) -> TauPrior:
    """The published cell, or the general distribution with specific=False."""

    if scale not in SCALES:
        raise ValueError(f"No informative tau prior for scale '{scale}'")
    cell = TABLE[scale].get(f"{outcome}|{comparison}")
    if cell is None and outcome == GENERAL and comparison == GENERAL:
        cell = TABLE[scale][f"{GENERAL}|{GENERAL}"]
    if cell is None:
        base = general_entry(scale)
        return TauPrior(scale, outcome, comparison, base.mu, base.sigma, base.df, False, base.source)
    mu, sigma, df = cell
    return TauPrior(scale, outcome, comparison, mu, sigma, df, True, SOURCES[scale])


def scale_for(outcome_type: str, measure: str) -> tuple[str | None, str]:
    """Which table applies to an analysis scale, and a note when it is only
    approximate ("" when exact)."""

    if outcome_type == "binary" and measure == "OR":
        return "binary", ""
    if outcome_type == "contrast" and measure in ("OR",):
        return "binary", ""
    if outcome_type == "continuous" and measure == "SMD":
        return "continuous", ""
    if measure in ("RR", "HR", "RATE_RATIO") or (outcome_type == "rate"):
        return "binary", f"Turner 2015 was derived for log odds ratios; applied to log {measure} as an approximation."
    if measure == "MD":
        return "continuous", ("Rhodes 2015 is on the SMD scale; the MD prior below is rescaled by the pooled SD "
                              "of the data (τ_MD = τ_SMD × SD), an approximation.")
    return None, f"No informative prior is available for {measure}."


def log_prior_tau(tau: float, mu: float, sigma: float, df: float | None) -> float:
    """log p(tau) (up to a constant) for log tau^2 ~ N(mu, sigma^2) or t_df(mu, sigma)."""

    if tau <= 0:
        return -math.inf
    z = (2.0 * math.log(tau) - mu) / sigma
    if df is None:
        return -0.5 * z * z
    return -0.5 * (df + 1.0) * math.log1p(z * z / df)
