"""Confidence in NMA results - CINeMA framework (Nikolakopoulou et al. 2020,
PLoS Med 17:e1003082; Papakonstantinou et al. 2020, Campbell Syst Rev).

Every network estimate (treatment vs versus) is judged on six domains:

1. Within-study bias   - study risk of bias weighted by the study's
                         contribution to the estimate (contribution matrix).
2. Reporting bias      - small-study effects: comparison-adjusted funnel plot
                         and Egger's test (network level).
3. Indirectness        - study-level indirectness (population / intervention /
                         outcome differ from the decision question), weighted
                         like risk of bias.
4. Imprecision         - the confidence interval against the range of clinical
                         equivalence (1/m, m) for ratios, (-m, m) otherwise.
5. Heterogeneity       - whether the prediction interval crosses more
                         equivalence boundaries than the confidence interval.
6. Incoherence         - node-splitting p-value for comparisons with direct and
                         indirect evidence, the design-by-treatment test
                         otherwise.

Each domain is rated No / Some / Major concerns (reporting bias: Undetected /
Suspected). The suggested confidence starts at High and drops one level per
"Some concerns" or "Suspected" and two per "Major concerns" (floor Very low).
CINeMA stresses that domains are related (imprecision, heterogeneity and
incoherence overlap) so any suggestion can be overridden with a reason.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from .nma_contribution import ContributionResult, contribution_matrix
from .nma_funnel import FunnelResult, comparison_adjusted_funnel
from .nma_inconsistency import InconsistencyResult
from .nma_models import NetworkMetaAnalysis, NMAResult, is_ratio_measure

DOMAINS = ("within_study_bias", "reporting_bias", "indirectness", "imprecision", "heterogeneity", "incoherence")
DOMAIN_LABELS = {
    "within_study_bias": "Within-study bias",
    "reporting_bias": "Reporting bias",
    "indirectness": "Indirectness",
    "imprecision": "Imprecision",
    "heterogeneity": "Heterogeneity",
    "incoherence": "Incoherence",
}
NO, SOME, MAJOR = "No concerns", "Some concerns", "Major concerns"
UNDETECTED, SUSPECTED = "Undetected", "Suspected"
CONCERN_LEVELS = (NO, SOME, MAJOR)
REPORTING_LEVELS = (UNDETECTED, SUSPECTED)
CONFIDENCE = ("Very low", "Low", "Moderate", "High")
DOWNGRADE = {NO: 0, SOME: 1, MAJOR: 2, UNDETECTED: 0, SUSPECTED: 1}
RULES = ("average", "majority")
LEVEL_SCORE = {"Low": 1, "Some concerns": 2, "High": 3}
SCORE_LEVEL = {1: NO, 2: SOME, 3: MAJOR}


def default_margin(measure: str) -> float:
    """1.25 for ratios (0.8-1.25); 0 = no margin (use the line of no effect)."""

    return 1.25 if is_ratio_measure(measure) else 0.0


def normalize_level(value: Any) -> str:
    """'high', 'H', 'high risk' -> 'High'; 'some'/'unclear'/'moderate' -> 'Some concerns'; 'low' -> 'Low'."""

    text = "" if value is None else str(value).strip().lower()
    if not text or text == "nan":
        return ""
    if text.startswith(("h", "major", "serious")):
        return "High"
    if text.startswith(("l", "no")):
        return "Low"
    if text.startswith(("s", "u", "m")):
        return "Some concerns"
    return ""


def comparison_key(treatment: str, versus: str) -> str:
    return f"{treatment}|{versus}"


@dataclass
class Judgement:
    level: str
    reason: str
    auto_level: str = ""  # suggestion before any override
    overridden: bool = False


@dataclass
class CinemaRow:
    treatment: str
    versus: str
    estimate: float
    lower: float
    upper: float
    natural: float
    natural_lower: float
    natural_upper: float
    natural_pred_lower: float | None
    natural_pred_upper: float | None
    n_direct_studies: int
    direct_share: float
    domains: dict[str, Judgement] = field(default_factory=dict)
    auto_confidence: str = "High"
    confidence: str = "High"
    confidence_overridden: bool = False
    note: str = ""

    @property
    def label(self) -> str:
        return f"{self.treatment} vs {self.versus}"

    @property
    def key(self) -> str:
        return comparison_key(self.treatment, self.versus)

    @property
    def downgrades(self) -> int:
        return sum(DOWNGRADE.get(judgement.level, 0) for judgement in self.domains.values())


@dataclass
class CinemaResult:
    measure: str = "OR"
    model: str = "random"
    margin: float = 0.0
    rob_rule: str = "average"
    rows: list[CinemaRow] = field(default_factory=list)
    contribution: ContributionResult | None = None
    funnel: FunnelResult | None = None
    warnings: list[str] = field(default_factory=list)
    errors: list = field(default_factory=list)


# --------------------------------------------------------------------------
# Domain rules
# --------------------------------------------------------------------------


def _weighted_level(studies: dict[str, float], levels: dict[str, str], missing: str, rule: str) -> tuple[str, dict[str, float]]:
    """Returns (concern level, contribution share per study level)."""

    shares = {"Low": 0.0, "Some concerns": 0.0, "High": 0.0, "": 0.0}
    for study_id, weight in studies.items():
        shares[levels.get(study_id, "") or ""] += weight
    total = sum(shares.values()) or 1.0
    shares = {key: value / total for key, value in shares.items()}
    effective = dict(shares)
    effective[missing] += effective.pop("")
    if rule == "majority":
        best = max(("High", "Some concerns", "Low"), key=lambda level: (round(effective[level], 9), LEVEL_SCORE[level]))
        return {"Low": NO, "Some concerns": SOME, "High": MAJOR}[best], shares
    score = sum(LEVEL_SCORE[level] * effective[level] for level in ("Low", "Some concerns", "High"))
    level = NO if score < 1.5 else (SOME if score < 2.5 else MAJOR)
    return level, shares


def _share_text(shares: dict[str, float]) -> str:
    parts = [f"{label} {shares[key] * 100:.0f}%" for key, label in
             (("Low", "low"), ("Some concerns", "some concerns"), ("High", "high"), ("", "not entered"))
             if shares.get(key, 0) >= 0.005]
    return ", ".join(parts)


def _thresholds(margin: float, ratio: bool) -> list[float]:
    if margin and margin > 0:
        delta = math.log(margin) if ratio else margin
        if ratio and margin < 1:
            delta = -delta
        return [-abs(delta), abs(delta)] if delta else [0.0]
    return [0.0]


def _crossed(lower: float, upper: float, thresholds: list[float]) -> int:
    return sum(1 for t in thresholds if lower < t < upper)


def _margin_text(margin: float, ratio: bool) -> str:
    if not margin or margin <= 0:
        return "the line of no effect"
    if ratio:
        high = max(margin, 1 / margin)
        return f"the equivalence range {1 / high:.2f}–{high:.2f}"
    return f"the equivalence range ±{margin:g}"


def _imprecision(lower: float, upper: float, margin: float, ratio: bool) -> Judgement:
    thresholds = _thresholds(margin, ratio)
    crossed = _crossed(lower, upper, thresholds)
    where = _margin_text(margin, ratio)
    if len(thresholds) == 1:
        if crossed:
            return Judgement(SOME, f"The CI includes no effect ({where}); no equivalence margin set.")
        return Judgement(NO, "The CI excludes no effect.")
    if crossed == 2:
        return Judgement(MAJOR, f"The CI extends into clinically important effects in both directions ({where}).")
    if crossed == 1:
        return Judgement(SOME, f"The CI crosses one boundary of {where}.")
    return Judgement(NO, f"The CI lies within or entirely outside {where}.")


def _heterogeneity(ci: tuple[float, float], pi: tuple[float | None, float | None], margin: float, ratio: bool,
                   tau2: float) -> Judgement:
    if tau2 <= 1e-12:
        return Judgement(NO, "tau² = 0: no between-study heterogeneity estimated.")
    if pi[0] is None or pi[1] is None:
        return Judgement(NO, "No prediction interval (too few studies or fixed effect); "
                             f"tau² = {tau2:.3f}. Judge by hand if heterogeneity is expected.")
    thresholds = _thresholds(margin, ratio)
    extra = _crossed(pi[0], pi[1], thresholds) - _crossed(ci[0], ci[1], thresholds)
    where = _margin_text(margin, ratio)
    if extra >= 2:
        return Judgement(MAJOR, f"The prediction interval crosses both boundaries of {where} while the CI does not.")
    if extra == 1:
        return Judgement(SOME, f"The prediction interval crosses one more boundary of {where} than the CI (tau² = {tau2:.3f}).")
    return Judgement(NO, f"The prediction interval and CI lead to the same conclusion about {where} (tau² = {tau2:.3f}).")


def _p_level(p: float | None) -> str:
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return NO
    return MAJOR if p <= 0.01 else (SOME if p <= 0.10 else NO)


def _incoherence(treatment: str, versus: str, incons: InconsistencyResult | None, model: str) -> Judgement:
    if incons is None or incons.errors:
        return Judgement(NO, "Inconsistency not run - press Inconsistency to assess.")
    splits = incons.splits_random if model == "random" else incons.splits_fixed
    split = next((s for s in splits if {s.treatment, s.versus} == {treatment, versus}), None)
    if split is not None:
        level = _p_level(split.p_value)
        return Judgement(level, f"Node-split (direct vs indirect) p = {split.p_value:.3f}.")
    dbt = incons.design_by_treatment
    if dbt is None or not dbt.df_between:
        return Judgement(NO, "No closed loops: incoherence cannot be tested.")
    p = dbt.p_between_random if model == "random" and dbt.p_between_random is not None else dbt.p_between
    return Judgement(_p_level(p), f"Only direct or only indirect evidence; design-by-treatment test p = "
                                  f"{p:.3f}." if p is not None else "Design-by-treatment test not available.")


def _reporting(funnel: FunnelResult | None) -> Judgement:
    if funnel is None or funnel.errors or funnel.egger_p is None:
        return Judgement(UNDETECTED, "Funnel plot not available.")
    n = len(funnel.points)
    text = f"Egger's test on the comparison-adjusted funnel plot p = {funnel.egger_p:.3f} ({n} points)"
    if funnel.egger_p < 0.10:
        return Judgement(SUSPECTED, text + ": small-study effects suspected.")
    if n < 10:
        return Judgement(UNDETECTED, text + "; low power below 10 points - consider unpublished trials too.")
    return Judgement(UNDETECTED, text + ".")


def _confidence(downgrades: int) -> str:
    return CONFIDENCE[max(0, 3 - downgrades)]


# --------------------------------------------------------------------------
# Main entry
# --------------------------------------------------------------------------


def run_cinema(model: NetworkMetaAnalysis, nma: NMAResult, *, effects: str = "random",
               characteristics: list[dict[str, Any]] | None = None,
               inconsistency: InconsistencyResult | None = None,
               funnel: FunnelResult | None = None,
               contribution: ContributionResult | None = None,
               margin: float | None = None, rob_rule: str = "average",
               overrides: dict[str, dict[str, str]] | None = None,
               funnel_order: list[str] | None = None) -> CinemaResult:
    ratio = is_ratio_measure(model.spec.measure)
    margin = default_margin(model.spec.measure) if margin is None else float(margin)
    result = CinemaResult(measure=model.spec.measure, model=effects, margin=margin,
                          rob_rule=rob_rule if rob_rule in RULES else "average")
    if nma.errors or nma.random is None:
        result.errors = list(nma.errors) or ["Run the NMA first."]
        return result
    fit = nma.random if effects == "random" else nma.fixed
    contribution = contribution or contribution_matrix(model, effects, nma.heterogeneity.tau2 if effects == "random" else 0.0)
    if contribution.errors:
        result.errors = contribution.errors
        return result
    funnel = funnel or comparison_adjusted_funnel(model, funnel_order)
    result.contribution = contribution
    result.funnel = funnel

    rows = characteristics or []
    rob = {str(row.get("study")): normalize_level(row.get("rob")) for row in rows if row.get("study")}
    indirect = {str(row.get("study")): normalize_level(row.get("indirectness")) for row in rows if row.get("study")}
    if not any(rob.values()):
        result.warnings.append("No risk-of-bias judgements entered (Study characteristics): studies without one "
                               "count as 'Some concerns'.")
    reporting = _reporting(funnel)
    overrides = overrides or {}

    for record in contribution.rows:
        effect = fit.league.get((record.treatment, record.versus))
        effect_re = nma.random.league.get((record.treatment, record.versus))
        if effect is None or effect_re is None:
            continue
        edge_studies = contribution.edge_studies.get((record.treatment, record.versus), [])
        row = CinemaRow(treatment=record.treatment, versus=record.versus, estimate=effect.estimate,
                        lower=effect.lower, upper=effect.upper, natural=effect.natural,
                        natural_lower=effect.natural_lower, natural_upper=effect.natural_upper,
                        natural_pred_lower=effect_re.natural_pred_lower, natural_pred_upper=effect_re.natural_pred_upper,
                        n_direct_studies=len(edge_studies), direct_share=record.direct_share)

        level, shares = _weighted_level(record.studies, rob, "Some concerns", result.rob_rule)
        row.domains["within_study_bias"] = Judgement(
            level, f"Contribution by risk of bias: {_share_text(shares)} ({result.rob_rule} rule).")
        row.domains["reporting_bias"] = Judgement(reporting.level, reporting.reason)
        level, shares = _weighted_level(record.studies, indirect, "Low", result.rob_rule)
        missing_note = " Studies without a rating are assumed Low." if shares.get("", 0) >= 0.005 else ""
        row.domains["indirectness"] = Judgement(
            level, f"Contribution by indirectness: {_share_text(shares)}.{missing_note}")
        row.domains["imprecision"] = _imprecision(effect.lower, effect.upper, margin, ratio)
        row.domains["heterogeneity"] = _heterogeneity((effect_re.lower, effect_re.upper),
                                                      (effect_re.pred_lower, effect_re.pred_upper),
                                                      margin, ratio, nma.heterogeneity.tau2)
        row.domains["incoherence"] = _incoherence(record.treatment, record.versus, inconsistency, effects)

        chosen = overrides.get(row.key, {})
        for domain, judgement in row.domains.items():
            judgement.auto_level = judgement.level
            allowed = REPORTING_LEVELS if domain == "reporting_bias" else CONCERN_LEVELS
            if chosen.get(domain) in allowed and chosen[domain] != judgement.level:
                judgement.level = chosen[domain]
                judgement.overridden = True
        row.auto_confidence = _confidence(row.downgrades)
        row.confidence = row.auto_confidence
        if chosen.get("confidence") in CONFIDENCE and chosen["confidence"] != row.auto_confidence:
            row.confidence = chosen["confidence"]
            row.confidence_overridden = True
        row.note = str(chosen.get("note", "") or "")
        result.rows.append(row)
    return result


def cinema_report_rows(result: CinemaResult) -> list[dict[str, Any]]:
    """Flat rows (for CSV export / tables)."""

    out = []
    for row in result.rows:
        record: dict[str, Any] = {
            "Comparison": row.label,
            "Estimate": row.natural, "Lower": row.natural_lower, "Upper": row.natural_upper,
            "Direct studies": row.n_direct_studies,
            "Direct evidence %": round(row.direct_share * 100, 1),
        }
        for domain in DOMAINS:
            record[DOMAIN_LABELS[domain]] = row.domains[domain].level
        record["Confidence"] = row.confidence
        record["Reasons"] = " | ".join(f"{DOMAIN_LABELS[d]}: {row.domains[d].reason}" for d in DOMAINS
                                       if DOWNGRADE.get(row.domains[d].level, 0))
        record["Note"] = row.note
        out.append(record)
    return out
