"""GRADE-style Summary of Findings (SoF) table for a network meta-analysis
(Yepes-Nuñez et al. 2019, BMJ 365:l1543; Puhan et al. 2014).

One row per comparison: number of direct studies and participants, relative
effect with its CI / CrI, assumed (comparator) risk, corresponding risk and
absolute difference per 1000 with NNT (binary outcomes), certainty of the
evidence from CINeMA with the domains that led to downgrading, and the
ranking of the treatment (P-score / SUCRA).

Three layouts: "reference" (every treatment vs the reference), "all"
(every pair, later treatment vs earlier in the treatment order) and
"comparator" (every treatment vs one chosen comparator).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pandas as pd

from heor_tree_engine.nma_absolute import ABSOLUTE_MEASURES, absolute_effects
from heor_tree_engine.nma_cinema import DOMAINS, DOMAIN_LABELS, DOWNGRADE
from heor_tree_engine.nma_models import is_ratio_measure

LAYOUTS = ("reference", "all", "comparator")
COLUMNS = ["Comparison", "Studies (direct)", "Participants (direct)", "Relative effect", "Assumed risk /1000",
           "Corresponding risk /1000", "Difference /1000", "NNT", "Certainty (CINeMA)", "Downgraded for", "Ranking"]


def observed_risk(model, treatment: str) -> float | None:
    """Pooled event risk in the arms of `treatment` (binary arm-level data)."""

    if model.spec.outcome != "binary":
        return None
    events = total = 0.0
    for study in model.studies:
        for arm in study.arms:
            if arm.treatment == treatment and arm.events is not None and arm.n:
                events += arm.events
                total += arm.n
    return events / total if total > 0 else None


def comparator_risks(model, document: dict[str, Any]) -> dict[str, float]:
    """Assumed risk per comparator: the user's baseline risk for the
    reference (document["absolute"]["baseline_risk"]), the observed risk for
    the others (falling back to the reference risk)."""

    risks: dict[str, float] = {}
    user = (document.get("absolute") or {}).get("baseline_risk")
    ref_risk = user or observed_risk(model, model.reference) or 0.10
    for name in model.treatment_names:
        if name == model.reference:
            risks[name] = float(ref_risk)
        else:
            risks[name] = float(observed_risk(model, name) or ref_risk)
    return risks


def pairs_for(fit, model, layout: str, comparator: str | None = None) -> list[tuple[str, str]]:
    names = [n for n in model.treatment_names if any(n in key for key in fit.league)] or list(model.treatment_names)
    if layout == "all":
        return [(later, earlier) for j, later in enumerate(names) for earlier in names[:j] if (later, earlier) in fit.league]
    base = comparator if layout == "comparator" and comparator else model.reference
    return [(name, base) for name in names if name != base and (name, base) in fit.league]


def _certainty(cinema, treatment: str, versus: str) -> tuple[str, str]:
    if cinema is None or getattr(cinema, "errors", None):
        return "", ""
    for row in cinema.rows:
        if {row.treatment, row.versus} == {treatment, versus}:
            reasons = [DOMAIN_LABELS[d] for d in DOMAINS if DOWNGRADE.get(row.domains[d].level, 0)]
            return row.confidence, ", ".join(reasons) or "–"
    return "", ""


def sof_frame(document: dict[str, Any], model, summary, fit, measure: str, layout: str = "reference",
              comparator: str | None = None, cinema=None, ranking_label: str = "P-score",
              interval_label: str = "95% CI", per: float = 1000.0) -> pd.DataFrame:
    if layout not in LAYOUTS:
        raise ValueError(f"Unknown layout '{layout}'")
    ratio = is_ratio_measure(measure)
    absolute_ok = measure in ABSOLUTE_MEASURES
    risks = comparator_risks(model, document) if absolute_ok else {}
    small_values = model.spec.small_values if hasattr(model.spec, "small_values") else "good"
    rows = []
    for treatment, versus in pairs_for(fit, model, layout, comparator):
        effect = fit.league[(treatment, versus)]
        edge = summary.edges.get((min(treatment, versus), max(treatment, versus)))
        digits = 2
        relative = f"{effect.natural:.{digits}f} ({effect.natural_lower:.{digits}f} to {effect.natural_upper:.{digits}f})"
        record: dict[str, Any] = {
            "Comparison": f"{treatment} vs {versus}",
            "Studies (direct)": edge.n_studies if edge else 0,
            "Participants (direct)": int(round(edge.patients)) if edge and edge.patients else ("" if edge else 0),
            "Relative effect": f"{measure} {relative}",
        }
        if absolute_ok:
            p0 = risks.get(versus, 0.10)
            abs_row = absolute_effects(SimpleNamespace(effects=[effect]), measure, p0, small_values)[0]
            record.update({
                "Assumed risk /1000": f"{abs_row.baseline_risk * per:.0f} ({versus})",
                "Corresponding risk /1000": f"{abs_row.risk * per:.0f} ({abs_row.risk_lower * per:.0f} to {abs_row.risk_upper * per:.0f})",
                "Difference /1000": f"{abs_row.difference * per:+.0f} ({abs_row.difference_lower * per:+.0f} to {abs_row.difference_upper * per:+.0f})",
                "NNT": abs_row.nnt_text,
            })
        else:
            record.update({"Assumed risk /1000": "–", "Corresponding risk /1000": "–", "Difference /1000": "–", "NNT": "–"})
        certainty, reasons = _certainty(cinema, treatment, versus)
        record["Certainty (CINeMA)"] = certainty or "not rated"
        record["Downgraded for"] = reasons
        score = fit.p_scores.get(treatment)
        record["Ranking"] = f"{ranking_label} {score:.2f}" if score is not None else ""
        rows.append(record)
    frame = pd.DataFrame(rows, columns=COLUMNS)
    if not absolute_ok:
        frame = frame.drop(columns=["Assumed risk /1000", "Corresponding risk /1000", "Difference /1000", "NNT"])
    frame.attrs["interval"] = interval_label
    frame.attrs["ratio"] = ratio
    return frame


def footnotes(measure: str, interval_label: str, bayes: bool, cinema_present: bool) -> list[str]:
    notes = [f"Relative effects with {interval_label}; " +
             ("posterior medians (Bayesian model)." if bayes else "network estimates (frequentist model).")]
    if measure in ABSOLUTE_MEASURES:
        notes.append("Assumed risk: the baseline risk entered for the reference, or the pooled observed risk of the "
                     "comparator's arms; corresponding risk and difference per 1000 are derived from the relative "
                     "effect; NNTB / NNTH with the CI (Altman method).")
    notes.append("Certainty of the evidence: CINeMA rating (High / Moderate / Low / Very low), with the domains that "
                 "led to downgrading." if cinema_present else "Certainty: run CINeMA (Additional analyses) to rate the evidence.")
    notes.append("Ranking: " + ("SUCRA" if bayes else "P-score") + " of the treatment (1 = certainly best).")
    return notes
