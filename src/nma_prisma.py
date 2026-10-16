"""PRISMA-NMA reporting checklist (Hutton et al., Ann Intern Med
2015;162:777-84): the 27 PRISMA items plus the five NMA-specific items
(S1-S5). The NMA Editor fills in the items that the report itself covers
(network geometry, summary measures, statistical methods, inconsistency,
results tables) and leaves the manuscript items (title, search, study
selection, funding, ...) for the author, who records where each item is
reported. Author entries live in document["prisma"][item_id] =
{"status": ..., "note": ...}."""

from __future__ import annotations

from typing import Any

import pandas as pd

STATUSES = ("Reported", "Partly", "To do", "Not applicable")
AUTO = "auto"  # pseudo-status: keep the editor's suggestion

# (id, section, topic, checklist item, NMA-specific label or "")
ITEMS: list[tuple[str, str, str, str, str]] = [
    ("1", "Title", "Title", "Identify the report as a systematic review incorporating a network meta-analysis (or related form of meta-analysis).", ""),
    ("2", "Abstract", "Structured summary", "Structured summary: background, objectives, data sources, eligibility criteria, participants, interventions, study appraisal and synthesis methods (incl. the network meta-analysis method and assumptions), results (number of studies and participants, network summary, effects with credible/confidence intervals, treatment rankings, inconsistency), limitations, conclusions, funding, registration.", ""),
    ("3", "Introduction", "Rationale", "Describe the rationale for the review in the context of what is already known, including why a network meta-analysis is needed.", ""),
    ("4", "Introduction", "Objectives", "Explicit statement of the questions addressed (PICOS).", ""),
    ("5", "Methods", "Protocol and registration", "Indicate whether a protocol exists, where it can be accessed, and the registration number.", ""),
    ("6", "Methods", "Eligibility criteria", "Study characteristics (PICOS, follow-up) and report characteristics used as criteria for eligibility; clarify the eligibility of treatments and any treatment nodes that were merged.", ""),
    ("7", "Methods", "Information sources", "All information sources (databases with dates of coverage, contact with authors, ...) and the date last searched.", ""),
    ("8", "Methods", "Search", "Full electronic search strategy for at least one database, such that it could be repeated.", ""),
    ("9", "Methods", "Study selection", "Process for selecting studies (screening, eligibility, inclusion in the review and in the network meta-analysis).", ""),
    ("10", "Methods", "Data collection process", "Method of data extraction (piloted forms, independent extraction, duplicate) and process for obtaining/confirming data from investigators.", ""),
    ("11", "Methods", "Data items", "All variables sought (PICOS, funding sources) and any assumptions and simplifications made.", ""),
    ("S1", "Methods", "Geometry of the network", "Describe methods used to explore the geometry of the treatment network and potential biases related to it (how many studies per link, choice of comparators, ...).", "S1"),
    ("12", "Methods", "Risk of bias within individual studies", "Methods for assessing risk of bias of individual studies (study or outcome level) and how this information was used in the synthesis.", ""),
    ("13", "Methods", "Summary measures", "Principal summary measures (risk ratio, mean difference, ...); describe additional summary measures for the network meta-analysis, such as treatment rankings and SUCRA / P-scores, and modified approaches used to present summary findings.", ""),
    ("14", "Methods", "Planned methods of analysis", "Methods of handling data and combining results, including the network meta-analysis model (fixed / random effects, Bayesian or frequentist), handling of multi-arm trials, and measures of heterogeneity (τ², I²).", ""),
    ("S2", "Methods", "Assessment of inconsistency", "Describe the statistical methods used to evaluate the agreement of direct and indirect evidence in the network (node-splitting, design-by-treatment interaction, UME model, ...).", "S2"),
    ("15", "Methods", "Risk of bias across studies", "Any assessment of risk of bias that may affect the cumulative evidence (publication bias, selective reporting, comparison-adjusted funnel plot).", ""),
    ("16", "Methods", "Additional analyses", "Methods of additional analyses (sensitivity, subgroup, meta-regression, alternative model formulations), stating which were pre-specified.", ""),
    ("17", "Results", "Study selection", "Numbers of studies screened, assessed for eligibility and included, with reasons for exclusions (flow diagram).", ""),
    ("S3", "Results", "Presentation of network structure", "Provide a network graph of the included studies.", "S3"),
    ("S4", "Results", "Summary of network geometry", "Provide a brief overview of the network's characteristics (number of studies and patients per comparison, multi-arm trials, closed loops).", "S4"),
    ("18", "Results", "Study characteristics", "For each study, present the characteristics for which data were extracted (study size, PICOS, follow-up) and provide the citations.", ""),
    ("19", "Results", "Risk of bias within studies", "Data on risk of bias of each study and, if available, any outcome-level assessment.", ""),
    ("20", "Results", "Results of individual studies", "For all outcomes, for each study: simple summary data for each intervention group and effect estimates with confidence intervals (forest plot or table).", ""),
    ("21", "Results", "Synthesis of results", "Results of each network meta-analysis: summary effects for each pairwise comparison with credible / confidence intervals (league table), and treatment rankings where reported; heterogeneity.", ""),
    ("S5", "Results", "Exploration for inconsistency", "Describe results from investigations of inconsistency (node-splitting, design-by-treatment test, UME comparison), including numerical and graphical results.", "S5"),
    ("22", "Results", "Risk of bias across studies", "Results of any assessment of risk of bias across studies (funnel plot, Egger's test).", ""),
    ("23", "Results", "Results of additional analyses", "Results of additional analyses (sensitivity, subgroup, meta-regression, alternative models).", ""),
    ("24", "Discussion", "Summary of evidence", "Summarise the main findings including the strength / confidence of the evidence for each main outcome, relevance to key groups.", ""),
    ("25", "Discussion", "Limitations", "Limitations at study and outcome level (risk of bias) and at review level (incomplete retrieval, reporting bias); comment on the validity of the assumptions (transitivity, consistency).", ""),
    ("26", "Discussion", "Conclusions", "General interpretation of the results in the context of other evidence, and implications for future research.", ""),
    ("27", "Funding", "Funding", "Sources of funding for the review and other support; role of funders.", ""),
]

MANUSCRIPT = "Manuscript"  # items the app cannot fill: the author reports where


def auto_status(*, has_characteristics: bool, has_rob: bool, has_inconsistency: bool, has_cinema: bool,
                has_funnel: bool, bayes: bool, has_absolute: bool, extra_analyses: list[str] | None = None
                ) -> dict[str, tuple[str, str]]:
    """Editor suggestion per item: (status, where reported)."""

    ranking = "SUCRA" if bayes else "P-score"
    incons = ("Reported", "Report: Inconsistency (design-by-treatment test, node-splitting" +
              (", UME)" if bayes else ")")) if has_inconsistency else ("To do", "Press Inconsistency in the NMA Editor")
    funnel = ("Reported", "Report: CINeMA reporting-bias domain (comparison-adjusted funnel plot, Egger's test)") \
        if has_cinema or has_funnel else ("To do", "Run CINeMA / Funnel plot in Additional analyses")
    extras = ", ".join(extra_analyses or [])
    additional = ("Partly", f"Available in the NMA Editor: {extras}" if extras else
                  "Leave-one-out, prior sensitivity, meta-regression available in the NMA Editor")
    out = {
        "S1": ("Reported", "Report: Methods, Network geometry"),
        "13": ("Reported", f"Report: Methods (summary measure, {ranking} ranking); Results: Relative effects, Ranking"),
        "14": ("Reported", "Report: Methods (model, multi-arm handling, τ²); Results: Heterogeneity"),
        "S2": ("Reported" if has_inconsistency else "To do", "Report: Methods (inconsistency)" if has_inconsistency
               else "Press Inconsistency in the NMA Editor"),
        "15": funnel,
        "16": additional,
        "S3": ("Reported", "Report: Figure 1 (network graph)"),
        "S4": ("Reported", "Report: Table 1 (direct comparisons: studies, patients)"),
        "18": ("Reported" if has_characteristics else "Partly",
               "Report: Table 2 (study characteristics)" if has_characteristics
               else "Enter Study characteristics in the NMA Editor; citations in the Source column"),
        "19": ("Reported" if has_rob else "To do",
               "Report: CINeMA within-study bias; Excel: Study characteristics (RoB)" if has_rob
               else "Enter risk of bias per study (Study characteristics)"),
        "20": ("Reported", "Excel: Data arms / Data contrasts sheets"),
        "21": ("Reported", "Report: Table 3 (effects), Table 4 (league table), Table 5 (ranking), Heterogeneity" +
               (", Absolute effects" if has_absolute else "")),
        "S5": incons,
        "22": funnel,
        "23": additional,
        "12": ("Partly", "Report: CINeMA within-study bias (contribution-weighted RoB)") if has_cinema
              else ("To do", MANUSCRIPT),
        "24": ("Partly", "Report: CINeMA confidence per comparison") if has_cinema else ("To do", MANUSCRIPT),
    }
    for item_id, *_ in ITEMS:
        out.setdefault(item_id, ("To do", MANUSCRIPT))
    return out


def checklist_frame(document: dict[str, Any], auto: dict[str, tuple[str, str]]) -> pd.DataFrame:
    """Rows for the table: editor suggestion merged with the author's
    entries (`document["prisma"]`)."""

    entries = document.get("prisma") or {}
    rows = []
    for item_id, section, topic, text, tag in ITEMS:
        status, where = auto.get(item_id, ("To do", MANUSCRIPT))
        entry = entries.get(item_id) or {}
        chosen = entry.get("status")
        if chosen in STATUSES:
            status = chosen
        note = str(entry.get("note", "") or "")
        rows.append({"Item": item_id, "Section": section, "Topic": topic, "Checklist item": text,
                     "Status": status, "Suggested": where, "Where reported / notes": note})
    return pd.DataFrame(rows)


def apply_edits(document: dict[str, Any], edited: pd.DataFrame, auto: dict[str, tuple[str, str]]) -> bool:
    """Store the author's status / note changes; returns True if anything changed."""

    entries = document.setdefault("prisma", {})
    changed = False
    for _, row in edited.iterrows():
        item_id = str(row["Item"])
        status = str(row.get("Status", "") or "")
        note = str(row.get("Where reported / notes", "") or "").strip()
        new = {}
        if status in STATUSES and status != auto.get(item_id, ("", ""))[0]:
            new["status"] = status
        if note:
            new["note"] = note
        if new != (entries.get(item_id) or {}):
            changed = True
            if new:
                entries[item_id] = new
            else:
                entries.pop(item_id, None)
    return changed


def summary(frame: pd.DataFrame) -> dict[str, int]:
    return {status: int((frame["Status"] == status).sum()) for status in STATUSES}


def report_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Compact version for the Word / Excel report."""

    out = frame.copy()
    out["Where reported"] = [note if note else where for note, where in
                             zip(out["Where reported / notes"], out["Suggested"])]
    return out[["Item", "Section", "Topic", "Checklist item", "Status", "Where reported"]]
