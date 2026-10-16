"""Link a BIA comparator to a DECIDE model: its cost per patient by year of treatment comes from the model.

The oncology-style BIA: a model (Markov, partitioned survival driven by PFS/OS, decision tree) gives the undiscounted cost of ONE
patient over time; the BIA multiplies it by the number of patients starting treatment each year.  A comparator keeps

    comparator["cost_by_treatment_year"] = [year-1 cost, year-2 cost, ...]   # per patient, undiscounted
    comparator["model_link"] = {"source": "markov"|"psm"|"tree", "key": ..., "label": ...}

and the document switches to ``document["cost_basis"] = "treatment_year"`` (see bia.calculate_budget_impact: each year's starting cohort
keeps its market share and follows the profile, cohorts overlap).  The profile is a snapshot: "refresh" re-reads the model.
Pure Python, no Streamlit.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Sequence

SOURCES = {"markov": "Markov model", "psm": "Partitioned survival (PFS/OS) model", "tree": "Decision tree"}
COST_BASES = {"annual": "Annual cost per patient (patients in treatment each year)",
              "treatment_year": "Cost by year of treatment (patients start each year, cohorts overlap)"}


def yearly_from_cycles(cost_per_cycle: Sequence[float], cycles_per_year: float = 1.0) -> list[float]:
    """Sum per-cycle costs into treatment years: cycle c belongs to year floor(c / cycles_per_year) + 1."""

    cpy = float(cycles_per_year) if cycles_per_year and cycles_per_year > 0 else 1.0
    years: list[float] = []
    for cycle, cost in enumerate(cost_per_cycle):
        index = int(cycle / cpy + 1e-9)
        while len(years) <= index:
            years.append(0.0)
        years[index] += float(cost or 0.0)
    return years


def profile_from_markov(result: Any, cycles_per_year: float) -> list[float]:
    """Undiscounted cost by treatment year from a MarkovResult (raw_cost_per_cycle; no half-cycle correction)."""

    return yearly_from_cycles(list(getattr(result, "raw_cost_per_cycle", None) or []), cycles_per_year)


def profile_from_psm_rows(rows: Iterable[dict[str, Any]], cycles_per_year: float) -> list[float]:
    """Same for the rows of a PSM trace; the end-of-horizon endpoint row (no reward) is ignored."""

    costs = [r.get("cost_undiscounted") for r in rows if r.get("cost_undiscounted") is not None]
    return yearly_from_cycles(costs, cycles_per_year)


def profile_from_tree(expected_cost: float) -> list[float]:
    """A decision tree is a single episode: its expected cost falls in treatment year 1."""

    return [float(expected_cost or 0.0)]


Provider = Callable[[str], "tuple[list[float] | None, str | None]"]


def apply_link(comparator: dict[str, Any], source: str, key: str, label: str, profile: list[float]) -> None:
    comparator["cost_by_treatment_year"] = [float(v) for v in profile]
    comparator["model_link"] = {"source": source, "key": key, "label": label}


def clear_link(comparator: dict[str, Any]) -> None:
    comparator.pop("cost_by_treatment_year", None)
    comparator.pop("model_link", None)


def refresh_links(document: dict[str, Any], providers: dict[str, Provider]) -> list[str]:
    """Re-read every linked comparator from its model; returns messages for the ones that could not be refreshed."""

    problems: list[str] = []
    for comparator in document.get("comparators") or []:
        link = comparator.get("model_link")
        if not link:
            continue
        provider = providers.get(link.get("source"))
        if provider is None:
            problems.append(f"{comparator.get('label')}: no {link.get('source')} model available")
            continue
        profile, error = provider(link.get("key", ""))
        if profile is None:
            problems.append(f"{comparator.get('label')}: {error or 'model could not be run'}")
            continue
        comparator["cost_by_treatment_year"] = [float(v) for v in profile]
    return problems


def profile_frame_rows(document: dict[str, Any]) -> list[dict[str, Any]]:
    """[{'Comparator', 'Source', 'Year 1', ...}] for the linked comparators (for display)."""

    rows = []
    longest = max((len(c.get("cost_by_treatment_year") or []) for c in document.get("comparators") or []), default=0)
    for comparator in document.get("comparators") or []:
        profile = comparator.get("cost_by_treatment_year")
        link = comparator.get("model_link") or {}
        if not profile:
            continue
        row: dict[str, Any] = {"Comparator": comparator.get("label"), "Source": f"{SOURCES.get(link.get('source'), 'manual')}: {link.get('label', '')}".strip(": ")}
        for i in range(longest):
            row[f"Year {i + 1}"] = f"{profile[i]:,.0f}" if i < len(profile) else ""
        row["Total per patient"] = f"{sum(profile):,.0f}"
        rows.append(row)
    return rows
