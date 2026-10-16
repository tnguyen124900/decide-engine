"""Budget Impact Analysis (BIA) -- ISPOR I/II good-practice structure.

A BIA compares total budget-holder spend across a multi-year time horizon
under two scenarios:

  - "current mix": the treatment mix as it exists today (the baseline,
    held constant unless the user varies it).
  - "new mix": the treatment mix after the new technology enters and
    takes up market share over time, whether by substitution (replacing
    an existing comparator), add-on (combined with existing therapy), or
    population expansion (previously untreated patients now treated) --
    the three mechanisms ISPOR's 2014 Good Practice II task force report
    (Sullivan et al.) names explicitly.

The per-year net budget impact is new-mix cost minus current-mix cost: a
positive number means the new technology increases the budget holder's
spend that year, a negative number means net savings. This mirrors the
generic skeleton of the ISPOR task force reports on budget impact analysis:
population by year ->
treatment-mix/market-share by year for each scenario -> per-patient cost
by category -> net budget impact per year and cumulative ->
sensitivity/scenario analysis.

This module is intentionally a pure-calculation layer (plain dicts in,
dataclasses out) with no Streamlit dependency, mirroring engine.py/
markov_engine.py/microsim_engine.py so it can be unit-tested directly.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Mapping

# The five ISPOR I/II-recommended cost categories a comparator's
# per-patient annual cost is broken into. Every comparator need only set
# the categories that apply to it -- the rest default to 0.
COST_CATEGORIES = (
    "cost_acquisition",
    "cost_administration",
    "cost_monitoring",
    "cost_adverse_events",
    "cost_disease_management",
)


@dataclass
class BIAComparator:
    """One treatment option competing for market share -- either an
    existing comparator in the current mix, or the new technology itself
    (``is_new=True``)."""

    id: str
    label: str
    is_new: bool = False
    cost_acquisition: float = 0.0
    cost_administration: float = 0.0
    cost_monitoring: float = 0.0
    cost_adverse_events: float = 0.0
    cost_disease_management: float = 0.0
    # Cost of one patient in each year since starting this treatment (treatment year 1, 2, ...), typically imported from a
    # DECIDE model (see bia_model_link.py). Only used when the document's cost_basis is "treatment_year".
    cost_by_treatment_year: list[float] | None = None

    def cost_per_patient(self) -> float:
        return sum(getattr(self, category) for category in COST_CATEGORIES)

    def cost_in_treatment_year(self, k: int) -> float:
        """Cost of one patient k years (0-based) after starting: the model profile if there is one (0 after its end),
        otherwise the flat annual cost per patient."""

        if self.cost_by_treatment_year:
            return float(self.cost_by_treatment_year[k]) if k < len(self.cost_by_treatment_year) else 0.0
        return self.cost_per_patient()


@dataclass
class BIAYearResult:
    """One time horizon year's worth of results."""

    year: int
    population: float
    current_mix_cost: float
    new_mix_cost: float
    budget_impact: float
    discounted_budget_impact: float
    cumulative_budget_impact: float
    cumulative_discounted_budget_impact: float
    current_mix_coverage: float
    new_mix_coverage: float


@dataclass
class BIAResult:
    years: list[BIAYearResult] = field(default_factory=list)
    total_budget_impact: float = 0.0
    total_discounted_budget_impact: float = 0.0


def _profile(raw: Any) -> list[float] | None:
    if not raw:
        return None
    try:
        return [float(v or 0.0) for v in raw]
    except (TypeError, ValueError):
        return None


def _parse_comparators(raw: list[Mapping[str, Any]]) -> list[BIAComparator]:
    comparators: list[BIAComparator] = []
    for row in raw:
        label = str(row.get("label", "") or "").strip()
        comparator_id = str(row.get("id") or label)
        comparators.append(
            BIAComparator(
                id=comparator_id,
                label=label,
                is_new=bool(row.get("is_new", False)),
                cost_acquisition=float(row.get("cost_acquisition", 0.0) or 0.0),
                cost_administration=float(row.get("cost_administration", 0.0) or 0.0),
                cost_monitoring=float(row.get("cost_monitoring", 0.0) or 0.0),
                cost_adverse_events=float(row.get("cost_adverse_events", 0.0) or 0.0),
                cost_disease_management=float(row.get("cost_disease_management", 0.0) or 0.0),
                cost_by_treatment_year=_profile(row.get("cost_by_treatment_year")),
            )
        )
    return comparators


def _share_at(share_map: Mapping[str, Any], comparator_id: str, year_index: int) -> float:
    values = share_map.get(comparator_id) or []
    if year_index >= len(values):
        return 0.0
    try:
        return float(values[year_index])
    except (TypeError, ValueError):
        return 0.0


def calculate_budget_impact(document: Mapping[str, Any]) -> tuple[BIAResult, list[str]]:
    """Compute year-by-year net budget impact from a BIA document (a plain
    dict, as stored in Streamlit session state -- see DEFAULT_BIA_DOCUMENT
    in app.py for the exact shape). Returns (result, errors); errors is
    non-empty only for structural problems (missing horizon, mismatched
    population length, no comparators) that make the calculation
    impossible to run at all -- market-share coverage that doesn't sum to
    100% is not treated as a hard error here (a partially-treated eligible
    population, i.e. some patients remaining untreated, is a legitimate
    real-world state), so callers should surface each year's coverage
    fields to the user as an informational check instead.
    """

    errors: list[str] = []
    horizon = int(document.get("time_horizon_years", 0) or 0)
    if horizon <= 0:
        errors.append("Time horizon must be at least 1 year")
        return BIAResult(), errors

    population = list(document.get("eligible_population") or [])
    if len(population) != horizon:
        errors.append(
            f"Eligible population must have exactly {horizon} year(s) of values (found {len(population)})"
        )
        return BIAResult(), errors

    comparators = _parse_comparators(document.get("comparators") or [])
    if not comparators:
        errors.append("Add at least one comparator (including the new technology)")
        return BIAResult(), errors

    discount_rate = float(document.get("discount_rate", 0.0) or 0.0)
    current_share = document.get("current_mix_share") or {}
    new_share = document.get("new_mix_share") or {}
    cost_by_id = {c.id: c.cost_per_patient() for c in comparators}

    cohort_mode = document.get("cost_basis") == "treatment_year"
    years: list[BIAYearResult] = []
    cumulative = 0.0
    cumulative_discounted = 0.0
    for year_index in range(horizon):
        pop = float(population[year_index] or 0.0)
        current_cost = 0.0
        new_cost = 0.0
        current_coverage = 0.0
        new_coverage = 0.0
        for comparator in comparators:
            current_coverage += _share_at(current_share, comparator.id, year_index)
            new_coverage += _share_at(new_share, comparator.id, year_index)
        if cohort_mode:
            # population[s] = patients who START treatment in year s; each cohort keeps the share it started with and costs
            # cost_in_treatment_year(y - s) in year y (cohorts overlap).
            for start in range(year_index + 1):
                start_pop = float(population[start] or 0.0)
                age = year_index - start
                for comparator in comparators:
                    unit = comparator.cost_in_treatment_year(age)
                    current_cost += start_pop * _share_at(current_share, comparator.id, start) * unit
                    new_cost += start_pop * _share_at(new_share, comparator.id, start) * unit
        else:
            for comparator in comparators:
                current_cost += pop * _share_at(current_share, comparator.id, year_index) * cost_by_id[comparator.id]
                new_cost += pop * _share_at(new_share, comparator.id, year_index) * cost_by_id[comparator.id]
        impact = new_cost - current_cost
        discount_factor = 1.0 / ((1.0 + discount_rate) ** year_index)
        discounted_impact = impact * discount_factor
        cumulative += impact
        cumulative_discounted += discounted_impact
        years.append(
            BIAYearResult(
                year=year_index + 1,
                population=pop,
                current_mix_cost=current_cost,
                new_mix_cost=new_cost,
                budget_impact=impact,
                discounted_budget_impact=discounted_impact,
                cumulative_budget_impact=cumulative,
                cumulative_discounted_budget_impact=cumulative_discounted,
                current_mix_coverage=current_coverage,
                new_mix_coverage=new_coverage,
            )
        )

    return (
        BIAResult(
            years=years,
            total_budget_impact=cumulative,
            total_discounted_budget_impact=cumulative_discounted,
        ),
        errors,
    )


def epidemiological_population(
    total_population: float,
    prevalence_per_100k: float,
    eligible_fraction: float,
    horizon: int,
    annual_growth_rate: float = 0.0,
) -> list[float]:
    """Estimate eligible population per year from epidemiological inputs --
    ISPOR I/II's preferred population-estimation method when the decision
    maker's own eligible-population figures aren't directly available:
    prevalent cases = total population x prevalence rate, of which only
    some fraction is eligible for the new technology (e.g. by severity,
    line of therapy, or diagnostic criteria). ``annual_growth_rate`` lets
    the prevalent pool expand or contract year over year (e.g. rising
    incidence, an aging population); year 1 (index 0) is never grown, so
    the base-year estimate is used as-is.
    """

    base = total_population * (prevalence_per_100k / 100_000.0) * eligible_fraction
    return [base * ((1.0 + annual_growth_rate) ** year_index) for year_index in range(horizon)]


# Unit -> divisor applied to a population estimate's raw value before it is
# multiplied into the cascade below, so a row entered as "per 100,000" or
# "%" behaves the same way total_population/prevalence_per_100k/
# eligible_fraction did in the old fixed-field formula above, without the
# user having to pre-divide it themselves. "count", "fraction" and
# "multiplier" all mean "use the number as typed" -- they exist as distinct
# options only so the row's label makes its role in the cascade clear.
POPULATION_ESTIMATE_UNITS = ("count", "fraction", "percent", "per_100k", "per_1000", "multiplier", "other")

_POPULATION_ESTIMATE_UNIT_DIVISORS: dict[str, float] = {
    "count": 1.0,
    "fraction": 1.0,
    "percent": 100.0,
    "per_100k": 100_000.0,
    "per_1000": 1_000.0,
    "multiplier": 1.0,
    "other": 1.0,
}


def population_from_estimates(
    estimates: list[Mapping[str, Any]],
    horizon: int,
) -> list[float]:
    """Estimate eligible population per year from a free-form list of named,
    unit-tagged rows (the "creative/customizable Population" builder) --
    each row's ``value`` is converted to a plain multiplier by its
    ``unit`` (e.g. a "per_100k" row is divided by 100,000, a "percent" row
    by 100), can compound at its own ``growth_rate`` year over year, and
    every row's per-year factor is multiplied together to get that year's
    eligible population. This generalizes ``epidemiological_population``
    above (total population x prevalence x eligible fraction) to any
    number of factors, each with its own name, unit and growth rate,
    instead of exactly those three fixed inputs.

    An empty ``estimates`` list returns all zeros rather than erroring, so
    an in-progress edit (e.g. the last row just deleted) never crashes the
    page -- ``calculate_budget_impact`` already treats an all-zero
    eligible_population as a normal (if uninteresting) result.
    """

    if not estimates or horizon <= 0:
        return [0.0] * max(horizon, 0)

    population: list[float] = []
    for year_index in range(horizon):
        year_value = 1.0
        for estimate in estimates:
            raw_value = float(estimate.get("value", 0.0) or 0.0)
            unit = estimate.get("unit") or "count"
            divisor = _POPULATION_ESTIMATE_UNIT_DIVISORS.get(unit, 1.0)
            growth_rate = float(estimate.get("growth_rate", 0.0) or 0.0)
            year_value *= (raw_value / divisor) * ((1.0 + growth_rate) ** year_index)
        population.append(year_value)
    return population


def scenario_result(
    document: Mapping[str, Any],
    new_tech_cost_multiplier: float = 1.0,
) -> tuple[BIAResult, list[str]]:
    """Recompute the budget impact after scaling the new technology's
    per-patient cost by a multiplier -- the ISPOR-recommended optimistic /
    most-likely / pessimistic scenario approach, without requiring a full
    PSA distribution setup. Only comparators flagged ``is_new`` are
    scaled; existing comparators and market-share assumptions are left
    untouched.
    """

    scaled_document = copy.deepcopy(dict(document))
    scaled_comparators = []
    for row in scaled_document.get("comparators") or []:
        row = dict(row)
        if row.get("is_new"):
            for category in COST_CATEGORIES:
                if row.get(category) is not None:
                    row[category] = float(row[category]) * new_tech_cost_multiplier
            if row.get("cost_by_treatment_year"):
                row["cost_by_treatment_year"] = [float(v or 0.0) * new_tech_cost_multiplier for v in row["cost_by_treatment_year"]]
        scaled_comparators.append(row)
    scaled_document["comparators"] = scaled_comparators
    return calculate_budget_impact(scaled_document)
