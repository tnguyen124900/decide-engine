"""Deterministic sensitivity-analysis helpers for Markov documents."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import numpy as np

from .engine import Engine
from .expressions import build_tables_mapping, evaluate_expression, resolve_parameter_definitions
from .markov_engine import run_markov
from .markov_parser import resolve_markov
from .microsim_engine import _resample_distributions, run_microsim
from .microsim_parser import resolve_microsim
from .models import Param
from .parser import resolve as resolve_tree


@dataclass(frozen=True)
class StrategyOutcome:
    """One strategy's outcome under a deterministic scenario."""

    strategy_id: str
    strategy_name: str
    total_cost: float
    total_effect: float
    nmb: float
    incremental_cost: float
    incremental_effect: float
    incremental_nmb: float
    icer: float | None
    # Microsimulation-only: the within-trial patient-level sample
    # variance/covariance of this strategy's total cost/effect, and the
    # patient count they were computed from -- None for Markov/Tree
    # scenarios, which have no patient-level sampling. Lets a PSA run
    # decompose each strategy's between-trial NMB variance into
    # parameter uncertainty vs. finite-patient-sample Monte Carlo noise
    # (see psa.decompose_psa_noise).
    patient_cost_variance: float | None = None
    patient_effect_variance: float | None = None
    patient_cost_effect_covariance: float | None = None
    n_patients: int | None = None
    # Engine-specific extra metrics an SA may target instead of NMB/ICER
    # (see _outcome_value): reserved for model-specific metrics; None otherwise.
    extras: Mapping[str, float | None] | None = None


@dataclass
class SensitivityResult:
    """A complete Markov scenario evaluation, including scenario-local errors."""

    reference_strategy_id: str | None = None
    willingness_to_pay: float = 50_000.0
    outcomes: list[StrategyOutcome] = field(default_factory=list)
    optimal_strategy_id: str | None = None
    errors: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class SensitivityInput:
    """A parameter or variable that can be varied in deterministic SA."""

    name: str
    kind: str
    base: float
    low: float | None
    high: float | None


@dataclass(frozen=True)
class TornadoEntry:
    """The low/high outcome range for one input in a Tornado analysis."""

    input_name: str
    base: float
    low: float
    high: float
    low_outcome: float
    high_outcome: float

    @property
    def range_width(self) -> float:
        return abs(self.high_outcome - self.low_outcome)


def _per_cycle_rate(value: float, unit: str, cycles_per_year: float) -> float:
    if unit == "per cycle" or cycles_per_year <= 0:
        return value
    return (1.0 + value) ** (1.0 / cycles_per_year) - 1.0


def _apply_overrides(document: dict[str, Any], overrides: Mapping[str, float]) -> list[str]:
    parameters = {str(item.get("name")): item for item in document.get("parameters", []) if item.get("name")}
    variables = {str(item.get("name")): item for item in document.get("variables", []) if item.get("name")}
    errors: list[str] = []
    for name, value in overrides.items():
        parameter = parameters.get(name)
        variable = variables.get(name)
        if parameter is not None and variable is not None:
            errors.append(f"Override '{name}' is ambiguous between a parameter and variable")
        elif parameter is not None:
            parameter["base"] = float(value)
            parameter.pop("definition", None)
        elif variable is not None:
            variable["definition"] = float(value)
            variable.pop("base", None)
            variable.pop("value", None)
        else:
            errors.append(f"Override '{name}' does not match a parameter or variable")
    return errors


def _build_inputs(
    document: Mapping[str, Any],
) -> tuple[dict[str, Param], dict[str, Any], dict[str, Any], dict[str, Any]]:
    distributions = {
        str(raw["name"]): raw for raw in document.get("distributions", []) if raw.get("name")
    }
    tables = build_tables_mapping(document.get("tables"))
    parameters: dict[str, Param] = {}
    definitions: dict[str, Any] = {}
    for raw in document.get("parameters", []):
        name = str(raw["name"])
        try:
            base = float(raw.get("base", 0.0))
        except (TypeError, ValueError):
            base = 0.0
        parameters[name] = Param(
            name=name,
            base=base,
            low=float(raw["low"]) if raw.get("low") is not None else None,
            high=float(raw["high"]) if raw.get("high") is not None else None,
        )
        if raw.get("definition") is not None:
            definitions[name] = raw["definition"]
    parameters = resolve_parameter_definitions(parameters, definitions, distributions, tables)
    variables = {
        str(raw["name"]): raw.get("definition", raw.get("base", raw.get("value", "")))
        for raw in document.get("variables", [])
        if raw.get("name")
    }
    return parameters, variables, distributions, tables


def _merged_document(document: Mapping[str, Any], strategy: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(document)
    merged["id"] = strategy.get("id", "markov")
    merged["name"] = strategy.get("name") or strategy.get("id", "Markov model")
    merged["states"] = strategy.get("states", [])
    merged["transitions"] = strategy.get("transitions", [])
    return merged


def _microsim_merged_document(document: Mapping[str, Any], strategy: Mapping[str, Any]) -> dict[str, Any]:
    """Analogue of `_merged_document()` for a Microsim document: the
    strategy supplies the states/transitions, everything else (parameters,
    variables, distributions, tables, n_patients, random_seed, trackers,
    tracker_reset_timing) stays shared at the document level -- Microsim's
    `document["strategies"]` shape matches Markov's exactly, unlike Tree.
    """

    merged = dict(document)
    merged["id"] = strategy.get("id", "microsim")
    merged["name"] = strategy.get("name") or strategy.get("id", "Microsim model")
    merged["states"] = strategy.get("states", [])
    merged["transitions"] = strategy.get("transitions", [])
    return merged


def sensitivity_inputs(document: Mapping[str, Any]) -> tuple[list[SensitivityInput], list[str]]:
    """Return all named parameters and variables with their SA bounds."""

    try:
        parameters, variables, distributions, tables = _build_inputs(document)
    except (KeyError, ValueError) as exc:
        return [], [str(exc)]
    inputs = [
        SensitivityInput(
            name=name,
            kind="parameter",
            base=parameter.base,
            low=parameter.low,
            high=parameter.high,
        )
        for name, parameter in parameters.items()
    ]
    # A variable's formula can reference the engine's reserved
    # `_cycle`/`_tenure` names (e.g. `DistToProb(pfs, _cycle, 1)`), which
    # a real Markov run binds to the current cycle as run_markov() steps
    # through the model (see markov_engine._engine_variables). This SA
    # input listing has no such loop -- it just needs each variable's
    # base value once -- so, matching how the standalone editor preview
    # handles the same gap (see app.py's calculate_document_values), bind
    # them to cycle 0 / tenure 1 here too. This copy is local to this
    # preview evaluation only: the real `parameters` dict returned above
    # (and later handed to run_markov for the actual Tornado/SA scenario
    # runs) is left untouched, so the real per-cycle `_cycle` value used
    # during those runs is never shadowed by this fixed preview value.
    preview_parameters = {
        **parameters,
        "_cycle": Param(name="_cycle", base=0.0),
        "_tenure": Param(name="_tenure", base=1.0),
    }
    for raw in document.get("variables", []):
        name = str(raw.get("name", ""))
        if not name:
            continue
        try:
            base = evaluate_expression(name, preview_parameters, variables, distributions, tables)
        except ValueError as exc:
            return [], [f"{name}: {exc}"]
        inputs.append(
            SensitivityInput(
                name=name,
                kind="variable",
                base=base,
                low=float(raw["low"]) if raw.get("low") is not None else None,
                high=float(raw["high"]) if raw.get("high") is not None else None,
            )
        )
    return inputs, []


def _outcome_value(outcome: StrategyOutcome, metric: str) -> float | None:
    if metric == "nmb":
        return outcome.incremental_nmb
    if metric == "icer":
        return outcome.icer
    if metric == "total_cost":
        return outcome.total_cost
    if metric == "total_effect":
        return outcome.total_effect
    if metric in ("incremental_cost", "incremental_effect"):
        return getattr(outcome, metric)
    if outcome.extras is not None and metric in outcome.extras:
        return outcome.extras[metric]
    raise ValueError(f"Unsupported sensitivity outcome '{metric}'")


def _sample_range(low: float, high: float, points: int) -> list[float]:
    if points < 2:
        raise ValueError("Sensitivity analysis requires at least two points")
    step = (high - low) / (points - 1)
    return [low + index * step for index in range(points)]


def _sample_range_by_interval(low: float, high: float, interval: float) -> list[float]:
    """Evenly-stepped values from `low` to `high`, exactly `interval`
    apart -- so the step between each new tested value and the
    previous one is explicit, instead of a fixed
    (and previously hidden) point count. The sweep always starts at
    `low` and always ends exactly at `high`; the final step is shorter
    than `interval` when (high - low) isn't an exact multiple of it, so
    the full range is still covered."""

    if interval <= 0:
        raise ValueError("Interval must be greater than 0")
    if high < low:
        low, high = high, low
    span = high - low
    # Computing each value from its step INDEX (not by repeatedly adding
    # `interval`) avoids float drift accumulating over many steps.
    n_full_steps = int(span // interval)
    values = [low + index * interval for index in range(n_full_steps + 1)]
    if not values or values[-1] < high - 1e-9:
        values.append(high)
    return values


def _scenario_outcome(
    document: Mapping[str, Any],
    overrides: Mapping[str, float],
    reference_strategy_id: str | None,
    target_strategy_id: str | None,
    willingness_to_pay: float,
    metric: str,
    evaluate_scenario: Callable[..., "SensitivityResult"] | None = None,
) -> tuple[float | None, str | None]:
    evaluate_scenario = evaluate_scenario or evaluate_markov_scenario
    result = evaluate_scenario(
        document,
        overrides=overrides,
        reference_strategy_id=reference_strategy_id,
        willingness_to_pay=willingness_to_pay,
    )
    if result.errors:
        return None, "; ".join(result.errors)
    target_id = target_strategy_id or next(
        (outcome.strategy_id for outcome in result.outcomes if outcome.strategy_id != result.reference_strategy_id),
        result.reference_strategy_id,
    )
    outcome = next((item for item in result.outcomes if item.strategy_id == target_id), None)
    if outcome is None:
        return None, f"Unknown target strategy '{target_id}'"
    try:
        value = _outcome_value(outcome, metric)
    except ValueError as exc:
        return None, str(exc)
    return value, None if value is not None else f"{metric.upper()} is undefined for the selected strategy"


def run_one_way(
    document: Mapping[str, Any],
    input_name: str,
    *,
    points: int = 21,
    interval: float | None = None,
    reference_strategy_id: str | None = None,
    target_strategy_id: str | None = None,
    willingness_to_pay: float = 50_000.0,
    metric: str = "nmb",
    evaluate_scenario: Callable[..., "SensitivityResult"] | None = None,
) -> tuple[list[dict[str, float]], list[str]]:
    """Vary one bounded input across evenly spaced values.

    By default the input is swept using `points` evenly-spaced values
    between its Low and High bounds. Pass `interval` -- the exact
    spacing between consecutive tested values -- to sweep by a fixed
    step instead; it takes priority over `points` when both are given."""

    inputs, errors = sensitivity_inputs(document)
    item = next((candidate for candidate in inputs if candidate.name == input_name), None)
    if item is None:
        return [], [f"Unknown sensitivity input '{input_name}'"]
    if item.low is None or item.high is None:
        return [], [f"Enter low and high values for: {input_name}"]
    sampled_values = (
        _sample_range_by_interval(item.low, item.high, interval)
        if interval is not None and interval > 0
        else _sample_range(item.low, item.high, points)
    )
    rows: list[dict[str, float]] = []
    for value in sampled_values:
        outcome, error = _scenario_outcome(document, {input_name: value}, reference_strategy_id, target_strategy_id, willingness_to_pay, metric, evaluate_scenario)
        if error:
            errors.append(f"{input_name}={value:g}: {error}")
        else:
            rows.append({"value": value, "outcome": outcome})
    return rows, errors


def run_grid(
    document: Mapping[str, Any],
    input_names: list[str],
    *,
    fixed_overrides: Mapping[str, float] | None = None,
    points: int = 21,
    first_interval: float | None = None,
    second_interval: float | None = None,
    reference_strategy_id: str | None = None,
    target_strategy_id: str | None = None,
    willingness_to_pay: float = 50_000.0,
    metric: str = "nmb",
    evaluate_scenario: Callable[..., "SensitivityResult"] | None = None,
) -> tuple[list[dict[str, float]], list[str]]:
    """Run a two-dimensional grid for the first two selected inputs.

    Each axis defaults to `points` evenly-spaced values; pass
    `first_interval`/`second_interval` to sweep that axis by a fixed
    step instead (see run_one_way's `interval`)."""

    if len(input_names) != 2:
        return [], ["Grid analysis requires exactly two inputs"]
    inputs, errors = sensitivity_inputs(document)
    selected = {item.name: item for item in inputs}
    first, second = (selected.get(name) for name in input_names)
    if first is None or second is None:
        return [], ["Select valid sensitivity inputs"]
    if first.low is None or first.high is None or second.low is None or second.high is None:
        return [], [f"Enter low and high values for: {', '.join(name for name, item in ((first.name, first), (second.name, second)) if item.low is None or item.high is None)}"]
    first_values = (
        _sample_range_by_interval(first.low, first.high, first_interval)
        if first_interval is not None and first_interval > 0
        else _sample_range(first.low, first.high, points)
    )
    second_values = (
        _sample_range_by_interval(second.low, second.high, second_interval)
        if second_interval is not None and second_interval > 0
        else _sample_range(second.low, second.high, points)
    )
    rows: list[dict[str, float]] = []
    for first_value in first_values:
        for second_value in second_values:
            overrides = dict(fixed_overrides or {})
            overrides.update({first.name: first_value, second.name: second_value})
            outcome, error = _scenario_outcome(document, overrides, reference_strategy_id, target_strategy_id, willingness_to_pay, metric, evaluate_scenario)
            if error:
                errors.append(f"{first.name}={first_value:g}, {second.name}={second_value:g}: {error}")
            else:
                rows.append({first.name: first_value, second.name: second_value, "outcome": outcome})
    return rows, errors


def find_threshold(
    document: Mapping[str, Any],
    input_name: str,
    *,
    points: int = 101,
    interval: float | None = None,
    reference_strategy_id: str | None = None,
    target_strategy_id: str | None = None,
    willingness_to_pay: float = 50_000.0,
    evaluate_scenario: Callable[..., "SensitivityResult"] | None = None,
) -> tuple[float | None, list[str]]:
    """Return the first sampled value where incremental NMB changes sign."""

    rows, errors = run_one_way(
        document,
        input_name,
        points=points,
        interval=interval,
        reference_strategy_id=reference_strategy_id,
        target_strategy_id=target_strategy_id,
        willingness_to_pay=willingness_to_pay,
        metric="nmb",
        evaluate_scenario=evaluate_scenario,
    )
    if errors:
        return None, errors
    for previous, current in zip(rows, rows[1:]):
        if previous["outcome"] == 0 or previous["outcome"] * current["outcome"] < 0:
            return (previous["value"] + current["value"]) / 2, []
    return None, ["No NMB threshold was found within the selected bounds"]


def run_scenarios(
    document: Mapping[str, Any],
    scenarios: Mapping[str, Mapping[str, float]],
    *,
    reference_strategy_id: str | None = None,
    target_strategy_id: str | None = None,
    willingness_to_pay: float = 50_000.0,
    metric: str = "nmb",
    evaluate_scenario: Callable[..., "SensitivityResult"] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Evaluate named scenario rows, used by scenario and multi-way analysis."""

    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for name, overrides in scenarios.items():
        outcome, error = _scenario_outcome(document, overrides, reference_strategy_id, target_strategy_id, willingness_to_pay, metric, evaluate_scenario)
        if error:
            errors.append(f"{name}: {error}")
        else:
            rows.append({"scenario": name, "outcome": outcome, "overrides": dict(overrides)})
    return rows, errors


def evaluate_markov_scenario(
    document: Mapping[str, Any],
    overrides: Mapping[str, float] | None = None,
    reference_strategy_id: str | None = None,
    willingness_to_pay: float = 50_000.0,
    allow_negative_cost: bool = False,
) -> SensitivityResult:
    """Run all Markov strategies for one deterministic parameter scenario.

    Overrides replace a parameter's base value or a variable's final definition
    only in an internal copy of the document. The saved model remains unchanged.
    """

    scenario = copy.deepcopy(dict(document))
    errors = _apply_overrides(scenario, overrides or {})
    if errors:
        return SensitivityResult(willingness_to_pay=willingness_to_pay, errors=errors)
    strategies = scenario.get("strategies", [])
    if not strategies:
        return SensitivityResult(willingness_to_pay=willingness_to_pay, errors=["No strategies defined"])
    strategy_ids = [str(strategy.get("id", "")) for strategy in strategies]
    reference_id = reference_strategy_id or strategy_ids[0]
    if reference_id not in strategy_ids:
        return SensitivityResult(
            willingness_to_pay=willingness_to_pay,
            errors=[f"Unknown reference strategy '{reference_id}'"],
        )
    try:
        parameters, variables, distributions, tables = _build_inputs(scenario)
    except (KeyError, ValueError) as exc:
        return SensitivityResult(
            reference_strategy_id=reference_id,
            willingness_to_pay=willingness_to_pay,
            errors=[str(exc)],
        )

    cycles_per_year = float(scenario.get("cycles_per_year", 1.0)) or 1.0
    cost_discount = _per_cycle_rate(
        float(scenario.get("discount_cost_value", 0.0)),
        str(scenario.get("discount_cost_unit", "per cycle")),
        cycles_per_year,
    )
    effect_discount = _per_cycle_rate(
        float(scenario.get("discount_effect_value", 0.0)),
        str(scenario.get("discount_effect_unit", "per cycle")),
        cycles_per_year,
    )
    raw_outcomes: list[tuple[str, str, float, float]] = []
    for strategy in strategies:
        strategy_id = str(strategy.get("id", ""))
        strategy_name = str(strategy.get("name") or strategy_id)
        try:
            model = resolve_markov(_merged_document(scenario, strategy))
        except ValueError as exc:
            errors.append(f"{strategy_name}: {exc}")
            continue
        result = run_markov(
            model,
            parameters,
            discount_cost_per_cycle=cost_discount,
            discount_effect_per_cycle=effect_discount,
            allow_negative_cost=allow_negative_cost,
            variables=variables,
            distributions=distributions,
            tables=tables,
        )
        if result.errors:
            errors.extend(f"{strategy_name}: {error.message}" for error in result.errors)
            continue
        raw_outcomes.append((strategy_id, strategy_name, result.total_cost, result.total_effect))
    if errors:
        return SensitivityResult(
            reference_strategy_id=reference_id,
            willingness_to_pay=willingness_to_pay,
            errors=errors,
        )

    reference = next(outcome for outcome in raw_outcomes if outcome[0] == reference_id)
    reference_nmb = willingness_to_pay * reference[3] - reference[2]
    outcomes = []
    for strategy_id, strategy_name, total_cost, total_effect in raw_outcomes:
        nmb = willingness_to_pay * total_effect - total_cost
        incremental_cost = total_cost - reference[2]
        incremental_effect = total_effect - reference[3]
        outcomes.append(
            StrategyOutcome(
                strategy_id=strategy_id,
                strategy_name=strategy_name,
                total_cost=total_cost,
                total_effect=total_effect,
                nmb=nmb,
                incremental_cost=incremental_cost,
                incremental_effect=incremental_effect,
                incremental_nmb=nmb - reference_nmb,
                icer=incremental_cost / incremental_effect if incremental_effect else None,
            )
        )
    optimal = max(outcomes, key=lambda outcome: outcome.nmb)
    return SensitivityResult(
        reference_strategy_id=reference_id,
        willingness_to_pay=willingness_to_pay,
        outcomes=outcomes,
        optimal_strategy_id=optimal.strategy_id,
    )


def evaluate_microsim_scenario(
    document: Mapping[str, Any],
    overrides: Mapping[str, float] | None = None,
    reference_strategy_id: str | None = None,
    willingness_to_pay: float = 50_000.0,
    allow_negative_cost: bool = False,
    psa_rng: np.random.Generator | None = None,
) -> SensitivityResult:
    """Run all Microsim strategies for one deterministic parameter scenario.

    Mirrors evaluate_markov_scenario() exactly -- Microsim's
    `document["strategies"]` shape is identical to Markov's -- with two
    differences: it resolves/runs each strategy through the Microsim
    engine instead of the Markov cohort engine, and it accepts an optional
    `psa_rng`.

    `psa_rng` is how run_psa() (psa.py) drives a real, two-level
    Monte-Carlo PSA for microsimulation (TreeAge/Andronis & Barton 2016
    methodology): the OUTER PSA loop resamples "per_psa"-tier
    (structural/parameter-uncertainty) Distributions once per PSA trial;
    the INNER loop, inside each strategy's own run_microsim(), simulates
    `model.n_patients` patients with their own "per_patient"/"per_cycle"
    variability on top. When `psa_rng` is given, this function draws the
    "per_psa" tier exactly ONCE per call (i.e. once per PSA trial) and
    threads that SAME draw into every strategy compared in the trial --
    standard PSA practice, so every strategy is compared under the same
    structural assumptions within a trial. When `psa_rng` is None (the
    plain Run button, or a deterministic SA scenario -- nothing that loops
    over PSA trials), no outer draw happens and any "per_psa" Distribution
    instead folds into "per_model" inside each strategy's own
    run_microsim() call, unchanged from before this existed.

    Overrides replace a parameter's base value or a variable's final definition
    only in an internal copy of the document. The saved model remains unchanged.
    """

    scenario = copy.deepcopy(dict(document))
    errors = _apply_overrides(scenario, overrides or {})
    if errors:
        return SensitivityResult(willingness_to_pay=willingness_to_pay, errors=errors)
    strategies = scenario.get("strategies", [])
    if not strategies:
        return SensitivityResult(willingness_to_pay=willingness_to_pay, errors=["No strategies defined"])
    strategy_ids = [str(strategy.get("id", "")) for strategy in strategies]
    reference_id = reference_strategy_id or strategy_ids[0]
    if reference_id not in strategy_ids:
        return SensitivityResult(
            willingness_to_pay=willingness_to_pay,
            errors=[f"Unknown reference strategy '{reference_id}'"],
        )
    try:
        parameters, variables, distributions, tables = _build_inputs(scenario)
    except (KeyError, ValueError) as exc:
        return SensitivityResult(
            reference_strategy_id=reference_id,
            willingness_to_pay=willingness_to_pay,
            errors=[str(exc)],
        )

    cycles_per_year = float(scenario.get("cycles_per_year", 1.0)) or 1.0
    cost_discount = _per_cycle_rate(
        float(scenario.get("discount_cost_value", 0.0)),
        str(scenario.get("discount_cost_unit", "per cycle")),
        cycles_per_year,
    )
    effect_discount = _per_cycle_rate(
        float(scenario.get("discount_effect_value", 0.0)),
        str(scenario.get("discount_effect_unit", "per cycle")),
        cycles_per_year,
    )

    psa_tier_values: dict[str, float] | None = None
    if psa_rng is not None:
        psa_tier_values = _resample_distributions(
            distributions,
            "per_psa",
            psa_rng,
            parameters,
            {**variables, "_cycle": 0, "_tenure": 1},
            tables,
            fold_psa_into_model=False,
        )

    raw_outcomes: list[tuple[str, str, float, float]] = []
    for strategy in strategies:
        strategy_id = str(strategy.get("id", ""))
        strategy_name = str(strategy.get("name") or strategy_id)
        try:
            model = resolve_microsim(_microsim_merged_document(scenario, strategy))
        except ValueError as exc:
            errors.append(f"{strategy_name}: {exc}")
            continue
        result = run_microsim(
            model,
            parameters,
            discount_cost_per_cycle=cost_discount,
            discount_effect_per_cycle=effect_discount,
            allow_negative_cost=allow_negative_cost,
            variables=variables,
            distributions=distributions,
            tables=tables,
            psa_tier_values=psa_tier_values,
        )
        if result.errors:
            errors.extend(f"{strategy_name}: {error.message}" for error in result.errors)
            continue
        patient_costs_arr = np.asarray(result.patient_costs, dtype=float)
        patient_effects_arr = np.asarray(result.patient_effects, dtype=float)
        n_patients_sampled = int(patient_costs_arr.size)
        if n_patients_sampled > 1:
            cost_variance = float(np.var(patient_costs_arr, ddof=1))
            effect_variance = float(np.var(patient_effects_arr, ddof=1))
            cost_effect_covariance = float(np.cov(patient_costs_arr, patient_effects_arr, ddof=1)[0, 1])
        else:
            cost_variance = effect_variance = cost_effect_covariance = 0.0
        raw_outcomes.append((
            strategy_id, strategy_name, result.total_cost, result.total_effect,
            cost_variance, effect_variance, cost_effect_covariance, n_patients_sampled,
        ))
    if errors:
        return SensitivityResult(
            reference_strategy_id=reference_id,
            willingness_to_pay=willingness_to_pay,
            errors=errors,
        )

    reference = next(outcome for outcome in raw_outcomes if outcome[0] == reference_id)
    reference_nmb = willingness_to_pay * reference[3] - reference[2]
    outcomes = []
    for (
        strategy_id, strategy_name, total_cost, total_effect,
        cost_variance, effect_variance, cost_effect_covariance, n_patients_sampled,
    ) in raw_outcomes:
        nmb = willingness_to_pay * total_effect - total_cost
        incremental_cost = total_cost - reference[2]
        incremental_effect = total_effect - reference[3]
        outcomes.append(
            StrategyOutcome(
                strategy_id=strategy_id,
                strategy_name=strategy_name,
                total_cost=total_cost,
                total_effect=total_effect,
                nmb=nmb,
                incremental_cost=incremental_cost,
                incremental_effect=incremental_effect,
                incremental_nmb=nmb - reference_nmb,
                icer=incremental_cost / incremental_effect if incremental_effect else None,
                patient_cost_variance=cost_variance,
                patient_effect_variance=effect_variance,
                patient_cost_effect_covariance=cost_effect_covariance,
                n_patients=n_patients_sampled,
            )
        )
    optimal = max(outcomes, key=lambda outcome: outcome.nmb)
    return SensitivityResult(
        reference_strategy_id=reference_id,
        willingness_to_pay=willingness_to_pay,
        outcomes=outcomes,
        optimal_strategy_id=optimal.strategy_id,
    )


def evaluate_tree_scenario(
    document: Mapping[str, Any],
    overrides: Mapping[str, float] | None = None,
    reference_strategy_id: str | None = None,
    willingness_to_pay: float = 50_000.0,
    allow_negative_cost: bool = False,
) -> SensitivityResult:
    """Run every Decision Tree strategy (a branch of the root decision node)
    for one deterministic parameter scenario.

    Mirrors evaluate_markov_scenario's contract exactly so the same
    run_one_way/run_grid/find_threshold/run_scenarios/run_tornado sampling
    helpers work for Decision Tree documents too, by passing
    evaluate_scenario=evaluate_tree_scenario. Overrides replace a
    parameter's base value or a variable's final definition only in an
    internal copy of the document; the saved model remains unchanged.
    """

    scenario = copy.deepcopy(dict(document))
    errors = _apply_overrides(scenario, overrides or {})
    if errors:
        return SensitivityResult(willingness_to_pay=willingness_to_pay, errors=errors)

    try:
        tree, parameters = resolve_tree(scenario)
    except ValueError as exc:
        return SensitivityResult(willingness_to_pay=willingness_to_pay, errors=[str(exc)])

    scenario_distributions = {
        str(raw["name"]): raw for raw in scenario.get("distributions", []) if raw.get("name")
    }
    scenario_tables = build_tables_mapping(scenario.get("tables"))
    cost_discount_rate = float(scenario.get("discount_cost_value", 0.0) or 0.0)
    effect_discount_rate = float(scenario.get("discount_effect_value", 0.0) or 0.0)

    result = Engine().run(
        tree,
        parameters,
        allow_negative_cost=allow_negative_cost,
        cost_discount_rate=cost_discount_rate,
        effect_discount_rate=effect_discount_rate,
        distributions=scenario_distributions,
        tables=scenario_tables,
    )
    if result.errors:
        return SensitivityResult(
            willingness_to_pay=willingness_to_pay,
            errors=[error.message for error in result.errors],
        )

    strategy_ids = list(result.strategies.keys())
    if not strategy_ids:
        return SensitivityResult(willingness_to_pay=willingness_to_pay, errors=["No strategies defined"])
    reference_id = reference_strategy_id or strategy_ids[0]
    if reference_id not in result.strategies:
        return SensitivityResult(
            willingness_to_pay=willingness_to_pay,
            errors=[f"Unknown reference strategy '{reference_id}'"],
        )

    reference = result.strategies[reference_id]
    reference_nmb = willingness_to_pay * reference.effect - reference.cost
    outcomes = []
    for strategy_id, strategy_result in result.strategies.items():
        nmb = willingness_to_pay * strategy_result.effect - strategy_result.cost
        incremental_cost = strategy_result.cost - reference.cost
        incremental_effect = strategy_result.effect - reference.effect
        outcomes.append(
            StrategyOutcome(
                strategy_id=strategy_id,
                strategy_name=strategy_id,
                total_cost=strategy_result.cost,
                total_effect=strategy_result.effect,
                nmb=nmb,
                incremental_cost=incremental_cost,
                incremental_effect=incremental_effect,
                incremental_nmb=nmb - reference_nmb,
                icer=incremental_cost / incremental_effect if incremental_effect else None,
            )
        )
    optimal = max(outcomes, key=lambda outcome: outcome.nmb)
    return SensitivityResult(
        reference_strategy_id=reference_id,
        willingness_to_pay=willingness_to_pay,
        outcomes=outcomes,
        optimal_strategy_id=optimal.strategy_id,
    )


def run_tornado(
    document: Mapping[str, Any],
    *,
    included_inputs: list[str] | None = None,
    reference_strategy_id: str | None = None,
    target_strategy_id: str | None = None,
    willingness_to_pay: float = 50_000.0,
    metric: str = "nmb",
    allow_negative_cost: bool = False,
    evaluate_scenario: Callable[..., "SensitivityResult"] | None = None,
) -> tuple[list[TornadoEntry], list[str]]:
    """Run low/high one-way scenarios for every selected SA input.

    The returned entries are sorted by absolute outcome range, ready for a
    Tornado chart. Every selected input must have both low and high bounds.
    """

    evaluate_scenario = evaluate_scenario or evaluate_markov_scenario
    inputs, errors = sensitivity_inputs(document)
    if errors:
        return [], errors
    selected = set(included_inputs) if included_inputs is not None else {item.name for item in inputs}
    chosen_inputs = [item for item in inputs if item.name in selected]
    missing = [item.name for item in chosen_inputs if item.low is None or item.high is None]
    if missing:
        return [], [f"Enter low and high values for: {', '.join(missing)}"]
    if not chosen_inputs:
        return [], ["Select at least one parameter or variable"]

    base_result = evaluate_scenario(
        document,
        reference_strategy_id=reference_strategy_id,
        willingness_to_pay=willingness_to_pay,
        allow_negative_cost=allow_negative_cost,
    )
    if base_result.errors:
        return [], base_result.errors
    target_id = target_strategy_id or next(
        (outcome.strategy_id for outcome in base_result.outcomes if outcome.strategy_id != base_result.reference_strategy_id),
        base_result.reference_strategy_id,
    )
    if target_id is None or not any(outcome.strategy_id == target_id for outcome in base_result.outcomes):
        return [], [f"Unknown target strategy '{target_id}'"]

    entries: list[TornadoEntry] = []
    for item in chosen_inputs:
        low_result = evaluate_scenario(
            document,
            overrides={item.name: item.low},
            reference_strategy_id=base_result.reference_strategy_id,
            willingness_to_pay=willingness_to_pay,
            allow_negative_cost=allow_negative_cost,
        )
        high_result = evaluate_scenario(
            document,
            overrides={item.name: item.high},
            reference_strategy_id=base_result.reference_strategy_id,
            willingness_to_pay=willingness_to_pay,
            allow_negative_cost=allow_negative_cost,
        )
        scenario_errors = low_result.errors + high_result.errors
        if scenario_errors:
            errors.extend(f"{item.name}: {error}" for error in scenario_errors)
            continue
        low_outcome = next(outcome for outcome in low_result.outcomes if outcome.strategy_id == target_id)
        high_outcome = next(outcome for outcome in high_result.outcomes if outcome.strategy_id == target_id)
        try:
            low_value = _outcome_value(low_outcome, metric)
            high_value = _outcome_value(high_outcome, metric)
        except ValueError as exc:
            return [], [str(exc)]
        if low_value is None or high_value is None:
            errors.append(f"{item.name}: {metric.upper()} is undefined for the selected strategy")
            continue
        entries.append(
            TornadoEntry(
                input_name=item.name,
                base=item.base,
                low=item.low,
                high=item.high,
                low_outcome=low_value,
                high_outcome=high_value,
            )
        )
    return sorted(entries, key=lambda entry: entry.range_width, reverse=True), errors