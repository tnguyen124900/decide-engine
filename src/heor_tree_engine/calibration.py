"""Model calibration: fit uncertain parameters/variables against observed
target data (e.g. a Kaplan-Meier survival curve digitized into cycle/value
pairs, or a single real-world outcome rate). Phase 1
only -- "local optimization" (bounded nonlinear least squares) and
"random/grid search" -- across all three model types:

- Markov and Microsim share the exact same trace-shaped objective
  (`_markov_residuals`/CalibrationTargetRow/markov_calibration_targets_from_rows(),
  reused as-is by Microsim's `run_microsim_calibration*` functions below,
  since a Microsim `trace` has the same {state_id: occupancy} per-cycle
  shape as the Markov cohort trace -- see microsim_engine.run_microsim()).
- Tree has no cycles, so it gets its own single-value objective
  (`_tree_residuals`/TreeCalibrationTargetRow/tree_calibration_targets_from_rows())
  further down: each target row is one observed number -- a strategy's
  expected total cost/effect, or the probability of reaching a specific
  terminal node (via engine.terminal_probabilities()) -- with no cycle
  dimension at all.

Every method within a model type shares one residual function so they
always score a candidate parameter set identically -- only how many
times, and in what order, they call it differs.

Phase 2 -- "Bayesian Calibration" -- adds two more algorithms, offered as
user-selectable alternatives to Phase 1's single-best-fit methods rather
than a replacement: Sampling-Importance-Resampling ("bayesian_sir") and
Metropolis-Hastings MCMC ("bayesian_mcmc"), see BayesianCalibrationResult
and _run_bayesian_calibration() further down. Unlike Phase 1 (deliberately
triplicated per model type to keep each version simple and independently
verifiable), SIR/MCMC share ONE core implementation across Markov/
Microsim/Tree -- see _run_bayesian_calibration()'s own docstring for why.
apply_posterior_to_psa() lets a Bayesian Calibration result be written
back into the model's PSA configuration (document["psa_parameters"]) as a
fitted parametric distribution, reusing psa.py's own Beta/Normal sampling
machinery unchanged.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .engine import Engine, des_state_share, des_strategy_runs, terminal_probabilities
from .expressions import build_tables_mapping
from .markov_engine import run_markov
from .markov_parser import resolve_markov
from .microsim_engine import run_microsim
from .microsim_parser import resolve_microsim
from .models import DecisionNode
from .parser import resolve as resolve_tree
from .psa import (
    PSAInputError,
    _as_float,
    _beta_from_mean_se,
    _gamma_from_mean_se,
    _lognormal_from_mean_se,
    make_sampler,
)
from .sensitivity import (
    _apply_overrides,
    _build_inputs,
    _merged_document,
    _microsim_merged_document,
    _per_cycle_rate,
)

# A residual this large makes an infeasible/errored candidate strictly worse
# than any real fit could plausibly be, without producing inf/nan (which
# would break scipy.optimize.least_squares's internal linear algebra).
_INFEASIBLE_RESIDUAL = 1.0e6


@dataclass(frozen=True)
class CalibrationParameter:
    """One Parameter or Variable allowed to vary during calibration, with
    the bounds the search is confined to. `initial` seeds the local
    optimizer's starting point; if omitted, the bounds' midpoint is used.
    """

    name: str
    low: float
    high: float
    initial: float | None = None

    def __post_init__(self) -> None:
        if self.high < self.low:
            raise ValueError(f"Calibration parameter '{self.name}': high ({self.high}) < low ({self.low})")

    def starting_value(self) -> float:
        if self.initial is not None:
            return min(max(self.initial, self.low), self.high)
        return (self.low + self.high) / 2.0


@dataclass(frozen=True)
class CalibrationTargetRow:
    """One row of the manual target-data table: the observed value a
    specific state's cycle-occupancy probability should be fit towards
    (e.g. one point digitized off a published Kaplan-Meier curve).
    """

    cycle: int
    state_id: str
    value: float
    weight: float = 1.0

    def __post_init__(self) -> None:
        if self.cycle < 0:
            raise ValueError(f"Calibration target: cycle must be >= 0, got {self.cycle}")
        if self.weight <= 0:
            raise ValueError(f"Calibration target: weight must be > 0, got {self.weight}")


@dataclass
class CalibrationResult:
    """Outcome of a calibration run -- always returned, even on failure
    (check `errors`), so the UI has one shape to render either way."""

    method: str = ""
    calibrated_values: dict[str, float] = field(default_factory=dict)
    initial_values: dict[str, float] = field(default_factory=dict)
    objective_value: float = float("inf")
    predicted_trace: list[dict[str, float]] = field(default_factory=list)
    predicted_at_targets: list[float] = field(default_factory=list)
    n_evaluations: int = 0
    converged: bool = False
    errors: list[str] = field(default_factory=list)


def markov_calibration_targets_from_rows(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[CalibrationTargetRow], list[str]]:
    """Parse+validate the UI's manual target-data table (a list of
    {cycle, state_id, value, weight?} dicts, as handed back by
    st.data_editor) into CalibrationTargetRow objects. Mirrors the
    partial-resolution pattern used elsewhere in this codebase: never
    raises, always returns (targets, errors), so the UI can show
    per-row problems without losing every other valid row.
    """

    targets: list[CalibrationTargetRow] = []
    errors: list[str] = []
    for index, row in enumerate(rows):
        label = f"Row {index + 1}"
        state_id = str(row.get("state_id") or row.get("state") or "").strip()
        if not state_id:
            errors.append(f"{label}: missing state")
            continue
        try:
            cycle = int(row["cycle"])
        except (KeyError, TypeError, ValueError):
            errors.append(f"{label} ({state_id}): cycle must be a whole number")
            continue
        try:
            value = float(row["value"])
        except (KeyError, TypeError, ValueError):
            errors.append(f"{label} ({state_id}): value must be a number")
            continue
        raw_weight = row.get("weight", 1.0)
        try:
            weight = float(raw_weight) if raw_weight not in (None, "") else 1.0
        except (TypeError, ValueError):
            errors.append(f"{label} ({state_id}): weight must be a number")
            continue
        try:
            targets.append(CalibrationTargetRow(cycle=cycle, state_id=state_id, value=value, weight=weight))
        except ValueError as exc:
            errors.append(f"{label} ({state_id}): {exc}")
    return targets, errors


@dataclass(frozen=True)
class TreeCalibrationTargetRow:
    """One row of the Tree calibration target table: a single observed
    value the chosen strategy's chosen metric should be fit towards --
    unlike Markov/Microsim's CalibrationTargetRow, there's no `cycle`
    field (a decision tree has no time dimension).

    `metric` is one of:
    - "total_cost" / "total_effect": the strategy's expected Cost/Effect
      rollup, exactly as shown in the Base Case results table.
    - "terminal_probability": the probability of reaching a specific
      terminal node under this strategy (see engine.terminal_probabilities()) --
      `target_id` names that terminal node's id and is required for this
      metric only.
    """

    metric: str
    value: float
    target_id: str | None = None
    weight: float = 1.0
    # "des_state_share" only: the time (years) the observed share refers to.
    time: float | None = None

    def __post_init__(self) -> None:
        if self.metric == "des_state_share":
            if not self.target_id:
                raise ValueError("Tree calibration target: 'des_state_share' rows need a state name")
            if self.time is None or self.time < 0:
                raise ValueError("Tree calibration target: 'des_state_share' rows need a time >= 0")
        elif self.metric not in ("total_cost", "total_effect", "terminal_probability"):
            raise ValueError(f"Tree calibration target: unknown metric '{self.metric}'")
        if self.metric == "terminal_probability" and not self.target_id:
            raise ValueError(
                "Tree calibration target: 'terminal_probability' rows need a target_id (terminal node id)"
            )
        if self.weight <= 0:
            raise ValueError(f"Tree calibration target: weight must be > 0, got {self.weight}")


def tree_calibration_targets_from_rows(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[TreeCalibrationTargetRow], list[str]]:
    """Parse+validate the Tree calibration UI's manual target table (a
    list of {metric, target_id?, value, weight?} dicts, as handed back by
    st.data_editor) into TreeCalibrationTargetRow objects. Same
    never-raises, always-(targets, errors) contract as
    markov_calibration_targets_from_rows()."""

    targets: list[TreeCalibrationTargetRow] = []
    errors: list[str] = []
    for index, row in enumerate(rows):
        label = f"Row {index + 1}"
        metric = str(row.get("metric") or "").strip()
        if metric not in ("total_cost", "total_effect", "terminal_probability", "des_state_share"):
            errors.append(
                f"{label}: metric must be one of total_cost/total_effect/terminal_probability/des_state_share"
            )
            continue
        time_value: float | None = None
        if metric == "des_state_share":
            try:
                time_value = float(row.get("time"))
            except (TypeError, ValueError):
                errors.append(f"{label} (des_state_share): time must be a number (years)")
                continue
        target_id = str(row.get("target_id") or row.get("terminal_id") or "").strip() or None
        if metric == "des_state_share" and not target_id:
            errors.append(f"{label}: des_state_share rows need a state name (e.g. Dead)")
            continue
        if metric == "terminal_probability" and not target_id:
            errors.append(f"{label}: terminal_probability rows need a terminal node")
            continue
        try:
            value = float(row["value"])
        except (KeyError, TypeError, ValueError):
            errors.append(f"{label} ({metric}): value must be a number")
            continue
        raw_weight = row.get("weight", 1.0)
        try:
            weight = float(raw_weight) if raw_weight not in (None, "") else 1.0
        except (TypeError, ValueError):
            errors.append(f"{label} ({metric}): weight must be a number")
            continue
        try:
            targets.append(
                TreeCalibrationTargetRow(metric=metric, value=value, target_id=target_id, weight=weight, time=time_value)
            )
        except ValueError as exc:
            errors.append(f"{label} ({metric}): {exc}")
    return targets, errors


def _find_strategy(document: Mapping[str, Any], strategy_id: str) -> dict[str, Any] | None:
    for strategy in document.get("strategies", []):
        if str(strategy.get("id", "")) == strategy_id:
            return strategy
    return None


def _markov_residuals(
    candidate: np.ndarray,
    *,
    document: Mapping[str, Any],
    strategy_id: str,
    parameter_names: Sequence[str],
    targets: Sequence[CalibrationTargetRow],
    allow_negative_cost: bool,
) -> np.ndarray:
    """Run one Markov strategy with `candidate` substituted for
    `parameter_names`, and return the weighted (predicted - target)
    residual for each target row, in the same order as `targets`. Any
    failure (unknown override, invalid model, a target cycle beyond the
    model's horizon) yields a large-but-finite residual for that row
    instead of raising, so an optimizer/search can keep exploring around
    an infeasible region rather than crashing.
    """

    overrides = {name: float(value) for name, value in zip(parameter_names, candidate)}
    scenario = copy.deepcopy(dict(document))
    override_errors = _apply_overrides(scenario, overrides)
    if override_errors:
        return np.full(len(targets), _INFEASIBLE_RESIDUAL)

    strategy = _find_strategy(scenario, strategy_id)
    if strategy is None:
        return np.full(len(targets), _INFEASIBLE_RESIDUAL)

    try:
        parameters, variables, distributions, tables = _build_inputs(scenario)
    except (KeyError, ValueError):
        return np.full(len(targets), _INFEASIBLE_RESIDUAL)

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

    try:
        model = resolve_markov(_merged_document(scenario, strategy))
    except ValueError:
        return np.full(len(targets), _INFEASIBLE_RESIDUAL)

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
        return np.full(len(targets), _INFEASIBLE_RESIDUAL)

    residuals = np.empty(len(targets))
    for i, target in enumerate(targets):
        if target.cycle >= len(result.trace):
            residuals[i] = _INFEASIBLE_RESIDUAL
            continue
        predicted = result.trace[target.cycle].get(target.state_id)
        if predicted is None:
            residuals[i] = _INFEASIBLE_RESIDUAL
            continue
        residuals[i] = target.weight * (predicted - target.value)
    return residuals


def _predicted_trace_and_values(
    document: Mapping[str, Any],
    strategy_id: str,
    parameter_names: Sequence[str],
    values: Sequence[float],
    targets: Sequence[CalibrationTargetRow],
    allow_negative_cost: bool,
) -> tuple[list[dict[str, float]], list[float]]:
    """Re-run the model once at a chosen candidate (the winner of a search,
    or the local optimizer's final point) to get the full trace for
    plotting plus the model's predicted value at each target row -- kept
    separate from `_markov_residuals` so callers that only need this final
    snapshot don't have to reconstruct it from raw residuals."""

    overrides = {name: float(value) for name, value in zip(parameter_names, values)}
    scenario = copy.deepcopy(dict(document))
    if _apply_overrides(scenario, overrides):
        return [], []
    strategy = _find_strategy(scenario, strategy_id)
    if strategy is None:
        return [], []
    try:
        parameters, variables, distributions, tables = _build_inputs(scenario)
        model = resolve_markov(_merged_document(scenario, strategy))
    except (KeyError, ValueError):
        return [], []
    cycles_per_year = float(scenario.get("cycles_per_year", 1.0)) or 1.0
    cost_discount = _per_cycle_rate(
        float(scenario.get("discount_cost_value", 0.0)), str(scenario.get("discount_cost_unit", "per cycle")), cycles_per_year
    )
    effect_discount = _per_cycle_rate(
        float(scenario.get("discount_effect_value", 0.0)), str(scenario.get("discount_effect_unit", "per cycle")), cycles_per_year
    )
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
        return [], []
    predicted_at_targets = [
        result.trace[t.cycle].get(t.state_id, float("nan")) if t.cycle < len(result.trace) else float("nan")
        for t in targets
    ]
    return result.trace, predicted_at_targets


def run_markov_calibration_local(
    document: Mapping[str, Any],
    strategy_id: str,
    parameters: Sequence[CalibrationParameter],
    targets: Sequence[CalibrationTargetRow],
    allow_negative_cost: bool = False,
    max_evaluations: int = 2000,
) -> CalibrationResult:
    """"Local Optimization": bounded nonlinear least squares
    (scipy.optimize.least_squares, Trust Region Reflective method -- the
    standard curve-fitting algorithm, and the technique TreeAge's
    "Auto-calibrate" uses under the hood). Converges fast when the
    residual surface is reasonably smooth; can settle in a local minimum
    for a poor starting point or a very non-smooth objective -- pair with
    Grid/Random Search below when that's a concern.
    """

    from scipy.optimize import least_squares  # local import: keep scipy optional for callers that never calibrate

    result = CalibrationResult(method="local")
    if not parameters:
        result.errors.append("Select at least one parameter to calibrate")
        return result
    if not targets:
        result.errors.append("Enter at least one target data row")
        return result
    if _find_strategy(document, strategy_id) is None:
        result.errors.append(f"Unknown strategy '{strategy_id}'")
        return result

    names = [p.name for p in parameters]
    x0 = np.array([p.starting_value() for p in parameters])
    lower = np.array([p.low for p in parameters])
    upper = np.array([p.high for p in parameters])
    result.initial_values = dict(zip(names, x0.tolist()))

    evaluation_count = 0

    def residuals(candidate: np.ndarray) -> np.ndarray:
        nonlocal evaluation_count
        evaluation_count += 1
        return _markov_residuals(
            candidate,
            document=document,
            strategy_id=strategy_id,
            parameter_names=names,
            targets=targets,
            allow_negative_cost=allow_negative_cost,
        )

    fit = least_squares(
        residuals,
        x0,
        bounds=(lower, upper),
        method="trf",
        max_nfev=max_evaluations,
    )
    result.n_evaluations = evaluation_count
    result.calibrated_values = dict(zip(names, fit.x.tolist()))
    result.objective_value = float(2.0 * fit.cost)  # scipy's `cost` is 0.5 * sum(residuals**2)
    result.converged = bool(fit.success)
    if not fit.success:
        result.errors.append(f"Optimizer did not fully converge: {fit.message}")
    trace, predicted = _predicted_trace_and_values(
        document, strategy_id, names, fit.x.tolist(), targets, allow_negative_cost
    )
    result.predicted_trace = trace
    result.predicted_at_targets = predicted
    return result


def run_markov_calibration_search(
    document: Mapping[str, Any],
    strategy_id: str,
    parameters: Sequence[CalibrationParameter],
    targets: Sequence[CalibrationTargetRow],
    method: str = "random",
    n_samples: int = 200,
    grid_points_per_dim: int = 5,
    random_seed: int | None = None,
    allow_negative_cost: bool = False,
    refine_with_local: bool = True,
) -> CalibrationResult:
    """"Grid / Random Search": try many candidate parameter sets (an
    evenly-spaced grid, or `n_samples` uniform-random draws) and keep the
    best-scoring one. Doesn't need a smooth objective or a good starting
    point (unlike Local Optimization above), at the cost of being much
    slower as the number of calibrated parameters grows. When
    `refine_with_local` is True (the default), the best sample found is
    then handed to `run_markov_calibration_local` as its starting point --
    a common "coarse-then-fine" combination that costs little extra and
    usually improves the final fit.
    """

    result = CalibrationResult(method=f"{method}_search")
    if method not in ("random", "grid"):
        result.errors.append(f"Unknown search method '{method}' (expected 'random' or 'grid')")
        return result
    if not parameters:
        result.errors.append("Select at least one parameter to calibrate")
        return result
    if not targets:
        result.errors.append("Enter at least one target data row")
        return result
    if _find_strategy(document, strategy_id) is None:
        result.errors.append(f"Unknown strategy '{strategy_id}'")
        return result

    names = [p.name for p in parameters]
    result.initial_values = {p.name: p.starting_value() for p in parameters}
    rng = np.random.default_rng(random_seed)

    if method == "random":
        candidates = rng.uniform(
            low=[p.low for p in parameters],
            high=[p.high for p in parameters],
            size=(n_samples, len(parameters)),
        )
    else:
        grid_points_per_dim = max(2, grid_points_per_dim)
        axes = [np.linspace(p.low, p.high, grid_points_per_dim) for p in parameters]
        mesh = np.meshgrid(*axes, indexing="ij")
        candidates = np.stack([m.ravel() for m in mesh], axis=1)

    best_score = float("inf")
    best_candidate = candidates[0]
    for candidate in candidates:
        residuals = _markov_residuals(
            candidate,
            document=document,
            strategy_id=strategy_id,
            parameter_names=names,
            targets=targets,
            allow_negative_cost=allow_negative_cost,
        )
        score = float(np.sum(residuals**2))
        if score < best_score:
            best_score = score
            best_candidate = candidate
    result.n_evaluations = len(candidates)
    result.calibrated_values = dict(zip(names, best_candidate.tolist()))
    result.objective_value = best_score
    result.converged = True

    if refine_with_local:
        refine_parameters = [
            CalibrationParameter(name=p.name, low=p.low, high=p.high, initial=float(value))
            for p, value in zip(parameters, best_candidate.tolist())
        ]
        refined = run_markov_calibration_local(
            document, strategy_id, refine_parameters, targets, allow_negative_cost=allow_negative_cost
        )
        result.n_evaluations += refined.n_evaluations
        if not refined.errors and refined.objective_value <= result.objective_value:
            result.calibrated_values = refined.calibrated_values
            result.objective_value = refined.objective_value
            result.predicted_trace = refined.predicted_trace
            result.predicted_at_targets = refined.predicted_at_targets
            result.converged = refined.converged
            return result

    trace, predicted = _predicted_trace_and_values(
        document, strategy_id, names, list(result.calibrated_values.values()), targets, allow_negative_cost
    )
    result.predicted_trace = trace
    result.predicted_at_targets = predicted
    return result


def run_markov_calibration(
    document: Mapping[str, Any],
    strategy_id: str,
    parameters: Sequence[CalibrationParameter],
    targets: Sequence[CalibrationTargetRow],
    method: str = "local",
    allow_negative_cost: bool = False,
    **kwargs: Any,
) -> CalibrationResult:
    """Single dispatch entry point the UI calls -- routes to the local
    optimizer or to grid/random search by `method` ("local" | "random" |
    "grid"), so the panel's algorithm dropdown only needs to pass its
    selected value straight through.
    """

    if method == "local":
        return run_markov_calibration_local(
            document, strategy_id, parameters, targets, allow_negative_cost=allow_negative_cost, **kwargs
        )
    if method in ("random", "grid"):
        return run_markov_calibration_search(
            document,
            strategy_id,
            parameters,
            targets,
            method=method,
            allow_negative_cost=allow_negative_cost,
            **kwargs,
        )
    result = CalibrationResult(method=method)
    result.errors.append(f"Unknown calibration method '{method}'")
    return result


def _microsim_residuals(
    candidate: np.ndarray,
    *,
    document: Mapping[str, Any],
    strategy_id: str,
    parameter_names: Sequence[str],
    targets: Sequence[CalibrationTargetRow],
    allow_negative_cost: bool,
) -> np.ndarray:
    """Microsim analogue of `_markov_residuals()`: identical shape (a
    Microsim `trace` is `{state_id: occupancy}` per cycle, exactly like
    the Markov cohort trace -- see microsim_engine.run_microsim()), just
    resolved/run through the Microsim engine instead. The document's own
    `n_patients`/`random_seed` are left untouched by calibration, so
    Common Random Numbers (the same per-patient RNG substreams reused
    across calls -- see run_microsim()'s own docstring) keep this
    residual reasonably smooth in the calibrated parameters despite
    Microsim's inherent Monte Carlo noise. Grid/Random Search is the
    safer choice over Local Optimization when that noise is large
    relative to the fit (few patients, or several calibrated parameters
    at once).
    """

    overrides = {name: float(value) for name, value in zip(parameter_names, candidate)}
    scenario = copy.deepcopy(dict(document))
    override_errors = _apply_overrides(scenario, overrides)
    if override_errors:
        return np.full(len(targets), _INFEASIBLE_RESIDUAL)

    strategy = _find_strategy(scenario, strategy_id)
    if strategy is None:
        return np.full(len(targets), _INFEASIBLE_RESIDUAL)

    try:
        parameters, variables, distributions, tables = _build_inputs(scenario)
    except (KeyError, ValueError):
        return np.full(len(targets), _INFEASIBLE_RESIDUAL)

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

    try:
        model = resolve_microsim(_microsim_merged_document(scenario, strategy))
    except ValueError:
        return np.full(len(targets), _INFEASIBLE_RESIDUAL)

    result = run_microsim(
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
        return np.full(len(targets), _INFEASIBLE_RESIDUAL)

    residuals = np.empty(len(targets))
    for i, target in enumerate(targets):
        if target.cycle >= len(result.trace):
            residuals[i] = _INFEASIBLE_RESIDUAL
            continue
        predicted = result.trace[target.cycle].get(target.state_id)
        if predicted is None:
            residuals[i] = _INFEASIBLE_RESIDUAL
            continue
        residuals[i] = target.weight * (predicted - target.value)
    return residuals


def _microsim_predicted_trace_and_values(
    document: Mapping[str, Any],
    strategy_id: str,
    parameter_names: Sequence[str],
    values: Sequence[float],
    targets: Sequence[CalibrationTargetRow],
    allow_negative_cost: bool,
) -> tuple[list[dict[str, float]], list[float]]:
    """Microsim analogue of `_predicted_trace_and_values()`."""

    overrides = {name: float(value) for name, value in zip(parameter_names, values)}
    scenario = copy.deepcopy(dict(document))
    if _apply_overrides(scenario, overrides):
        return [], []
    strategy = _find_strategy(scenario, strategy_id)
    if strategy is None:
        return [], []
    try:
        parameters, variables, distributions, tables = _build_inputs(scenario)
        model = resolve_microsim(_microsim_merged_document(scenario, strategy))
    except (KeyError, ValueError):
        return [], []
    cycles_per_year = float(scenario.get("cycles_per_year", 1.0)) or 1.0
    cost_discount = _per_cycle_rate(
        float(scenario.get("discount_cost_value", 0.0)), str(scenario.get("discount_cost_unit", "per cycle")), cycles_per_year
    )
    effect_discount = _per_cycle_rate(
        float(scenario.get("discount_effect_value", 0.0)), str(scenario.get("discount_effect_unit", "per cycle")), cycles_per_year
    )
    result = run_microsim(
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
        return [], []
    predicted_at_targets = [
        result.trace[t.cycle].get(t.state_id, float("nan")) if t.cycle < len(result.trace) else float("nan")
        for t in targets
    ]
    return result.trace, predicted_at_targets


def run_microsim_calibration_local(
    document: Mapping[str, Any],
    strategy_id: str,
    parameters: Sequence[CalibrationParameter],
    targets: Sequence[CalibrationTargetRow],
    allow_negative_cost: bool = False,
    max_evaluations: int = 2000,
    diff_step: float = 0.02,
) -> CalibrationResult:
    """Microsim analogue of `run_markov_calibration_local()` -- see
    `_microsim_residuals()` for the Monte-Carlo-noise caveat. Prefer a
    larger `model.n_patients` for a cleaner fit, or fall back to
    `run_microsim_calibration_search()` when this struggles to converge.

    `diff_step` (relative, passed straight to `scipy.optimize.least_squares`)
    defaults much larger than scipy's own default (~1.5e-8) on purpose: a
    Microsim residual is a *sum of per-patient indicator flips*, which is
    a step function of the candidate under Common Random Numbers, so an
    infinitesimal finite-difference probe almost always lands between two
    flips and measures an exact-zero gradient -- the optimizer then
    reports false "convergence" at the starting point without having
    moved at all (confirmed with a real-execution test during
    development). A relative step around 0.01-0.05 (default 0.02) is
    large enough that the probe reliably captures enough patients
    flipping to see a real, non-zero signal above the sampling noise,
    while still small enough for a locally-linear Jacobian approximation
    to be reasonable near the optimum.
    """

    from scipy.optimize import least_squares  # local import: keep scipy optional for callers that never calibrate

    result = CalibrationResult(method="local")
    if not parameters:
        result.errors.append("Select at least one parameter to calibrate")
        return result
    if not targets:
        result.errors.append("Enter at least one target data row")
        return result
    if _find_strategy(document, strategy_id) is None:
        result.errors.append(f"Unknown strategy '{strategy_id}'")
        return result

    names = [p.name for p in parameters]
    x0 = np.array([p.starting_value() for p in parameters])
    lower = np.array([p.low for p in parameters])
    upper = np.array([p.high for p in parameters])
    result.initial_values = dict(zip(names, x0.tolist()))

    evaluation_count = 0

    def residuals(candidate: np.ndarray) -> np.ndarray:
        nonlocal evaluation_count
        evaluation_count += 1
        return _microsim_residuals(
            candidate,
            document=document,
            strategy_id=strategy_id,
            parameter_names=names,
            targets=targets,
            allow_negative_cost=allow_negative_cost,
        )

    fit = least_squares(
        residuals,
        x0,
        bounds=(lower, upper),
        method="trf",
        max_nfev=max_evaluations,
        diff_step=diff_step,
    )
    result.n_evaluations = evaluation_count
    result.calibrated_values = dict(zip(names, fit.x.tolist()))
    result.objective_value = float(2.0 * fit.cost)  # scipy's `cost` is 0.5 * sum(residuals**2)
    result.converged = bool(fit.success)
    if not fit.success:
        result.errors.append(f"Optimizer did not fully converge: {fit.message}")
    trace, predicted = _microsim_predicted_trace_and_values(
        document, strategy_id, names, fit.x.tolist(), targets, allow_negative_cost
    )
    result.predicted_trace = trace
    result.predicted_at_targets = predicted
    return result


def run_microsim_calibration_search(
    document: Mapping[str, Any],
    strategy_id: str,
    parameters: Sequence[CalibrationParameter],
    targets: Sequence[CalibrationTargetRow],
    method: str = "random",
    n_samples: int = 200,
    grid_points_per_dim: int = 5,
    random_seed: int | None = None,
    allow_negative_cost: bool = False,
    refine_with_local: bool = True,
) -> CalibrationResult:
    """Microsim analogue of `run_markov_calibration_search()`."""

    result = CalibrationResult(method=f"{method}_search")
    if method not in ("random", "grid"):
        result.errors.append(f"Unknown search method '{method}' (expected 'random' or 'grid')")
        return result
    if not parameters:
        result.errors.append("Select at least one parameter to calibrate")
        return result
    if not targets:
        result.errors.append("Enter at least one target data row")
        return result
    if _find_strategy(document, strategy_id) is None:
        result.errors.append(f"Unknown strategy '{strategy_id}'")
        return result

    names = [p.name for p in parameters]
    result.initial_values = {p.name: p.starting_value() for p in parameters}
    rng = np.random.default_rng(random_seed)

    if method == "random":
        candidates = rng.uniform(
            low=[p.low for p in parameters],
            high=[p.high for p in parameters],
            size=(n_samples, len(parameters)),
        )
    else:
        grid_points_per_dim = max(2, grid_points_per_dim)
        axes = [np.linspace(p.low, p.high, grid_points_per_dim) for p in parameters]
        mesh = np.meshgrid(*axes, indexing="ij")
        candidates = np.stack([m.ravel() for m in mesh], axis=1)

    best_score = float("inf")
    best_candidate = candidates[0]
    for candidate in candidates:
        residuals = _microsim_residuals(
            candidate,
            document=document,
            strategy_id=strategy_id,
            parameter_names=names,
            targets=targets,
            allow_negative_cost=allow_negative_cost,
        )
        score = float(np.sum(residuals**2))
        if score < best_score:
            best_score = score
            best_candidate = candidate
    result.n_evaluations = len(candidates)
    result.calibrated_values = dict(zip(names, best_candidate.tolist()))
    result.objective_value = best_score
    result.converged = True

    if refine_with_local:
        refine_parameters = [
            CalibrationParameter(name=p.name, low=p.low, high=p.high, initial=float(value))
            for p, value in zip(parameters, best_candidate.tolist())
        ]
        refined = run_microsim_calibration_local(
            document, strategy_id, refine_parameters, targets, allow_negative_cost=allow_negative_cost
        )
        result.n_evaluations += refined.n_evaluations
        if not refined.errors and refined.objective_value <= result.objective_value:
            result.calibrated_values = refined.calibrated_values
            result.objective_value = refined.objective_value
            result.predicted_trace = refined.predicted_trace
            result.predicted_at_targets = refined.predicted_at_targets
            result.converged = refined.converged
            return result

    trace, predicted = _microsim_predicted_trace_and_values(
        document, strategy_id, names, list(result.calibrated_values.values()), targets, allow_negative_cost
    )
    result.predicted_trace = trace
    result.predicted_at_targets = predicted
    return result


def run_microsim_calibration(
    document: Mapping[str, Any],
    strategy_id: str,
    parameters: Sequence[CalibrationParameter],
    targets: Sequence[CalibrationTargetRow],
    method: str = "local",
    allow_negative_cost: bool = False,
    **kwargs: Any,
) -> CalibrationResult:
    """Microsim analogue of `run_markov_calibration()` -- same dispatch,
    routing on `method` ("local" | "random" | "grid").
    """

    if method == "local":
        return run_microsim_calibration_local(
            document, strategy_id, parameters, targets, allow_negative_cost=allow_negative_cost, **kwargs
        )
    if method in ("random", "grid"):
        return run_microsim_calibration_search(
            document,
            strategy_id,
            parameters,
            targets,
            method=method,
            allow_negative_cost=allow_negative_cost,
            **kwargs,
        )
    result = CalibrationResult(method=method)
    result.errors.append(f"Unknown calibration method '{method}'")
    return result


def _tree_strategy_ids(document: Mapping[str, Any]) -> list[str]:
    """Return a Tree document's strategy ids (the root decision node's
    branch labels -- see engine.evaluate_decision()/parser.resolve()), or
    [] if the document doesn't resolve at all. Tree documents have no
    `document["strategies"]` list the way Markov/Microsim do (a decision
    tree is one connected `nodes`/`root_id` graph, not a set of parallel
    strategy definitions), so unlike `_find_strategy()` above this has to
    actually resolve the tree to discover them."""

    try:
        tree, _parameters = resolve_tree(dict(document))
    except ValueError:
        return []
    if not isinstance(tree, DecisionNode):
        return []
    return [branch.label for branch in tree.branches]


def _tree_residuals(
    candidate: np.ndarray,
    *,
    document: Mapping[str, Any],
    strategy_id: str,
    parameter_names: Sequence[str],
    targets: Sequence[TreeCalibrationTargetRow],
    allow_negative_cost: bool,
) -> np.ndarray:
    """Tree analogue of `_markov_residuals()`: no cycle dimension, so each
    target row is scored against one of three single-number readouts of
    one strategy -- `total_cost`/`total_effect` (from the same rollback
    `Engine().run()` that powers Base Case results) or
    `terminal_probability` (via engine.terminal_probabilities(), computed
    at most once per candidate regardless of how many terminal-probability
    rows are in `targets`). Same soft-failure contract as the Markov/
    Microsim residual functions: any problem (bad override, unresolvable
    model, unknown strategy, unknown terminal node id) yields a large-but-
    finite residual for the affected row(s) instead of raising.
    """

    overrides = {name: float(value) for name, value in zip(parameter_names, candidate)}
    scenario = copy.deepcopy(dict(document))
    override_errors = _apply_overrides(scenario, overrides)
    if override_errors:
        return np.full(len(targets), _INFEASIBLE_RESIDUAL)

    try:
        tree, parameters = resolve_tree(scenario)
    except ValueError:
        return np.full(len(targets), _INFEASIBLE_RESIDUAL)
    if not isinstance(tree, DecisionNode):
        return np.full(len(targets), _INFEASIBLE_RESIDUAL)

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
        return np.full(len(targets), _INFEASIBLE_RESIDUAL)
    strategy_result = result.strategies.get(strategy_id)
    if strategy_result is None:
        return np.full(len(targets), _INFEASIBLE_RESIDUAL)

    terminal_probs: dict[str, float] | None = None
    des_runs: list | None = None
    residuals = np.empty(len(targets))
    for i, target in enumerate(targets):
        if target.metric == "total_cost":
            predicted = strategy_result.cost
        elif target.metric == "total_effect":
            predicted = strategy_result.effect
        elif target.metric == "des_state_share":
            if des_runs is None:
                try:
                    des_runs = des_strategy_runs(
                        tree, strategy_id, parameters,
                        distributions=scenario_distributions, tables=scenario_tables,
                    )
                except (ValueError, TypeError):
                    des_runs = []
            if not des_runs:
                residuals[i] = _INFEASIBLE_RESIDUAL
                continue
            predicted = des_state_share(des_runs, str(target.target_id), float(target.time or 0.0))
        else:  # "terminal_probability"
            if terminal_probs is None:
                try:
                    terminal_probs = terminal_probabilities(
                        tree, strategy_id, parameters,
                        distributions=scenario_distributions, tables=scenario_tables,
                    )
                except (ValueError, TypeError):
                    terminal_probs = {}
            predicted = terminal_probs.get(target.target_id)
            if predicted is None:
                residuals[i] = _INFEASIBLE_RESIDUAL
                continue
        residuals[i] = target.weight * (predicted - target.value)
    return residuals


def _predicted_tree_values(
    document: Mapping[str, Any],
    strategy_id: str,
    parameter_names: Sequence[str],
    values: Sequence[float],
    targets: Sequence[TreeCalibrationTargetRow],
    allow_negative_cost: bool,
) -> list[float]:
    """Re-run the tree once at a chosen candidate to get the model's
    predicted value at each target row -- kept separate from
    `_tree_residuals` exactly as `_predicted_trace_and_values` is for
    Markov, so callers that only need this final snapshot don't have to
    reconstruct it from raw residuals. Returns one float (NaN for a row
    that can't be read, e.g. an unknown terminal node id) per target, in
    the same order as `targets`, or [] if the model itself fails to run.
    """

    overrides = {name: float(value) for name, value in zip(parameter_names, values)}
    scenario = copy.deepcopy(dict(document))
    if _apply_overrides(scenario, overrides):
        return []
    try:
        tree, parameters = resolve_tree(scenario)
    except ValueError:
        return []
    if not isinstance(tree, DecisionNode):
        return []

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
        return []
    strategy_result = result.strategies.get(strategy_id)
    if strategy_result is None:
        return []

    terminal_probs: dict[str, float] | None = None
    des_runs: list | None = None
    predicted: list[float] = []
    for target in targets:
        if target.metric == "total_cost":
            predicted.append(strategy_result.cost)
        elif target.metric == "total_effect":
            predicted.append(strategy_result.effect)
        elif target.metric == "des_state_share":
            if des_runs is None:
                try:
                    des_runs = des_strategy_runs(
                        tree, strategy_id, parameters,
                        distributions=scenario_distributions, tables=scenario_tables,
                    )
                except (ValueError, TypeError):
                    des_runs = []
            predicted.append(
                des_state_share(des_runs, str(target.target_id), float(target.time or 0.0)) if des_runs else float("nan")
            )
        else:  # "terminal_probability"
            if terminal_probs is None:
                try:
                    terminal_probs = terminal_probabilities(
                        tree, strategy_id, parameters,
                        distributions=scenario_distributions, tables=scenario_tables,
                    )
                except (ValueError, TypeError):
                    terminal_probs = {}
            predicted.append(terminal_probs.get(target.target_id, float("nan")))
    return predicted


def run_tree_calibration_local(
    document: Mapping[str, Any],
    strategy_id: str,
    parameters: Sequence[CalibrationParameter],
    targets: Sequence[TreeCalibrationTargetRow],
    allow_negative_cost: bool = False,
    max_evaluations: int = 2000,
) -> CalibrationResult:
    """"Local Optimization" for a Decision Tree: same bounded nonlinear
    least squares as run_markov_calibration_local(), scored against
    `_tree_residuals()` instead. `predicted_trace` is always left empty
    on the returned CalibrationResult (there's no cycle-by-cycle trace to
    plot for a tree) -- use `predicted_at_targets` for the before/after
    comparison instead.
    """

    from scipy.optimize import least_squares  # local import: keep scipy optional for callers that never calibrate

    result = CalibrationResult(method="local")
    if not parameters:
        result.errors.append("Select at least one parameter to calibrate")
        return result
    if not targets:
        result.errors.append("Enter at least one target data row")
        return result
    if strategy_id not in _tree_strategy_ids(document):
        result.errors.append(f"Unknown strategy '{strategy_id}'")
        return result

    names = [p.name for p in parameters]
    x0 = np.array([p.starting_value() for p in parameters])
    lower = np.array([p.low for p in parameters])
    upper = np.array([p.high for p in parameters])
    result.initial_values = dict(zip(names, x0.tolist()))

    evaluation_count = 0

    def residuals(candidate: np.ndarray) -> np.ndarray:
        nonlocal evaluation_count
        evaluation_count += 1
        return _tree_residuals(
            candidate,
            document=document,
            strategy_id=strategy_id,
            parameter_names=names,
            targets=targets,
            allow_negative_cost=allow_negative_cost,
        )

    fit = least_squares(
        residuals,
        x0,
        bounds=(lower, upper),
        method="trf",
        max_nfev=max_evaluations,
    )
    result.n_evaluations = evaluation_count
    result.calibrated_values = dict(zip(names, fit.x.tolist()))
    result.objective_value = float(2.0 * fit.cost)  # scipy's `cost` is 0.5 * sum(residuals**2)
    result.converged = bool(fit.success)
    if not fit.success:
        result.errors.append(f"Optimizer did not fully converge: {fit.message}")
    result.predicted_at_targets = _predicted_tree_values(
        document, strategy_id, names, fit.x.tolist(), targets, allow_negative_cost
    )
    return result


def run_tree_calibration_search(
    document: Mapping[str, Any],
    strategy_id: str,
    parameters: Sequence[CalibrationParameter],
    targets: Sequence[TreeCalibrationTargetRow],
    method: str = "random",
    n_samples: int = 200,
    grid_points_per_dim: int = 5,
    random_seed: int | None = None,
    allow_negative_cost: bool = False,
    refine_with_local: bool = True,
) -> CalibrationResult:
    """"Grid / Random Search" for a Decision Tree -- same shape as
    run_markov_calibration_search(), scored against `_tree_residuals()`.
    """

    result = CalibrationResult(method=f"{method}_search")
    if method not in ("random", "grid"):
        result.errors.append(f"Unknown search method '{method}' (expected 'random' or 'grid')")
        return result
    if not parameters:
        result.errors.append("Select at least one parameter to calibrate")
        return result
    if not targets:
        result.errors.append("Enter at least one target data row")
        return result
    if strategy_id not in _tree_strategy_ids(document):
        result.errors.append(f"Unknown strategy '{strategy_id}'")
        return result

    names = [p.name for p in parameters]
    result.initial_values = {p.name: p.starting_value() for p in parameters}
    rng = np.random.default_rng(random_seed)

    if method == "random":
        candidates = rng.uniform(
            low=[p.low for p in parameters],
            high=[p.high for p in parameters],
            size=(n_samples, len(parameters)),
        )
    else:
        grid_points_per_dim = max(2, grid_points_per_dim)
        axes = [np.linspace(p.low, p.high, grid_points_per_dim) for p in parameters]
        mesh = np.meshgrid(*axes, indexing="ij")
        candidates = np.stack([m.ravel() for m in mesh], axis=1)

    best_score = float("inf")
    best_candidate = candidates[0]
    for candidate in candidates:
        residuals = _tree_residuals(
            candidate,
            document=document,
            strategy_id=strategy_id,
            parameter_names=names,
            targets=targets,
            allow_negative_cost=allow_negative_cost,
        )
        score = float(np.sum(residuals**2))
        if score < best_score:
            best_score = score
            best_candidate = candidate
    result.n_evaluations = len(candidates)
    result.calibrated_values = dict(zip(names, best_candidate.tolist()))
    result.objective_value = best_score
    result.converged = True

    if refine_with_local:
        refine_parameters = [
            CalibrationParameter(name=p.name, low=p.low, high=p.high, initial=float(value))
            for p, value in zip(parameters, best_candidate.tolist())
        ]
        refined = run_tree_calibration_local(
            document, strategy_id, refine_parameters, targets, allow_negative_cost=allow_negative_cost
        )
        result.n_evaluations += refined.n_evaluations
        if not refined.errors and refined.objective_value <= result.objective_value:
            result.calibrated_values = refined.calibrated_values
            result.objective_value = refined.objective_value
            result.predicted_at_targets = refined.predicted_at_targets
            result.converged = refined.converged
            return result

    result.predicted_at_targets = _predicted_tree_values(
        document, strategy_id, names, list(result.calibrated_values.values()), targets, allow_negative_cost
    )
    return result


def run_tree_calibration(
    document: Mapping[str, Any],
    strategy_id: str,
    parameters: Sequence[CalibrationParameter],
    targets: Sequence[TreeCalibrationTargetRow],
    method: str = "local",
    allow_negative_cost: bool = False,
    **kwargs: Any,
) -> CalibrationResult:
    """Single dispatch entry point for Tree calibration -- same shape as
    run_markov_calibration()/run_microsim_calibration().
    """

    if method == "local":
        return run_tree_calibration_local(
            document, strategy_id, parameters, targets, allow_negative_cost=allow_negative_cost, **kwargs
        )
    if method in ("random", "grid"):
        return run_tree_calibration_search(
            document,
            strategy_id,
            parameters,
            targets,
            method=method,
            allow_negative_cost=allow_negative_cost,
            **kwargs,
        )
    result = CalibrationResult(method=method)
    result.errors.append(f"Unknown calibration method '{method}'")
    return result


@dataclass
class BayesianCalibrationResult:
    """Outcome of a Bayesian calibration run (Sampling-Importance-Resampling
    or MCMC) -- always returned, even on failure (check `errors`). Unlike
    CalibrationResult (Phase 1's single best-fit point), this carries a
    full posterior SAMPLE per calibrated parameter, not just one value --
    """

    method: str = ""  # "bayesian_sir" | "bayesian_mcmc"
    parameter_names: list[str] = field(default_factory=list)
    posterior_samples: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))  # (n_samples, n_params)
    posterior_mean: dict[str, float] = field(default_factory=dict)
    posterior_sd: dict[str, float] = field(default_factory=dict)
    credible_interval_95: dict[str, tuple[float, float]] = field(default_factory=dict)
    prior_source: dict[str, str] = field(default_factory=dict)  # name -> "psa:<type>" | "uniform"
    predicted_trace: list[dict[str, float]] = field(default_factory=list)
    predicted_at_targets: list[float] = field(default_factory=list)
    n_evaluations: int = 0
    diagnostics: dict[str, float] = field(default_factory=dict)
    converged: bool = False
    errors: list[str] = field(default_factory=list)


def _psa_parameter_row(document: Mapping[str, Any], name: str) -> Mapping[str, Any] | None:
    """The document's own PSA configuration row for a named Parameter/
    Variable (target_id == "param:<name>"), if one exists -- see
    psa.parse_target_id() / app.py's PSA setup panel."""

    for row in document.get("psa_parameters", []):
        if str(row.get("target_id", "")) == f"param:{name}":
            return row
    return None


def _prior_log_pdf(psa_type: str, values: Mapping[str, Any]) -> Callable[[float], float]:
    """Log-density function for one of psa.py's parametric PSA distribution
    types, built the exact same way make_sampler() builds a sampler for it
    (same mean/SE -> native-shape-parameter conversions, so a Bayesian
    Calibration prior and the model's own PSA sampler always agree on what
    the distribution actually is). Used only by MCMC's Metropolis-Hastings
    acceptance ratio; SIR only ever needs samples, not densities."""

    from scipy import stats  # local import: keep scipy optional for callers that never run MCMC

    if psa_type == "Beta":
        alpha, beta = _beta_from_mean_se(_as_float(values, "mean", "Beta"), _as_float(values, "se", "Beta"))
        dist = stats.beta(alpha, beta)
    elif psa_type == "Gamma":
        shape, scale = _gamma_from_mean_se(_as_float(values, "mean", "Gamma"), _as_float(values, "se", "Gamma"))
        dist = stats.gamma(shape, scale=scale)
    elif psa_type == "Normal":
        mean = _as_float(values, "mean", "Normal")
        se = _as_float(values, "se", "Normal")
        if se <= 0:
            raise PSAInputError("Normal PSA input needs a positive SE to use as a Bayesian prior")
        dist = stats.norm(mean, se)
    elif psa_type == "Lognormal":
        mu, sigma = _lognormal_from_mean_se(_as_float(values, "mean", "Lognormal"), _as_float(values, "se", "Lognormal"))
        dist = stats.lognorm(sigma, scale=float(np.exp(mu)))
    elif psa_type == "Uniform":
        low = _as_float(values, "min", "Uniform")
        high = _as_float(values, "max", "Uniform")
        if not (low < high):
            raise PSAInputError("Uniform PSA input needs Min < Max")
        dist = stats.uniform(low, high - low)
    elif psa_type == "Triangular":
        low = _as_float(values, "min", "Triangular")
        mode = _as_float(values, "mode", "Triangular")
        high = _as_float(values, "max", "Triangular")
        if not (low <= mode <= high) or low == high:
            raise PSAInputError("Triangular PSA input needs Min <= Mode <= Max (and Min different from Max)")
        dist = stats.triang((mode - low) / (high - low), loc=low, scale=high - low)
    else:
        raise PSAInputError(f"Unsupported PSA distribution '{psa_type}'")
    return lambda x: float(dist.logpdf(x))


@dataclass(frozen=True)
class _Prior:
    """One calibrated parameter's Bayesian prior: a sampler (for SIR) and
    a log-density (for MCMC), always truncated to the parameter's own
    [low, high] calibration bounds regardless of source -- a posterior
    value outside those bounds was never a candidate Phase 1 could have
    returned either, so it isn't a meaningful answer here."""

    sample: Callable[[np.random.Generator], float]
    log_density: Callable[[float], float]
    source: str


def _uniform_prior(low: float, high: float) -> _Prior:
    span = high - low
    uniform_density = -float(np.log(span)) if span > 0 else 0.0
    return _Prior(
        sample=lambda rng: float(rng.uniform(low, high)),
        log_density=lambda x: uniform_density if low <= x <= high else float("-inf"),
        source="uniform",
    )


def _build_prior(parameter: CalibrationParameter, document: Mapping[str, Any]) -> _Prior:
    """The prior used for Bayesian Calibration of `parameter`: reuses the
    model's own PSA configuration for that Parameter/Variable
    (document["psa_parameters"], see psa.make_sampler()) when one is
    already set up, so a parameter that already has a PSA uncertainty
    distribution keeps using it here -- otherwise falls back to
    Uniform(low, high), the same implicit prior Phase 1's Random/Grid
    Search already used."""

    low, high = parameter.low, parameter.high
    row = _psa_parameter_row(document, parameter.name)
    if row is None:
        return _uniform_prior(low, high)

    psa_type = str(row.get("psa_type", "Beta"))
    values = row.get("values", {}) or {}
    try:
        base_sampler = make_sampler(psa_type, values)
        base_log_pdf = _prior_log_pdf(psa_type, values)
    except PSAInputError:
        # A broken/incomplete PSA setup for this parameter shouldn't block
        # calibration over a problem unrelated to this run -- fall back to
        # uniform instead of surfacing a PSA-specific error here.
        return _uniform_prior(low, high)

    def truncated_sample(rng: np.random.Generator) -> float:
        for _ in range(50):
            value = base_sampler(rng)
            if low <= value <= high:
                return float(value)
        return float(min(max(base_sampler(rng), low), high))  # give up cleanly: clip to bounds

    def truncated_log_density(x: float) -> float:
        if not (low <= x <= high):
            return float("-inf")
        return base_log_pdf(x)

    return _Prior(sample=truncated_sample, log_density=truncated_log_density, source=f"psa:{psa_type}")


def _noise_variance_from_sum_sq(sum_sq: np.ndarray, percentile: float = 10.0) -> float:
    """Auto-calibrates the Gaussian likelihood's noise variance (sigma^2)
    for SIR/MCMC from the model itself, rather than assuming unit
    variance on the raw (weighted) residuals.

    A HEOR calibration target rarely comes with an explicit measurement-
    uncertainty figure the way a lab assay would -- Phase 1's `weight`
    column is meant to bring different target *types* (e.g. a probability
    in [0,1] vs a cost in the thousands) onto a comparable scale for a
    least-squares point estimate, not to serve as 1/variance. Treating the
    weighted residual as unit-variance Gaussian noise leaves the
    likelihood far too flat whenever the well-scaled residual's natural
    range across the whole prior is itself well below 1 (the common case
    once targets are properly weighted) -- every candidate then looks
    almost equally (im)plausible and the "posterior" barely differs from
    the prior.

    Instead, sigma^2 is set from the *empirical spread* of sum-of-squared
    -residuals already seen across a batch of blind draws from the same
    priors SIR/MCMC will sample from: `percentile`-th percentile of that
    batch (default the best ~10%). A candidate that fits about as well as
    the best decile of blind draws is then weighted close to 1; a
    typical/bad draw is downweighted sharply -- exactly the discriminating
    behavior Bayesian calibration needs, self-scaled to whatever units and
    weights this particular target table happens to use, with no new
    field required on CalibrationTargetRow/TreeCalibrationTargetRow.
    """

    sigma2 = float(np.percentile(sum_sq, percentile)) if sum_sq.size else 0.0
    return max(sigma2, 1e-9)


def _sir_sample(
    residuals_fn: Callable[[np.ndarray], np.ndarray],
    priors: Sequence[_Prior],
    n_prior_samples: int,
    n_posterior_samples: int,
    random_seed: int | None,
) -> tuple[np.ndarray, dict[str, float], int]:
    """Sampling-Importance-Resampling: draw `n_prior_samples` candidates
    from each parameter's own prior, score each with a Gaussian
    log-likelihood on the (already weighted) residuals `residuals_fn`
    returns -- with the noise variance auto-calibrated from this same
    batch, see _noise_variance_from_sum_sq() -- then resample
    `n_posterior_samples` of them with replacement proportional to their
    (normalized) importance weight. Needs no gradient or proposal step,
    so it's unaffected by a non-smooth residual surface (e.g. Microsim's
    Common-Random-Numbers step function, see
    run_microsim_calibration_local()'s docstring for that story in
    Phase 1) -- every candidate is scored independently.
    """

    rng = np.random.default_rng(random_seed)
    n_params = len(priors)
    prior_draws = np.empty((n_prior_samples, n_params))
    sum_sq = np.empty(n_prior_samples)
    for i in range(n_prior_samples):
        candidate = np.array([prior.sample(rng) for prior in priors])
        prior_draws[i] = candidate
        residuals = residuals_fn(candidate)
        sum_sq[i] = float(np.sum(np.asarray(residuals, dtype=float) ** 2))

    sigma2 = _noise_variance_from_sum_sq(sum_sq)
    log_lik = -0.5 * sum_sq / sigma2

    max_ll = float(np.max(log_lik)) if n_prior_samples else 0.0
    weights = np.exp(log_lik - max_ll)
    weight_sum = float(np.sum(weights))
    if not np.isfinite(weight_sum) or weight_sum <= 0:
        # Every prior draw scored as infeasible (or identically) -- fall
        # back to an unweighted (uniform) resample rather than dividing by
        # zero; the caller sees this as ess == n_prior_samples, which is
        # a red flag worth surfacing to the user, not a crash.
        weights = np.full(n_prior_samples, 1.0 / n_prior_samples)
    else:
        weights = weights / weight_sum
    effective_sample_size = float(1.0 / np.sum(weights**2)) if n_prior_samples else 0.0

    posterior_indices = rng.choice(n_prior_samples, size=n_posterior_samples, replace=True, p=weights)
    posterior_samples = prior_draws[posterior_indices]
    diagnostics = {
        "effective_sample_size": effective_sample_size,
        "n_prior_samples": float(n_prior_samples),
        "noise_variance": sigma2,
    }
    return posterior_samples, diagnostics, n_prior_samples


def _mcmc_sample(
    residuals_fn: Callable[[np.ndarray], np.ndarray],
    priors: Sequence[_Prior],
    start: np.ndarray,
    bounds_low: np.ndarray,
    bounds_high: np.ndarray,
    n_iterations: int,
    burn_in: int,
    step_scale: float,
    random_seed: int | None,
    sigma2: float,
) -> tuple[np.ndarray, dict[str, float], int]:
    """Random-walk Metropolis-Hastings: propose a Gaussian step from the
    current point (SD = `step_scale` * each parameter's own [low, high]
    range), accept/reject by the standard MH ratio on log-posterior
    (log-prior + Gaussian log-likelihood on the weighted residuals scaled
    by `sigma2` -- see _noise_variance_from_sum_sq(), estimated by the
    caller from a pilot batch of prior draws before the chain starts),
    keep every post-burn-in draw as the chain. A candidate outside its
    parameter's bounds has log-prior -inf (see _Prior/_build_prior) and is
    always rejected, so the chain never leaves [low, high].
    """

    rng = np.random.default_rng(random_seed)
    n_params = len(priors)
    step_sd = step_scale * (bounds_high - bounds_low)
    step_sd = np.where(step_sd > 0, step_sd, 1.0)  # a degenerate (zero-width) bound would zero every proposal

    def log_posterior(candidate: np.ndarray) -> float:
        log_prior = 0.0
        for j in range(n_params):
            log_prior += priors[j].log_density(float(candidate[j]))
            if not np.isfinite(log_prior):
                return float("-inf")
        residuals = residuals_fn(candidate)
        sum_sq = float(np.sum(np.asarray(residuals, dtype=float) ** 2))
        return log_prior - 0.5 * sum_sq / sigma2

    current = np.array(start, dtype=float)
    current_log_post = log_posterior(current)
    samples = np.empty((n_iterations, n_params))
    n_accepted = 0
    n_evaluations = 1

    for i in range(n_iterations):
        proposal = current + rng.normal(0.0, step_sd)
        proposal_log_post = log_posterior(proposal)
        n_evaluations += 1
        if not np.isfinite(current_log_post) and not np.isfinite(proposal_log_post):
            accept = False
        elif not np.isfinite(current_log_post):
            accept = True
        elif not np.isfinite(proposal_log_post):
            accept = False
        else:
            accept = bool(np.log(rng.uniform()) < (proposal_log_post - current_log_post))
        if accept:
            current = proposal
            current_log_post = proposal_log_post
            n_accepted += 1
        samples[i] = current

    kept = samples[burn_in:] if burn_in < n_iterations else samples[-1:]
    diagnostics = {
        "acceptance_rate": float(n_accepted / n_iterations) if n_iterations else 0.0,
        "n_iterations": float(n_iterations),
        "burn_in": float(burn_in),
        "noise_variance": sigma2,
    }
    return kept, diagnostics, n_evaluations


def _summarize_posterior(
    names: Sequence[str], samples: np.ndarray
) -> tuple[dict[str, float], dict[str, float], dict[str, tuple[float, float]]]:
    mean = {name: float(np.mean(samples[:, j])) for j, name in enumerate(names)}
    sd = {name: float(np.std(samples[:, j], ddof=1)) if samples.shape[0] > 1 else 0.0 for j, name in enumerate(names)}
    ci95 = {
        name: (float(np.percentile(samples[:, j], 2.5)), float(np.percentile(samples[:, j], 97.5)))
        for j, name in enumerate(names)
    }
    return mean, sd, ci95


def _run_bayesian_calibration(
    method: str,
    document: Mapping[str, Any],
    parameters: Sequence[CalibrationParameter],
    residuals_fn: Callable[[np.ndarray], np.ndarray],
    predict_fn: Callable[[Sequence[float]], tuple[list[dict[str, float]], list[float]]],
    *,
    n_prior_samples: int = 500,
    n_posterior_samples: int = 200,
    n_iterations: int = 1000,
    burn_in: int = 200,
    step_scale: float = 0.05,
    n_pilot_samples: int = 500,
    random_seed: int | None = None,
) -> BayesianCalibrationResult:
    """Shared core for run_markov_calibration_bayesian() /
    run_microsim_calibration_bayesian() / run_tree_calibration_bayesian():
    the only things that differ between the three model types are how
    `residuals_fn`/`predict_fn` are built (each model's own
    `_markov_residuals`/`_microsim_residuals`/`_tree_residuals` and
    `_predicted_trace_and_values`/`_predicted_tree_values`) -- everything
    about SIR/MCMC themselves lives here once.

    Unlike Phase 1 (which deliberately triplicated Local Optimization/
    Grid/Random Search per model to keep each version simple and
    independently easy to verify), SIR and MCMC are complex enough that a
    shared core outweighs that benefit: tripling this code would triple
    the surface area for a subtle bug in the sampler itself, which is a
    worse risk than the three model wrappers depending on one shared
    implementation.
    """

    result = BayesianCalibrationResult(method=method)
    if method not in ("bayesian_sir", "bayesian_mcmc"):
        result.errors.append(f"Unknown Bayesian calibration method '{method}'")
        return result
    if not parameters:
        result.errors.append("Select at least one parameter to calibrate")
        return result

    names = [p.name for p in parameters]
    priors = [_build_prior(p, document) for p in parameters]
    result.parameter_names = names
    result.prior_source = {p.name: prior.source for p, prior in zip(parameters, priors)}

    if method == "bayesian_sir":
        samples, diagnostics, n_evaluations = _sir_sample(
            residuals_fn, priors, n_prior_samples, n_posterior_samples, random_seed
        )
        # A near-1 effective sample size means almost every prior draw
        # carried the same tiny weight -- the target data barely
        # discriminated between candidates (too loose a target, or bounds
        # too tight already) -- still a valid run, just a weak signal.
        result.converged = diagnostics["effective_sample_size"] >= max(10.0, 0.02 * n_prior_samples)
        if not result.converged:
            result.errors.append(
                f"Low effective sample size ({diagnostics['effective_sample_size']:.1f} of "
                f"{n_prior_samples} prior draws) -- the posterior may be poorly resolved; try more "
                "prior samples or check the target data/bounds."
            )
    else:
        parameter_arrays_low = np.array([p.low for p in parameters])
        parameter_arrays_high = np.array([p.high for p in parameters])
        start = np.array([p.starting_value() for p in parameters])
        # MCMC has no big prior batch of its own (unlike SIR) to read a
        # noise scale off of, so draw a small, separate pilot batch from
        # the same priors purely to calibrate sigma^2 before the chain
        # starts -- see _noise_variance_from_sum_sq(). Reuses random_seed
        # so a fixed seed still reproduces the whole run deterministically.
        pilot_rng = np.random.default_rng(random_seed)
        pilot_sum_sq = np.empty(max(n_pilot_samples, 1))
        for i in range(pilot_sum_sq.size):
            candidate = np.array([prior.sample(pilot_rng) for prior in priors])
            residuals = residuals_fn(candidate)
            pilot_sum_sq[i] = float(np.sum(np.asarray(residuals, dtype=float) ** 2))
        sigma2 = _noise_variance_from_sum_sq(pilot_sum_sq)
        samples, diagnostics, n_evaluations = _mcmc_sample(
            residuals_fn, priors, start, parameter_arrays_low, parameter_arrays_high,
            n_iterations, burn_in, step_scale, random_seed, sigma2,
        )
        n_evaluations += pilot_sum_sq.size
        diagnostics["n_pilot_samples"] = float(pilot_sum_sq.size)
        # Standard random-walk MH rule of thumb: healthy acceptance is
        # roughly 15-50%; well outside that, the chain is barely moving
        # (step too large) or moving without discriminating (step too
        # small / posterior too flat) -- still returned, just flagged.
        result.converged = 0.05 <= diagnostics["acceptance_rate"] <= 0.80
        if not result.converged:
            result.errors.append(
                f"Acceptance rate {diagnostics['acceptance_rate']:.1%} is outside the healthy 5-80% range -- "
                "try a different step size, or treat this posterior with caution."
            )

    result.posterior_samples = samples
    result.n_evaluations = n_evaluations
    result.diagnostics = diagnostics
    if samples.shape[0] > 0:
        result.posterior_mean, result.posterior_sd, result.credible_interval_95 = _summarize_posterior(names, samples)
        trace, predicted = predict_fn(list(result.posterior_mean.values()))
        result.predicted_trace = trace
        result.predicted_at_targets = predicted
    return result


def run_markov_calibration_bayesian(
    document: Mapping[str, Any],
    strategy_id: str,
    parameters: Sequence[CalibrationParameter],
    targets: Sequence[CalibrationTargetRow],
    method: str = "bayesian_sir",
    allow_negative_cost: bool = False,
    **kwargs: Any,
) -> BayesianCalibrationResult:
    """Bayesian Calibration for Markov -- "bayesian_sir" (Sampling-
    Importance-Resampling) or "bayesian_mcmc" (Metropolis-Hastings). See
    _run_bayesian_calibration() for what the two methods share, and
    run_markov_calibration()/run_markov_calibration_local() for the
    Phase-1 single-best-fit counterpart this complements rather than
    replaces."""

    if not targets:
        result = BayesianCalibrationResult(method=method)
        result.errors.append("Enter at least one target data row")
        return result
    if _find_strategy(document, strategy_id) is None:
        result = BayesianCalibrationResult(method=method)
        result.errors.append(f"Unknown strategy '{strategy_id}'")
        return result

    names = [p.name for p in parameters]

    def residuals_fn(candidate: np.ndarray) -> np.ndarray:
        return _markov_residuals(
            candidate, document=document, strategy_id=strategy_id, parameter_names=names,
            targets=targets, allow_negative_cost=allow_negative_cost,
        )

    def predict_fn(values: Sequence[float]) -> tuple[list[dict[str, float]], list[float]]:
        return _predicted_trace_and_values(document, strategy_id, names, values, targets, allow_negative_cost)

    return _run_bayesian_calibration(method, document, parameters, residuals_fn, predict_fn, **kwargs)


def run_microsim_calibration_bayesian(
    document: Mapping[str, Any],
    strategy_id: str,
    parameters: Sequence[CalibrationParameter],
    targets: Sequence[CalibrationTargetRow],
    method: str = "bayesian_sir",
    allow_negative_cost: bool = False,
    **kwargs: Any,
) -> BayesianCalibrationResult:
    """Microsim analogue of run_markov_calibration_bayesian() -- same
    shared SIR/MCMC core, scored through the Microsim engine instead.
    Each prior draw / MCMC step is a full Microsim run, so this is far
    slower than the Markov version; the UI should default to much
    smaller n_prior_samples/n_iterations for Microsim than for Markov/
    Tree (see the Model Calibration section of the DECIDE manual)."""

    if not targets:
        result = BayesianCalibrationResult(method=method)
        result.errors.append("Enter at least one target data row")
        return result
    if _find_strategy(document, strategy_id) is None:
        result = BayesianCalibrationResult(method=method)
        result.errors.append(f"Unknown strategy '{strategy_id}'")
        return result

    names = [p.name for p in parameters]

    def residuals_fn(candidate: np.ndarray) -> np.ndarray:
        return _microsim_residuals(
            candidate, document=document, strategy_id=strategy_id, parameter_names=names,
            targets=targets, allow_negative_cost=allow_negative_cost,
        )

    def predict_fn(values: Sequence[float]) -> tuple[list[dict[str, float]], list[float]]:
        # Microsim has no _predicted_trace_and_values of its own in Phase 1
        # (see calibration.py) -- re-run once at the posterior mean the
        # same way _microsim_residuals itself would, via the shared
        # _tree/_microsim predicted-value pattern.
        overrides = {name: float(value) for name, value in zip(names, values)}
        scenario = copy.deepcopy(dict(document))
        if _apply_overrides(scenario, overrides):
            return [], []
        strategy = _find_strategy(scenario, strategy_id)
        if strategy is None:
            return [], []
        try:
            parameters_, variables, distributions, tables = _build_inputs(scenario)
            model = resolve_microsim(_microsim_merged_document(scenario, strategy))
        except (KeyError, ValueError):
            return [], []
        cycles_per_year = float(scenario.get("cycles_per_year", 1.0)) or 1.0
        cost_discount = _per_cycle_rate(
            float(scenario.get("discount_cost_value", 0.0)), str(scenario.get("discount_cost_unit", "per cycle")), cycles_per_year
        )
        effect_discount = _per_cycle_rate(
            float(scenario.get("discount_effect_value", 0.0)), str(scenario.get("discount_effect_unit", "per cycle")), cycles_per_year
        )
        result = run_microsim(
            model, parameters_, discount_cost_per_cycle=cost_discount, discount_effect_per_cycle=effect_discount,
            allow_negative_cost=allow_negative_cost, variables=variables, distributions=distributions, tables=tables,
        )
        if result.errors:
            return [], []
        predicted_at_targets = [
            result.trace[t.cycle].get(t.state_id, float("nan")) if t.cycle < len(result.trace) else float("nan")
            for t in targets
        ]
        return result.trace, predicted_at_targets

    return _run_bayesian_calibration(method, document, parameters, residuals_fn, predict_fn, **kwargs)


def run_tree_calibration_bayesian(
    document: Mapping[str, Any],
    strategy_id: str,
    parameters: Sequence[CalibrationParameter],
    targets: Sequence[TreeCalibrationTargetRow],
    method: str = "bayesian_sir",
    allow_negative_cost: bool = False,
    **kwargs: Any,
) -> BayesianCalibrationResult:
    """Tree analogue of run_markov_calibration_bayesian() -- same shared
    SIR/MCMC core, scored via _tree_residuals()/_predicted_tree_values()
    instead."""

    if not targets:
        result = BayesianCalibrationResult(method=method)
        result.errors.append("Enter at least one target data row")
        return result
    if strategy_id not in _tree_strategy_ids(document):
        result = BayesianCalibrationResult(method=method)
        result.errors.append(f"Unknown strategy '{strategy_id}'")
        return result

    names = [p.name for p in parameters]

    def residuals_fn(candidate: np.ndarray) -> np.ndarray:
        return _tree_residuals(
            candidate, document=document, strategy_id=strategy_id, parameter_names=names,
            targets=targets, allow_negative_cost=allow_negative_cost,
        )

    def predict_fn(values: Sequence[float]) -> tuple[list[dict[str, float]], list[float]]:
        predicted = _predicted_tree_values(document, strategy_id, names, values, targets, allow_negative_cost)
        return [], predicted

    return _run_bayesian_calibration(method, document, parameters, residuals_fn, predict_fn, **kwargs)


def apply_posterior_to_psa(document: dict[str, Any], parameter_name: str, samples: np.ndarray) -> str:
    """Write a Bayesian Calibration posterior back into the model's PSA
    setup (document["psa_parameters"]) for `parameter_name`, so the next
    PSA run draws from the calibrated uncertainty instead of whatever was
    configured (or nothing) before. Summarizes the (possibly irregular)
    empirical posterior as a fitted parametric PSA distribution via
    method-of-moments on its mean/SD -- reusing psa.py's own Beta/Normal
    machinery unchanged rather than adding a new "raw empirical samples"
    PSA distribution type -- so this never needs psa.py or its Setup PSA
    UI to change at all. Beta is tried first when the posterior mean sits
    strictly inside (0, 1) (the common case for a calibrated probability),
    since a probability's uncertainty is almost always better represented
    bounded to [0, 1] than by an unbounded Normal; Normal is the fallback
    whenever Beta's variance constraint (variance < mean*(1-mean)) can't
    be met, or the mean falls outside (0, 1) altogether (e.g. a calibrated
    cost or a rate that can exceed 1).

    Returns the psa_type actually written ("Beta" or "Normal"), so the
    caller can tell the user which one was used.
    """

    mean = float(np.mean(samples))
    sd = float(np.std(samples, ddof=1)) if samples.shape[0] > 1 else 0.0
    psa_type = "Normal"
    values: dict[str, Any] = {"mean": mean, "se": max(sd, 1e-9)}
    if 0.0 < mean < 1.0 and sd > 0:
        try:
            _beta_from_mean_se(mean, sd)  # raises PSAInputError if the variance constraint fails
            psa_type = "Beta"
        except PSAInputError:
            pass

    psa_parameters = document.setdefault("psa_parameters", [])
    target_id = f"param:{parameter_name}"
    row = next((r for r in psa_parameters if str(r.get("target_id", "")) == target_id), None)
    if row is None:
        row = {"target_id": target_id}
        psa_parameters.append(row)
    row["psa_type"] = psa_type
    row["values"] = {key: str(value) for key, value in values.items()}
    return psa_type
