"""Discrete Event Simulation (DES) engine.

Per patient (own RNG sub-stream, SeedSequence.spawn -- same Common Random
Numbers convention as microsim_engine):

1. Resample `per_patient` Distributions; set every Tracker to its initial
   value (t = 0).
2. Pick the starting branch of the DES node (probabilities), accrue that
   path's event rewards and the starting Time node's STARTUP rewards.
3. At a Time node entered at time t: sample a time-to-event for every
   competing event (branch); the shortest wins (TreeAge 43.4).
   * If t + tte >= time horizon: accrue the TIME rewards up to the horizon
     and stop (censored -- no event rewards).
   * Otherwise accrue time rewards over [t, t + tte] (per-time-unit value x
     time spent, continuously discounted), move the clock, then walk the
     winning event's path: branch event rewards + tracker updates, any
     Chance nodes (sampled), and the Terminal's event reward.
4. A Terminal with `jump_to` sends the patient to that Time node (step 3);
   without it the patient is absorbed (e.g. Dead) and stops.

Rewards follow TreeAge 43.5 (time / event / startup rewards) and 43.13
(discounting): event rewards at time t get (1+r)^-t, time rewards over
[a, b] get the integral of (1+r)^-s ds.

Keywords available in every expression (TreeAge 43.2, DES keywords):
  _time         current patient time (entry time at a Time node; the event
                time while walking an event path)
  _event_time   time of the last event
  _prior_time   time before the last event
  _total_cost / _total_effect   undiscounted value accumulated so far
                (TreeAge _total_rwdx[1] / [2])
  _stage_cost / _stage_effect   undiscounted value accumulated since the
                patient last entered a Time node (TreeAge _stage_rwdx)
  _time_in_state  time spent so far in the current Time node
plus every Tracker by name and every sampled per_model/per_patient
Distribution by name.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import Any

import numpy as np

from .distributions import conditional_event_time
from .des_models import (
    NAMED_DISTRIBUTION_CLOSED_FORMS,
    DESBranch,
    DESChance,
    DESEventRecord,
    DESLogic,
    ELSE_MARKERS,
    DESModel,
    DESResult,
    DESTerminal,
    TimeNode,
    TimeToEvent,
)
from .des_sampling import (
    TableCDF,
    build_table_cdf,
    discount_factor,
    discounted_duration,
    sample_parametric,
    survival,
    table_to_time,
)
from .des_validation import validate_des
from .markov_engine import _resolve, _resolve_probability_group
from .microsim_engine import _resample_distributions
from .models import Param, ValidationError

_PARAMETRIC = {"exponential", "weibull", "log_normal", "log_logistic", "gompertz", "generalized_gamma"}


# Pseudo-state / event label for patients stopped by the DES node's
# termination condition (trace, event counts, patient timeline).
TERMINATED = "Terminated"


class _Patient:
    """Mutable per-patient state + the expression context built from it."""

    def __init__(self, base_context: Mapping[str, Any]) -> None:
        self.base = dict(base_context)
        self.trackers: dict[str, float] = {}
        self.time = 0.0
        self.event_time = 0.0
        self.prior_time = 0.0
        self.state_entry_time = 0.0
        self.total_cost = 0.0
        self.total_effect = 0.0
        self.stage_cost = 0.0
        self.stage_effect = 0.0
        self.disc_cost = 0.0
        self.disc_effect = 0.0
        self.sampled_once: dict[tuple, float] = {}
        self.rng: Any = None
        self.extra_raw: dict[str, float] = {}
        self.extra_disc: dict[str, float] = {}
        self.extra_stage: dict[str, float] = {}
        # keyword name -> ("total"|"stage", outcome id) for every Cost/Effect
        # outcome: _total_<id>, _stage_<id>, and _total_<symbol>/_stage_<symbol>
        # when the symbol is a plain name (e.g. _total_QALY).
        self.outcome_keywords: dict[str, tuple[str, str]] = {}

    def outcome_value(self, which: str, outcome_id: str) -> float:
        if outcome_id == "cost":
            return self.total_cost if which == "total" else self.stage_cost
        if outcome_id == "effect":
            return self.total_effect if which == "total" else self.stage_effect
        source = self.extra_raw if which == "total" else self.extra_stage
        return source.get(outcome_id, 0.0)

    def _outcome_keyword_values(self) -> dict[str, float]:
        return {name: self.outcome_value(which, outcome_id)
                for name, (which, outcome_id) in self.outcome_keywords.items()}

    def context(self) -> dict[str, Any]:
        return {
            **self.base,
            **self.trackers,
            "_time": self.time,
            "_event_time": self.event_time,
            "_prior_time": self.prior_time,
            "_total_cost": self.total_cost,
            "_total_effect": self.total_effect,
            "_stage_cost": self.stage_cost,
            "_stage_effect": self.stage_effect,
            "_time_in_state": self.time - self.state_entry_time,
            **self._outcome_keyword_values(),
            # Compatibility with Tree/Markov formulas that reference these.
            "_cycle": 0,
            "_tenure": 1,
            # The patient's own random stream: DistSample/DistToTime/
            # TableToTime inside formulas draw from it (reproducible, CRN).
            "__rng__": self.rng,
        }


def _fmt_values(values: Mapping[str, Any]) -> str:
    return ", ".join(f"{name} = {value:.6g}" if isinstance(value, (int, float)) else f"{name} = {value}"
                     for name, value in values.items())


_PLAIN_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


def outcome_keywords(outcome_symbols: Mapping[str, str], extra_ids: Any = ()) -> dict[str, tuple[str, str]]:
    """TreeAge _total_rwdx[i] / _stage_rwdx[i] for every Cost/Effect
    outcome, by id and by symbol: _total_effect_2 and _total_QALY (symbol
    "QALY"), _stage_QALY ... The primary pair keeps _total_cost/_total_effect
    and also gets its symbol (_total_E). Symbols that are not plain names
    (e.g. "$") only have the id form."""

    keywords: dict[str, tuple[str, str]] = {}
    for outcome_id in list(outcome_symbols) + [i for i in extra_ids if i not in outcome_symbols]:
        names = [outcome_id]
        symbol = str(outcome_symbols.get(outcome_id) or "").strip()
        if symbol and _PLAIN_NAME.match(symbol):
            names.append(symbol)
        for name in names:
            if not _PLAIN_NAME.match(name):
                continue
            keywords.setdefault(f"_total_{name}", ("total", outcome_id))
            keywords.setdefault(f"_stage_{name}", ("stage", outcome_id))
    return keywords


def run_des(
    model: DESModel,
    parameters: Mapping[str, Param] | None = None,
    cost_discount_rate: float = 0.0,
    effect_discount_rate: float = 0.0,
    variables: Mapping[str, Any] | None = None,
    distributions: Mapping[str, Mapping[str, Any]] | None = None,
    tables: Mapping[str, Mapping[str, Any]] | None = None,
    psa_tier_values: Mapping[str, float] | None = None,
    outcome_kinds: Mapping[str, str] | None = None,
    trace_patient: int | None = None,
) -> DESResult:
    """Simulate `model.des.n_patients` patients; discount rates are annual
    (per time unit). `outcome_kinds` ({outcome id: "cost"|"effect"}) lists
    EXTRA Cost/Effect outcomes (Outcomes table "+") to accumulate alongside
    the primary pair, each discounted with its kind's rate.

    `trace_patient` (1-based) records a step-by-step Calculation trace of
    that patient in DESResult.calculation_trace (TreeAge Calculation Trace
    Console): characteristics, every sampled time-to-event, the winning
    event, time/event rewards, Chance/Logic choices, tracker updates.
    Patient k uses the same random stream whatever N is, so a run with
    n_patients = k reproduces patient k of the full run."""

    parameters = parameters or {}
    tables = tables or {}
    errors = validate_des(model, parameters, tables, distributions)
    if errors:
        return DESResult(errors=errors)

    des = model.des
    nodes = model.nodes

    def resolve(value: Any, ctx: Mapping[str, Any]) -> float:
        return _resolve(value, parameters, ctx, distributions, tables)

    base_variables = dict(variables or {})
    # Only Distributions with an explicit "resample" setting are pre-sampled
    # as per-model/per-patient VALUES (e.g. a patient's start age). The rest
    # -- in particular every time-to-event distribution, in this or another
    # strategy's DES -- are used by name (DistToTime) and never pre-sampled.
    tte_names = {
        str(branch.time_to_event.params.get("name"))
        for node in nodes.values() for branch in getattr(node, "branches", [])
        if branch.time_to_event is not None and branch.time_to_event.distribution == "named"
    }
    value_distributions = {
        name: d for name, d in (distributions or {}).items()
        if name not in tte_names and (d or {}).get("resample")
    }
    model_rng = np.random.default_rng(des.random_seed)
    model_tier = _resample_distributions(
        value_distributions, "per_model", model_rng, parameters,
        {**base_variables, "_time": 0.0, "_cycle": 0, "_tenure": 1}, tables,
        fold_psa_into_model=psa_tier_values is None,
    )
    run_context = {**base_variables, **model_tier, **(psa_tier_values or {})}

    try:
        horizon = resolve(des.time_horizon, {**run_context, "_time": 0.0})
    except ValueError as exc:
        return DESResult(errors=[ValidationError("invalid_time_horizon", str(exc), des.id, "time_horizon")])
    if not horizon > 0:
        return DESResult(errors=[ValidationError("invalid_time_horizon", "Time horizon must be > 0", des.id)])

    table_cache: dict[tuple, TableCDF] = {}

    def table_cdf(tte: TimeToEvent) -> TableCDF:
        key = (str(tte.params["table"]), str(tte.params["index_column"]), str(tte.params["value_column"]),
               tte.prob_type, tte.interpolate)
        if key not in table_cache:
            table = tables[key[0]]
            table_cache[key] = build_table_cdf(table.get("rows", []), key[1], key[2], key[3], key[4])
        return table_cache[key]

    def sample_tte(branch: DESBranch, patient: _Patient, rng: np.random.Generator) -> float:
        """Time from NOW until this event (>= 0, may be inf)."""

        tte = branch.time_to_event
        assert tte is not None
        ctx = patient.context()
        dist = tte.distribution
        if dist == "fixed":
            return max(0.0, resolve(tte.params["time"], ctx))
        if dist == "expression":
            # "__rng__": the patient's own random stream, so DistToTime(...)
            # inside the expression is reproducible (common random numbers).
            return max(0.0, resolve(tte.params["value"], {**ctx, "__rng__": rng}))

        if dist == "named":
            return _sample_named(tte, patient, rng, ctx)

        forward = tte.clock == "forward"
        offset = resolve(tte.time_offset, ctx) if forward else 0.0
        now_axis = patient.time + offset if forward else 0.0

        once_key = None
        if tte.sample_once:
            once_key = (branch.label, dist, tuple(sorted((k, str(v)) for k, v in tte.params.items())),
                        str(tte.time_offset), tte.prob_type, tte.interpolate)
            if once_key in patient.sampled_once:
                return max(0.0, patient.sampled_once[once_key] - now_axis)

        uniform = 1.0 - rng.random()  # (0, 1]
        if dist == "table_to_time":
            if not forward:
                # Reset clock: the table is indexed by time since entering the state.
                return max(0.0, table_cdf(tte).sample(uniform))
            if once_key is None:
                return table_to_time(table_cdf(tte), now_axis, uniform)
            absolute = now_axis + table_to_time(table_cdf(tte), now_axis, uniform)
        else:
            params = {name: resolve(value, ctx) for name, value in tte.params.items()}
            absolute = sample_parametric(dist, params, uniform, lower=now_axis if forward else 0.0)
        if once_key is not None:
            patient.sampled_once[once_key] = absolute
        return max(0.0, absolute - now_axis)

    def _sample_named(tte: TimeToEvent, patient: _Patient, rng: np.random.Generator, ctx: Mapping[str, Any]) -> float:
        """Time-to-event from a named Distribution (Distributions section).
        Closed-form types reuse sample_parametric(); other continuous types
        use scipy's sf/isf through the same conditional_event_time()."""

        name = str(tte.params["name"])
        definition = (distributions or {})[name]
        forward = tte.clock == "forward"
        offset = resolve(tte.time_offset, ctx) if forward else 0.0
        now_axis = patient.time + offset if forward else 0.0
        once_key = ("named", name, str(tte.time_offset)) if tte.sample_once else None
        if once_key is not None and once_key in patient.sampled_once:
            return max(0.0, patient.sampled_once[once_key] - now_axis)
        uniform = 1.0 - rng.random()
        raw_params = definition.get("parameters") or {}
        closed = NAMED_DISTRIBUTION_CLOSED_FORMS.get(str(definition.get("type", "")))
        if closed is not None:
            key, names = closed
            params = {param: resolve(raw_params[param], ctx) for param in names}
            absolute = sample_parametric(key, params, uniform, lower=now_axis if forward else 0.0)
        else:
            from .distributions import _distribution

            frozen = _distribution(definition, lambda value: resolve(value, ctx))
            lower = now_axis if forward else 0.0
            absolute = lower + conditional_event_time(
                lambda x: float(frozen.sf(x)), lambda u: float(frozen.isf(u)), lower, uniform,
            )
        if once_key is not None:
            patient.sampled_once[once_key] = absolute
        return max(0.0, absolute - now_axis)

    def pick_branch(branches: list[DESBranch], ctx: Mapping[str, Any], rng: np.random.Generator) -> DESBranch:
        probabilities = _resolve_probability_group(
            [(str(i), branch.probability) for i, branch in enumerate(branches)],
            parameters, ctx, distributions, tables,
        )
        weights = [max(0.0, probabilities[str(i)]) for i in range(len(branches))]
        total = sum(weights)
        if total <= 0:
            raise ValueError("Branch probabilities sum to 0")
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Branch probabilities sum to {total:g}, not 1")
        draw = rng.random() * total
        running = 0.0
        for branch, weight in zip(branches, weights):
            running += weight
            if draw < running:
                return branch
        return branches[-1]

    extra_kinds = dict(outcome_kinds or {})
    keyword_map = outcome_keywords(model.outcome_symbols, extra_kinds)
    calculation_trace: list[dict[str, Any]] = []
    tracing = {"patient": None}  # the _Patient being traced, else None

    def note(patient: _Patient, step: str, detail: str, **values: Any) -> None:
        if tracing["patient"] is not patient:
            return
        calculation_trace.append({
            "time": patient.time, "step": step, "detail": detail,
            "cost (disc.)": patient.disc_cost, "effect (disc.)": patient.disc_effect,
            **values,
        })

    def extra_rate(outcome_id: str) -> float:
        return cost_discount_rate if extra_kinds.get(outcome_id) == "cost" else effect_discount_rate

    def add_extra_event(patient: _Patient, values: Mapping[str, Any], ctx: Mapping[str, Any]) -> None:
        for outcome_id in extra_kinds:
            if outcome_id in values:
                value = resolve(values[outcome_id], ctx)
                patient.extra_raw[outcome_id] = patient.extra_raw.get(outcome_id, 0.0) + value
                patient.extra_stage[outcome_id] = patient.extra_stage.get(outcome_id, 0.0) + value
                patient.extra_disc[outcome_id] = patient.extra_disc.get(outcome_id, 0.0) + value * discount_factor(
                    extra_rate(outcome_id), patient.time)

    def add_event_reward(patient: _Patient, cost: float, effect: float) -> None:
        patient.total_cost += cost
        patient.total_effect += effect
        patient.stage_cost += cost
        patient.stage_effect += effect
        patient.disc_cost += cost * discount_factor(cost_discount_rate, patient.time)
        patient.disc_effect += effect * discount_factor(effect_discount_rate, patient.time)

    def traverse(branch: DESBranch, patient: _Patient) -> None:
        """Event rewards + tracker updates of one branch."""

        ctx = patient.context()
        cost = resolve(branch.cost_increment, ctx) if branch.cost_increment is not None else 0.0
        effect = resolve(branch.effect_increment, ctx) if branch.effect_increment is not None else 0.0
        new_values: dict[str, Any] = {}
        if branch.tracker_updates:
            # All updates see the same (pre-update) context, like Microsim.
            new_values = {name: resolve(formula, ctx) for name, formula in branch.tracker_updates.items()}
            patient.trackers.update(new_values)
        add_event_reward(patient, cost, effect)
        if branch.extra_outcomes:
            add_extra_event(patient, branch.extra_outcomes, ctx)
        if cost or effect or new_values or branch.extra_outcomes:
            note(patient, "Branch rewards", f"'{branch.label}': cost +{cost:g}, effect +{effect:g}"
                 + (f"; trackers {_fmt_values(new_values)}" if new_values else ""))

    def apply_tracker_updates(patient: _Patient, updates: Mapping[str, Any]) -> None:
        if updates:
            ctx = patient.context()
            new_values = {name: resolve(formula, ctx) for name, formula in updates.items()}
            patient.trackers.update(new_values)
            note(patient, "Tracker update", _fmt_values(new_values))

    def pick_logic_branch(node: DESLogic, patient: _Patient) -> DESBranch:
        ctx = patient.context()
        default: DESBranch | None = None
        for branch in node.branches:
            condition = branch.condition
            if isinstance(condition, str) and condition.strip().lower() in ELSE_MARKERS:
                default = default or branch
                continue
            if bool(resolve(condition, ctx)):
                return branch
        if default is None:
            raise ValueError(f"Logic node '{node.id}': no condition is true and there is no 'else' (#) branch")
        return default

    def walk_path(node_id: str, patient: _Patient, rng: np.random.Generator) -> tuple[str | None, str | None]:
        """From a node after an event (or a DES start branch), through any
        Chance nodes, to a Terminal. Returns the Time node id to continue
        at (None when absorbed) and the Terminal id reached (None when the
        path led straight to a Time node -- a DES start branch may)."""

        node = nodes[node_id]
        while isinstance(node, (DESChance, DESLogic)):
            if isinstance(node, DESLogic):
                branch = pick_logic_branch(node, patient)
                note(patient, "Logic", f"{node.id} -> '{branch.label}'"
                     + (f" (condition {branch.condition})" if branch.condition is not None else ""))
            else:
                branch = pick_branch(node.branches, patient.context(), rng)
                note(patient, "Chance", f"{node.id} -> '{branch.label}' (p = {branch.probability})")
            traverse(branch, patient)
            node = nodes[branch.child_id]
        if isinstance(node, TimeNode):
            return node.id, None
        assert isinstance(node, DESTerminal)
        ctx = patient.context()
        terminal_cost, terminal_effect = resolve(node.cost, ctx), resolve(node.effect, ctx)
        add_event_reward(patient, terminal_cost, terminal_effect)
        note(patient, "Terminal", f"{node.id}: cost +{terminal_cost:g}, effect +{terminal_effect:g}; "
             + (f"jump to {node.jump_to}" if node.jump_to else "absorbed"))
        apply_tracker_updates(patient, node.tracker_updates)
        if node.extra_outcomes:
            add_extra_event(patient, node.extra_outcomes, ctx)
        return node.jump_to, node.id

    n_patients = des.n_patients
    step = des.trace_step
    n_grid = int(math.floor(horizon / step + 1e-9)) + 1
    trace_times = [min(i * step, horizon) for i in range(n_grid)]
    time_node_ids = [node_id for node_id, node in nodes.items() if isinstance(node, TimeNode)]
    absorbing_ids = [node_id for node_id, node in nodes.items()
                     if isinstance(node, DESTerminal) and node.jump_to is None]
    stop_rule = des.termination_condition
    trace_states = time_node_ids + absorbing_ids + ([TERMINATED] if stop_rule is not None else [])
    occupancy = [dict.fromkeys(trace_states, 0.0) for _ in range(n_grid)]
    event_count_sum: dict[str, float] = {
        f"{node_id}:{branch.label}": 0.0
        for node_id, node in nodes.items() if isinstance(node, TimeNode) for branch in node.branches
    }
    time_in_state_sum = dict.fromkeys(time_node_ids, 0.0)
    patient_costs: list[float] = []
    patient_effects: list[float] = []
    raw_cost_sum = raw_effect_sum = simulated_time_sum = 0.0
    tracker_finals: dict[str, list[float]] = {t.name: [] for t in des.trackers if t.include_in_output}
    event_logs: list[list[DESEventRecord]] = []
    patient_rows: list[dict[str, Any]] = []
    exposures: dict[str, list[tuple[float, float, bool]]] = {}
    log_limit = min(des.event_log_sample_size, n_patients)

    def stop_now(patient: _Patient, at_time: float,
                 rates: tuple[float, float, Mapping[str, float]] | None = None) -> bool:
        """Termination condition evaluated at _time = at_time. Between two
        events only _time and the time rewards change, so with `rates`
        (cost/effect/extra rates of the current state, accruing since
        patient.time) the _total_*/_stage_* keywords are advanced too."""

        saved = (patient.time, patient.total_cost, patient.total_effect, patient.stage_cost,
                 patient.stage_effect, dict(patient.extra_raw), dict(patient.extra_stage))
        elapsed = at_time - patient.time
        patient.time = at_time
        if rates is not None and elapsed > 0:
            cost_rate, effect_rate, extra = rates
            patient.total_cost += cost_rate * elapsed
            patient.stage_cost += cost_rate * elapsed
            patient.total_effect += effect_rate * elapsed
            patient.stage_effect += effect_rate * elapsed
            for outcome_id, value in extra.items():
                patient.extra_raw[outcome_id] = patient.extra_raw.get(outcome_id, 0.0) + value * elapsed
                patient.extra_stage[outcome_id] = patient.extra_stage.get(outcome_id, 0.0) + value * elapsed
        try:
            return bool(resolve(stop_rule, patient.context()))
        finally:
            (patient.time, patient.total_cost, patient.total_effect, patient.stage_cost,
             patient.stage_effect, patient.extra_raw, patient.extra_stage) = saved

    def stop_time_within(patient: _Patient, start: float, end: float,
                         rates: tuple[float, float, Mapping[str, float]]) -> float | None:
        """First time in (start, end] at which the termination condition
        becomes true (it is false at `start`), located by bisection to
        1e-9; None if it is still false at `end`."""

        if not stop_now(patient, end, rates):
            return None
        low, high = start, end
        for _ in range(200):
            if high - low <= 1e-9 * max(1.0, abs(high)):
                break
            middle = 0.5 * (low + high)
            if stop_now(patient, middle, rates):
                high = middle
            else:
                low = middle
        return high

    def mark_occupancy(state: str, start: float, end: float) -> None:
        """Occupancy on grid points g with start <= g < end (end=inf for the
        last segment, so the horizon point is covered)."""

        first = max(0, math.ceil(start / step - 1e-9))
        last = n_grid - 1 if math.isinf(end) else min(n_grid - 1, math.ceil(end / step - 1e-9) - 1)
        for g in range(first, last + 1):
            occupancy[g][state] += 1.0

    extra_disc_sum = dict.fromkeys(extra_kinds, 0.0)
    extra_raw_sum = dict.fromkeys(extra_kinds, 0.0)
    patient_seeds = np.random.SeedSequence(des.random_seed).spawn(n_patients)
    try:
        for patient_index in range(n_patients):
            rng = np.random.default_rng(patient_seeds[patient_index])
            patient_tier = _resample_distributions(
                value_distributions, "per_patient", rng, parameters,
                {**run_context, "_time": 0.0, "_cycle": 0, "_tenure": 1}, tables,
            )
            patient = _Patient({**run_context, **patient_tier})
            patient.outcome_keywords = keyword_map
            if trace_patient is not None and patient_index + 1 == trace_patient:
                tracing["patient"] = patient
                note(patient, "Patient", f"patient {trace_patient}"
                     + (f"; characteristics {_fmt_values(patient_tier)}" if patient_tier else ""))
            else:
                tracing["patient"] = None
            patient.rng = rng
            for tracker in des.trackers:
                patient.trackers[tracker.name] = resolve(tracker.initial_value, patient.context())
            log: list[DESEventRecord] = []
            keep_log = patient_index < log_limit

            if patient.trackers:
                note(patient, "Trackers", f"initial {_fmt_values(patient.trackers)}")
            start_branch = pick_branch(des.branches, patient.context(), rng)
            note(patient, "Start", f"'{start_branch.label}' (p = {start_branch.probability})")
            traverse(start_branch, patient)
            current, terminal_id = walk_path(start_branch.child_id, patient, rng)
            if keep_log:
                log.append(DESEventRecord(0.0, "start", start_branch.label, current))
            final_state = current if current is not None else terminal_id
            if current is not None:
                ctx = patient.context()
                start_node = nodes[current]
                add_event_reward(patient, resolve(start_node.startup_cost, ctx),
                                 resolve(start_node.startup_effect, ctx))
                if start_node.extra_startup:
                    add_extra_event(patient, start_node.extra_startup, ctx)
            else:
                mark_occupancy(terminal_id, 0.0, math.inf)

            events = 0
            while current is not None:
                node: TimeNode = nodes[current]
                patient.state_entry_time = patient.time
                patient.stage_cost = patient.stage_effect = 0.0
                patient.extra_stage = {}
                note(patient, "Enter state", f"{node.name or node.id}")
                apply_tracker_updates(patient, node.tracker_updates)
                ctx = patient.context()
                cost_rate = resolve(node.cost_rate, ctx)
                effect_rate = resolve(node.effect_rate, ctx)
                extra_rates = {
                    outcome_id: resolve(node.extra_rates[outcome_id], ctx)
                    for outcome_id in extra_kinds if outcome_id in node.extra_rates
                }

                if stop_rule is not None and stop_now(patient, patient.time):
                    note(patient, "Terminated", f"termination condition true on entering {node.name or node.id}")
                    # Termination condition already true on entering this
                    # state (at the start or right after an event).
                    event_count_sum[f"{current}:{TERMINATED}"] = event_count_sum.get(f"{current}:{TERMINATED}", 0.0) + 1.0
                    mark_occupancy(TERMINATED, patient.time, math.inf)
                    if keep_log:
                        log.append(DESEventRecord(patient.time, current, TERMINATED, None))
                    final_state = TERMINATED
                    break

                winner: DESBranch | None = None
                best = math.inf
                for branch in node.branches:
                    tte = sample_tte(branch, patient, rng)
                    note(patient, "Sample time", f"'{branch.label}': time to event = {tte:.6g}"
                         + f" (at t = {patient.time + tte:.6g})")
                    if tte < best:
                        best, winner = tte, branch
                start = patient.time
                end = min(start + best, horizon)
                terminated_at = (stop_time_within(patient, start, end, (cost_rate, effect_rate, extra_rates))
                                 if stop_rule is not None else None)
                if terminated_at is not None:
                    end = terminated_at
                duration = end - start
                # Time rewards: per-time-unit value x time spent there.
                patient.total_cost += cost_rate * duration
                patient.total_effect += effect_rate * duration
                patient.stage_cost += cost_rate * duration
                patient.stage_effect += effect_rate * duration
                patient.disc_cost += cost_rate * discounted_duration(cost_discount_rate, start, end)
                patient.disc_effect += effect_rate * discounted_duration(effect_discount_rate, start, end)
                for outcome_id, rate_value in extra_rates.items():
                    patient.extra_raw[outcome_id] = patient.extra_raw.get(outcome_id, 0.0) + rate_value * duration
                    patient.extra_stage[outcome_id] = patient.extra_stage.get(outcome_id, 0.0) + rate_value * duration
                    patient.extra_disc[outcome_id] = patient.extra_disc.get(outcome_id, 0.0) + rate_value * \
                        discounted_duration(extra_rate(outcome_id), start, end)
                time_in_state_sum[current] += duration
                # Exposure per competing event, for the simulated-vs-input
                # survival check: (entry, exit, happened) on the event's own
                # time axis -- time since entering the state (reset clock) or
                # patient time + offset (patient clock; left-truncated).
                event_happened = terminated_at is None and winner is not None and start + best < horizon
                for branch in node.branches:
                    tte_def = branch.time_to_event
                    axis_shift = 0.0
                    if tte_def is not None and tte_def.clock == "forward" and tte_def.distribution not in (
                            "fixed", "expression"):
                        axis_shift = start + (resolve(tte_def.time_offset, ctx) if tte_def.time_offset is not None else 0.0)
                    exposures.setdefault(f"{current}:{branch.label}", []).append(
                        (axis_shift, axis_shift + duration, event_happened and branch is winner))
                note(patient, "Time rewards", f"{node.name or node.id} from t = {start:.6g} to {end:.6g} "
                     f"({duration:.6g} y) at cost {cost_rate:g}/y, effect {effect_rate:g}/y")

                if terminated_at is not None:
                    note(patient, "Terminated", f"termination condition true at t = {end:.6g}")
                    # Stopped by the termination condition before the next
                    # event: time rewards up to that moment, no event reward.
                    mark_occupancy(current, start, end)
                    mark_occupancy(TERMINATED, end, math.inf)
                    patient.time = end
                    event_count_sum[f"{current}:{TERMINATED}"] = event_count_sum.get(f"{current}:{TERMINATED}", 0.0) + 1.0
                    if keep_log:
                        log.append(DESEventRecord(end, current, TERMINATED, None))
                    final_state = TERMINATED
                    break
                if winner is None or start + best >= horizon:
                    patient.time = horizon
                    note(patient, "Horizon", f"time horizon {horizon:g} reached in {node.name or node.id}")
                    mark_occupancy(current, start, math.inf)
                    patient.time = horizon
                    break
                mark_occupancy(current, start, end)
                patient.prior_time = patient.event_time
                patient.event_time = end
                patient.time = end
                event_count_sum[f"{current}:{winner.label}"] += 1.0
                note(patient, "Event", f"'{winner.label}' wins (shortest time) at t = {end:.6g}")
                traverse(winner, patient)
                next_state, terminal_id = walk_path(winner.child_id, patient, rng)
                if keep_log:
                    log.append(DESEventRecord(end, current, winner.label, next_state))
                if next_state is None:
                    mark_occupancy(terminal_id, end, math.inf)
                final_state = next_state if next_state is not None else terminal_id
                current = next_state
                events += 1
                if events >= des.max_events_per_patient:
                    raise ValueError(
                        f"Patient {patient_index} exceeded {des.max_events_per_patient} events "
                        "(check for a zero time-to-event loop)"
                    )

            patient_costs.append(patient.disc_cost)
            patient_effects.append(patient.disc_effect)
            patient_rows.append({
                "patient": patient_index + 1,
                "cost": patient.disc_cost,
                "effect": patient.disc_effect,
                "cost_undiscounted": patient.total_cost,
                "effect_undiscounted": patient.total_effect,
                **{f"{outcome_id}": patient.extra_disc.get(outcome_id, 0.0) for outcome_id in extra_kinds},
                **{f"{outcome_id}_undiscounted": patient.extra_raw.get(outcome_id, 0.0) for outcome_id in extra_kinds},
                "time_simulated": patient.time,
                "n_events": events,
                "final_state": final_state,
                **{f"tracker:{name}": value for name, value in patient.trackers.items()},
            })
            for outcome_id in extra_kinds:
                extra_disc_sum[outcome_id] += patient.extra_disc.get(outcome_id, 0.0)
                extra_raw_sum[outcome_id] += patient.extra_raw.get(outcome_id, 0.0)
            raw_cost_sum += patient.total_cost
            raw_effect_sum += patient.total_effect
            simulated_time_sum += patient.time
            for name in tracker_finals:
                tracker_finals[name].append(patient.trackers[name])
            if keep_log:
                event_logs.append(log)
    except ValueError as exc:
        return DESResult(errors=[ValidationError("des_runtime_error", str(exc), des.id)])

    n = float(n_patients)
    tracker_summary = {
        name: {
            "mean": float(np.mean(values)),
            "p10": float(np.percentile(values, 10)),
            "p50": float(np.percentile(values, 50)),
            "p90": float(np.percentile(values, 90)),
        }
        for name, values in tracker_finals.items() if values
    }
    return DESResult(
        total_cost=float(np.mean(patient_costs)),
        total_effect=float(np.mean(patient_effects)),
        total_cost_undiscounted=raw_cost_sum / n,
        total_effect_undiscounted=raw_effect_sum / n,
        extra_totals={key: value / n for key, value in extra_disc_sum.items()},
        extra_totals_undiscounted={key: value / n for key, value in extra_raw_sum.items()},
        patient_costs=patient_costs,
        patient_effects=patient_effects,
        event_counts={key: value / n for key, value in event_count_sum.items()},
        time_in_state={key: value / n for key, value in time_in_state_sum.items()},
        trace_times=trace_times,
        trace=[{key: value / n for key, value in row.items()} for row in occupancy],
        mean_time_simulated=simulated_time_sum / n,
        tracker_summary=tracker_summary,
        event_log_sample=event_logs,
        patient_rows=patient_rows,
        calculation_trace=calculation_trace,
        event_exposures=exposures,
    )



def run_des_document(
    document: Mapping[str, Any],
    cost_discount_rate: float | None = None,
    effect_discount_rate: float | None = None,
    psa_tier_values: Mapping[str, float] | None = None,
) -> dict[str, DESResult]:
    """Parse + run a whole DES document (Decision -> one DES per strategy)
    and return {strategy label: DESResult}. Discount rates default to the
    document's own "cost_discount_rate"/"effect_discount_rate" (annual)."""

    from .des_parser import resolve_des_namespace, resolve_des_strategies

    parameters, variables, distributions, tables = resolve_des_namespace(document)
    if cost_discount_rate is None:
        cost_discount_rate = float(document.get("cost_discount_rate", 0.0) or 0.0)
    if effect_discount_rate is None:
        effect_discount_rate = float(document.get("effect_discount_rate", 0.0) or 0.0)
    return {
        label: run_des(
            model, parameters, cost_discount_rate, effect_discount_rate,
            variables=variables, distributions=distributions, tables=tables,
            psa_tier_values=psa_tier_values,
        )
        for label, model in resolve_des_strategies(document).items()
    }


def kaplan_meier(exposures: list[tuple[float, float, bool]]) -> tuple[list[float], list[float]]:
    """Kaplan-Meier from (entry, exit, happened) records with left
    truncation (a record is at risk on (entry, exit]); other events and the
    horizon count as censoring -- valid here because DES samples every
    competing event independently. Returns (times, S) as a step function
    starting at (first entry, 1)."""

    if not exposures:
        return [], []
    entries = np.sort(np.array([e[0] for e in exposures], dtype=float))
    exits = np.sort(np.array([e[1] for e in exposures], dtype=float))
    event_times = np.sort(np.array([e[1] for e in exposures if e[2]], dtype=float))
    start = float(entries[0])
    times, values = [start], [1.0]
    if event_times.size == 0:
        return times, values
    unique, counts = np.unique(event_times, return_counts=True)
    at_risk = np.searchsorted(entries, unique, side="left") - np.searchsorted(exits, unique, side="left")
    survival_value = 1.0
    for t, d, n in zip(unique, counts, at_risk):
        if n <= 0:
            continue
        survival_value *= 1.0 - d / n
        times.append(float(t))
        values.append(survival_value)
    return times, values


def input_survival_curve(
    model: DESModel,
    event_key: str,
    times: list[float],
    parameters: Mapping[str, Param] | None = None,
    variables: Mapping[str, Any] | None = None,
    distributions: Mapping[str, Mapping[str, Any]] | None = None,
    tables: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[list[float] | None, str]:
    """S(t) the event was sampled from, on the same axis as its exposures
    ("<time node id>:<event label>"), with parameters resolved at the model
    level (t = 0, initial trackers). Patient clock: conditional on the
    earliest entry, S(t)/S(t0) with t0 = times[0]. Returns (values, note);
    values is None when there is no single input curve (Fixed/Expression
    time, or parameters that differ per patient)."""

    state_id, _, label = event_key.partition(":")
    node = model.nodes.get(state_id)
    branch = next((b for b in getattr(node, "branches", []) if b.label == label), None)
    if branch is None or branch.time_to_event is None:
        return None, "Event not found."
    tte = branch.time_to_event
    if tte.distribution in ("fixed", "expression"):
        return None, "Fixed/Formula time: no input distribution to compare with."
    context: dict[str, Any] = {**(variables or {}), "_time": 0.0, "_cycle": 0, "_tenure": 1,
                               "_time_in_state": 0.0, "_event_time": 0.0, "_prior_time": 0.0}

    def _resolve_plain(value: Any) -> float:
        return float(_resolve(value, parameters or {}, context, distributions or {}, tables or {}))

    for tracker in model.des.trackers:
        try:
            context[tracker.name] = _resolve_plain(tracker.initial_value)
        except (ValueError, KeyError, TypeError):
            pass

    try:
        if tte.distribution == "table_to_time":
            cdf = build_table_cdf((tables or {})[str(tte.params["table"])].get("rows", []),
                                  str(tte.params["index_column"]), str(tte.params["value_column"]),
                                  tte.prob_type, tte.interpolate)
            surv = lambda x: max(0.0, 1.0 - cdf.value_at(x))  # noqa: E731
        elif tte.distribution == "named":
            definition = (distributions or {})[str(tte.params["name"])]
            raw_params = definition.get("parameters") or {}
            closed = NAMED_DISTRIBUTION_CLOSED_FORMS.get(str(definition.get("type", "")))
            if closed is not None:
                key, names = closed
                params = {param: _resolve_plain(raw_params[param]) for param in names}
                surv = lambda x: survival(key, params, x)  # noqa: E731
            else:
                from .distributions import _distribution

                frozen = _distribution(definition, _resolve_plain)
                surv = lambda x: float(frozen.sf(x))  # noqa: E731
        else:
            params = {name: _resolve_plain(value) for name, value in tte.params.items()}
            surv = lambda x: survival(tte.distribution, params, x)  # noqa: E731
    except (ValueError, KeyError, TypeError, AttributeError) as exc:
        return None, f"Input curve not available (parameters may differ per patient): {exc}"
    base = surv(times[0]) if times and tte.clock == "forward" else 1.0
    if base <= 0:
        return None, "Input survival is 0 at the earliest entry."
    note = ("Patient clock: input shown conditional on the earliest entry, S(t)/S(t0)."
            if tte.clock == "forward" else "")
    return [surv(t) / base for t in times], note
