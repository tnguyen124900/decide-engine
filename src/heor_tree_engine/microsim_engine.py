"""Patient-level (microsimulation / individual sampling) simulation.

Execution order, Tracker reset timing and resampling scope are described
in the docstrings below.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from .des_sampling import sample_parametric
from .distributions import distribution_sample
from .markov_engine import _engine_variables, _resolve, _resolve_probability_group
from .microsim_models import MicrosimModel, MicrosimPSMNode, MicrosimResult
from .microsim_validation import validate_microsim
from .models import PSM_STATE_ROLES, NumericRef, Param


_NUMPY_CHOICE_CHECK: list[bool] = []


def _numpy_choice_matches() -> bool:
    """One-off self-check (cached per process) that numpy's
    Generator.choice(a, p=p) is exactly cdf-cumsum + searchsorted on
    rng.random(), which the vectorised microsim loop relies on. If a numpy
    release ever changed that, the original per-patient loop is used."""

    if not _NUMPY_CHOICE_CHECK:
        ok = True
        try:
            probs = np.array([0.2, 0.5, 0.3])
            cdf = probs.cumsum()
            cdf /= cdf[-1]
            labels = ["a", "b", "c"]
            for seed in range(5):
                one = np.random.default_rng(seed)
                two = np.random.default_rng(seed)
                for _ in range(50):
                    picked = str(one.choice(labels, p=probs))
                    mine = labels[int(cdf.searchsorted(two.random(), side="right"))]
                    if picked != mine:
                        ok = False
                batch = np.random.default_rng(seed).random(7)
                seq = np.random.default_rng(seed)
                if not all(batch[i] == seq.random() for i in range(7)):
                    ok = False
        except Exception:
            ok = False
        _NUMPY_CHOICE_CHECK.append(ok)
    return _NUMPY_CHOICE_CHECK[0]



# Per-patient uniform draws are the single most expensive step of a vectorised run (one numpy Generator is built per
# patient, ~15 us each) and are IDENTICAL for every strategy / PSA-trial strategy that shares the seed (Common
# Random Numbers): patient i's stream depends only on (random_seed, i), and a stream's first m draws are the same
# whatever total length is requested.  So the matrix of the last chunk(s) is kept and re-used, which roughly halves the
# time of a two-strategy comparison without changing a single number.  Never cached without a fixed seed.
_UNIFORM_CACHE: dict[tuple[int, int, int], np.ndarray] = {}
_UNIFORM_CACHE_MAX_BYTES = 160 * 1024 * 1024


def _patient_uniforms(random_seed: Any, patient_seeds: Any, chunk_start: int, n: int, total_draws: int) -> np.ndarray:
    """(n, total_draws) uniforms, row j = first `total_draws` draws of patient `chunk_start + j`'s own stream."""

    def build(columns: int) -> np.ndarray:
        out = np.empty((n, columns))
        for j in range(n):
            out[j] = np.random.default_rng(patient_seeds[chunk_start + j]).random(columns)
        return out

    if random_seed is None or n * total_draws * 8 > _UNIFORM_CACHE_MAX_BYTES:
        return build(total_draws)
    try:
        key = (int(random_seed), chunk_start, n)
    except (TypeError, ValueError):
        return build(total_draws)
    cached = _UNIFORM_CACHE.get(key)
    if cached is not None and cached.shape[1] >= total_draws:
        return cached[:, :total_draws]
    columns = max(total_draws, cached.shape[1] if cached is not None else 0)
    matrix = build(columns)
    _UNIFORM_CACHE.pop(key, None)
    while _UNIFORM_CACHE and sum(m.nbytes for m in _UNIFORM_CACHE.values()) + matrix.nbytes > _UNIFORM_CACHE_MAX_BYTES:
        _UNIFORM_CACHE.pop(next(iter(_UNIFORM_CACHE)))
    _UNIFORM_CACHE[key] = matrix
    return matrix[:, :total_draws]


def _resample_distributions(
    distributions: Mapping[str, Mapping[str, Any]] | None,
    tier: str,
    rng: np.random.Generator,
    parameters: Mapping[str, Param],
    context_variables: Mapping[str, Any],
    tables: Mapping[str, Mapping[str, Any]] | None,
    *,
    fold_psa_into_model: bool = True,
) -> dict[str, float]:
    """Draw a fresh sample for every Distribution whose `resample` setting
    matches `tier`, keyed by the distribution's own name so callers can
    merge it straight into that scope's variable context -- e.g. a
    Distribution named `dist_start_age` becomes usable as a plain variable
    `dist_start_age` in any formula evaluated within that scope (a
    Tracker's initial_value, a state's cost, ...), fixed for the whole
    scope instead of resampling on every reference.

    `fold_psa_into_model` (default True) treats a `per_psa`-tier
    Distribution as `per_model` -- run_microsim()'s own default when it
    has no outer PSA loop calling it (the plain Run button, or a
    deterministic SA scenario -- neither passes a `psa_rng`, see
    evaluate_microsim_scenario()). evaluate_microsim_scenario() passes
    False here and draws the "per_psa" tier itself, once per PSA trial,
    shared across every strategy compared in that trial -- matching
    standard PSA practice -- then threads the result into every
    strategy's run_microsim() call via its own `psa_tier_values`
    parameter, which is where this fold is actually skipped (see that
    function's docstring for the full nested-Monte-Carlo picture).
    """

    if not distributions:
        return {}
    sampled: dict[str, float] = {}
    for name, definition in distributions.items():
        resample_tier = str((definition or {}).get("resample") or "per_model")
        if resample_tier == "per_psa" and fold_psa_into_model:
            resample_tier = "per_model"
        if resample_tier != tier:
            continue

        def resolve(value: Any, _context: Mapping[str, Any] = context_variables) -> float:
            return _resolve(value, parameters, _context, distributions, tables)

        sampled[name] = distribution_sample(definition, resolve, rng=rng)
    return sampled


def _psm_role_at(elapsed_cycles: float, pfs_cycle: float, os_cycle: float) -> str:
    """Which of PSM_STATE_ROLES a patient occupies `elapsed_cycles` after
    entering a PSM node, given their own sampled pfs_cycle/os_cycle (both
    already on the model's cycle axis, and already reflecting
    effective_pfs_time = min(pfs_time, os_time) for pfs_cycle -- see
    _sample_psm_entry_times). Deterministic: no further RNG draws needed
    for as long as the patient stays in the node."""

    if elapsed_cycles >= os_cycle:
        return "dead"
    if elapsed_cycles >= pfs_cycle:
        return "post_progression"
    return "pre_progression"


def _sample_psm_entry_times(
    node: MicrosimPSMNode,
    rng: np.random.Generator,
    parameters: Mapping[str, Param],
    ctx_variables: Mapping[str, Any],
    distributions: Mapping[str, Mapping[str, Any]] | None,
    tables: Mapping[str, Mapping[str, Any]] | None,
    cycles_per_year: float,
) -> tuple[float, float]:
    """Draw ONE pfs_time and ONE os_time for a single patient entering
    `node` (inverse-transform sampling via des_sampling.sample_parametric,
    the same closed-form machinery DES time-to-event sampling already
    uses -- reused, not reimplemented), on the curves' own time axis
    (years, this codebase's established PSM convention). Returns
    (pfs_cycle, os_cycle): both already converted to the model's own
    cycle axis via cycles_per_year, and pfs_cycle already folds in
    effective_pfs_time = min(pfs_time, os_time) so a patient whose
    sampled PFS event lands after their sampled OS event (individual-
    level crossing) still gets a well-defined role sequence."""

    pfs_params = {
        name: _resolve(value, parameters, ctx_variables, distributions, tables)
        for name, value in node.pfs_curve.params.items()
    }
    os_params = {
        name: _resolve(value, parameters, ctx_variables, distributions, tables)
        for name, value in node.os_curve.params.items()
    }
    pfs_time = sample_parametric(node.pfs_curve.distribution, pfs_params, 1.0 - rng.random(), lower=0.0)
    os_time = sample_parametric(node.os_curve.distribution, os_params, 1.0 - rng.random(), lower=0.0)
    effective_pfs_time = min(pfs_time, os_time)
    return effective_pfs_time * cycles_per_year, os_time * cycles_per_year


def run_microsim(
    model: MicrosimModel,
    parameters: Mapping[str, Param] | None = None,
    discount_cost_per_cycle: float = 0.0,
    discount_effect_per_cycle: float = 0.0,
    allow_negative_cost: bool = False,
    variables: Mapping[str, Any] | None = None,
    distributions: Mapping[str, Mapping[str, Any]] | None = None,
    tables: Mapping[str, Mapping[str, Any]] | None = None,
    psa_tier_values: Mapping[str, float] | None = None,
    cycles_per_year: float = 1.0,
) -> MicrosimResult:
    """Simulate `model.n_patients` independent patients through the model
    and return population-level totals (mean across patients, so existing
    cohort-style result rendering keeps working unchanged) plus
    per-patient samples and Tracker summaries.

    Execution order each cycle:
      1. Seed context with this cycle's `_cycle`/`_tenure` and every
         Tracker's value as carried from the previous cycle (or
         `initial_value` at cycle 0).
      2. Resolve this state's outgoing transition probabilities using
         that context (read-only, doesn't touch Trackers) -- also used to
         blend a tenure-applied state's initial/recurring/terminal cost
         as an expectation over the resolved probabilities, exactly like
         the Markov cohort engine's `reward()`, since a single sampled
         patient can't otherwise know before sampling whether it is about
         to leave the state this cycle.
      3. Compute cost/effect from that same context.
      4. Sample the actual transition from the probabilities already
         resolved in step 2.
      5. Evaluate every Tracker's update_formula using the step-1 context
         (so nothing sees a "newer" value of anything else this cycle;
         avoids a same-cycle circular dependency between probability and
         Tracker).
      6. If the patient just entered a new state, apply that state's
         reset_rules -- immediately ("on_entry") or delayed until
         the state's second cycle ("next_cycle") per
         model.tracker_reset_timing.

    Each patient is simulated with its own independent RNG sub-stream
    (derived from `model.random_seed` via `numpy.random.SeedSequence.spawn`)
    so that "patient i" draws the same underlying random numbers across
    different strategies/scenarios that share the same seed and
    n_patients -- true Common Random Numbers (CRN), robust to strategies
    consuming different amounts of randomness per patient.

    `psa_tier_values`, when given, is a dict of already-sampled `per_psa`
    Distribution values (sampled once per PSA trial, shared across every
    strategy compared in that trial) to merge into this run's variables --
    see evaluate_microsim_scenario() in sensitivity.py, which is the only
    caller that passes it. When omitted (the plain Run button, or a
    deterministic SA scenario), any `per_psa` Distribution instead folds
    into `per_model` inside this single run, as before.
    """

    initial_variables = _engine_variables(0, variables=variables)
    resolved_parameters = parameters or {}
    errors = validate_microsim(
        model,
        resolved_parameters,
        allow_negative_cost,
        variables=initial_variables,
        distributions=distributions,
        tables=tables,
        cycles_per_year=cycles_per_year,
    )
    if errors:
        return MicrosimResult(errors=errors)

    rng = np.random.default_rng(model.random_seed)

    states = model.states
    state_ids = [state.id for state in states]
    states_by_id = {state.id: state for state in states}
    # Same real / non-real split as the Markov cohort engine: a "real"
    # state (a direct child of the strategy's root "M" node on the canvas)
    # is a health state a patient can occupy for a whole cycle. A non-real
    # state (a canvas child of another state -- the branch nodes and
    # terminal leaves drawn under a health state) is NEVER occupied: it is
    # only a point along the way within ONE cycle. A terminal leaf means
    # "this cycle ends here, in the real health state this leaf stands for";
    # the next cycle the patient moves on again from that real state.
    real_state_ids = [state.id for state in states if state.is_real_state]
    real_state_by_name = {state.name: state.id for state in states if state.is_real_state}
    non_real_present = len(real_state_ids) != len(states)
    psm_nodes_by_id = {node.id: node for node in model.psm_nodes}
    psm_role_ids_all = [f"{node.id}::{role}" for node in model.psm_nodes for role in PSM_STATE_ROLES]
    all_ids = real_state_ids + psm_role_ids_all
    initial_choice_ids = real_state_ids + list(psm_nodes_by_id.keys())
    grouped: dict[str, list[tuple[str, NumericRef]]] = {state_id: [] for state_id in state_ids}
    transition_by_pair: dict[tuple[str, str], Any] = {}
    for transition in model.transitions:
        grouped[transition.from_state].append((transition.to_state, transition.probability))
        transition_by_pair[(transition.from_state, transition.to_state)] = transition
    # Skip the per-patient path walk entirely (zero overhead, identical RNG
    # draws to before) on the common case: every state is real and no
    # transition carries a one-time Incre_$/Incre_E.
    has_path_rewards = non_real_present or any(
        transition.cost_increment or transition.effect_increment for transition in model.transitions
    )

    def outgoing_for_state(state_id: str, ctx_variables: Mapping[str, Any]) -> dict[str, float]:
        entries = grouped[state_id]
        if not entries:
            # Absorbing state with no declared transitions: implicit
            # self-loop, exactly like the Markov cohort engine.
            return {state_id: 1.0}
        return _resolve_probability_group(entries, resolved_parameters, ctx_variables, distributions, tables)

    def _alias_target(state_obj: Any) -> str | None:
        """The real state a non-real dead-end leaf stands for: its explicit
        alias_of (set by "relabel as terminal"), else a real state with the
        same display name (models saved before alias_of existed)."""
        target = state_obj.alias_of or real_state_by_name.get(state_obj.name)
        target_obj = states_by_id.get(target) if target else None
        return target if target_obj is not None and target_obj.is_real_state else None

    def _loop_error(state_id: str) -> ValueError:
        return ValueError(
            f"Microsimulation model has a loop entirely among non-real (canvas-child) "
            f"states -- '{state_id}' is reached again without ever passing through a real "
            f"state. A branch/decision chain must eventually reach a real state."
        )

    def _unresolved_alias_error(state_id: str) -> ValueError:
        return ValueError(
            f"State '{state_id}' has no outgoing transitions and doesn't match any real "
            f"state by alias or name -- use 'relabel as terminal' on it to link it to the "
            f"real state it represents, or give it its own outgoing transition(s)."
        )

    def _collapsed_probabilities(state_id: str, ctx_variables: Mapping[str, Any]) -> dict[str, float]:
        """Analytic (no sampling) probability of ending ONE cycle in each
        real state (or PSM node), starting from `state_id` and passing
        through any chain of non-real states in zero elapsed time. Used to
        get a tenure-applied state's exit probability, exactly like the
        Markov cohort engine's _collapse()."""
        landing: dict[str, float] = {}

        def walk(current_id: str, prob_so_far: float, visited: frozenset[str]) -> None:
            for to_state, probability in outgoing_for_state(current_id, ctx_variables).items():
                if probability == 0.0:
                    continue
                path_prob = prob_so_far * probability
                to_obj = states_by_id.get(to_state)
                if to_obj is None or to_obj.is_real_state:
                    landing[to_state] = landing.get(to_state, 0.0) + path_prob
                    continue
                if to_state in visited:
                    raise _loop_error(to_state)
                if not grouped[to_state]:
                    alias_id = _alias_target(to_obj)
                    if alias_id is None:
                        raise _unresolved_alias_error(to_state)
                    landing[alias_id] = landing.get(alias_id, 0.0) + path_prob
                    continue
                walk(to_state, path_prob, visited | {to_state})

        walk(state_id, 1.0, frozenset({state_id}))
        return landing

    def _choose_target(outgoing_map: Mapping[str, float], path_rng: np.random.Generator) -> str:
        outcomes = list(outgoing_map.keys())
        probs = np.clip(np.array(list(outgoing_map.values()), dtype=float), 0.0, None)
        prob_total = probs.sum()
        probs = probs / prob_total if prob_total > 0 else np.full(len(outcomes), 1.0 / len(outcomes))
        return str(path_rng.choice(outcomes, p=probs))

    def _follow_path(
        origin_id: str, first_target: str, ctx_variables: Mapping[str, Any], path_rng: np.random.Generator
    ) -> tuple[str, float, float]:
        """Walk ONE patient's sampled path from the first sampled hop all
        the way to the real state where this cycle ends (zero elapsed
        cycles through non-real branch/terminal nodes), summing every
        transition's one-time Incre_$/Incre_E along the way. Returns
        (landing_state_id, one_time_cost, one_time_effect). A non-real
        node's own cost/effect fields are deliberately NOT charged: its
        one-time contribution is, by design, the Incre_$/Incre_E on the
        transition INTO it (what the canvas shows)."""
        path_cost = 0.0
        path_effect = 0.0
        hop_from = origin_id
        target_id = first_target
        visited = {origin_id}
        while True:
            transition = transition_by_pair.get((hop_from, target_id))
            if transition is not None:
                if transition.cost_increment:
                    path_cost += _resolve(
                        transition.cost_increment, resolved_parameters, ctx_variables, distributions, tables
                    )
                if transition.effect_increment:
                    path_effect += _resolve(
                        transition.effect_increment, resolved_parameters, ctx_variables, distributions, tables
                    )
            target_obj = states_by_id.get(target_id)
            if target_obj is None or target_obj.is_real_state:
                return target_id, path_cost, path_effect
            if target_id in visited:
                raise _loop_error(target_id)
            visited.add(target_id)
            if not grouped[target_id]:
                alias_id = _alias_target(target_obj)
                if alias_id is None:
                    raise _unresolved_alias_error(target_id)
                return alias_id, path_cost, path_effect
            hop_from = target_id
            target_id = _choose_target(outgoing_for_state(target_id, ctx_variables), path_rng)

    base_variables = dict(variables or {})
    model_tier_values = _resample_distributions(
        distributions, "per_model", rng, resolved_parameters,
        {**base_variables, "_cycle": 0, "_tenure": 1}, tables,
        fold_psa_into_model=psa_tier_values is None,
    )
    run_level_variables = {**base_variables, **model_tier_values, **(psa_tier_values or {})}

    n_cycles = model.cycles
    n_patients = model.n_patients
    reset_on_entry = model.tracker_reset_timing == "on_entry"

    cost_sum_per_cycle = [0.0] * n_cycles
    effect_sum_per_cycle = [0.0] * n_cycles
    raw_cost_sum_per_cycle = [0.0] * n_cycles
    raw_effect_sum_per_cycle = [0.0] * n_cycles
    occupancy_sum: list[dict[str, float]] = [dict.fromkeys(all_ids, 0.0) for _ in range(n_cycles + 1)]
    # transition_sum[cycle][from_state][to_state] -- a simple tally
    # incremented once per patient per cycle, right where the patient's
    # actual sampled transition is already known (see the CRN docstring
    # above for the RNG-substream context this loop runs under). No
    # per-patient identity is kept, only the count, so this doesn't change
    # what's computed, just what's additionally recorded.
    transition_sum: list[dict[str, dict[str, float]]] = [
        {from_id: dict.fromkeys(all_ids, 0.0) for from_id in all_ids} for _ in range(n_cycles)
    ]
    patient_costs: list[float] = []
    patient_effects: list[float] = []
    tracker_finals: dict[str, list[float]] = {
        tracker.name: [] for tracker in model.trackers if tracker.include_in_output
    }

    # Each patient gets an independent, reproducible RNG sub-stream derived
    # from the model's seed via SeedSequence.spawn(), rather than all
    # patients sharing one sequential `rng` stream. This is what makes
    # Common Random Numbers (CRN) across strategies actually valid: two
    # strategies with different state/transition/tracker structure consume
    # different amounts of randomness per patient from a shared stream, so
    # "patient i" would silently desynchronize between strategies under
    # naive seed reuse. With per-patient sub-streams, patient i's draws are
    # pinned to seed i regardless of what either strategy does around them.
    patient_seeds = np.random.SeedSequence(model.random_seed).spawn(n_patients)
    # First `patient_trace_sample_size` patients (by index -- always the
    # same ones for a given n_patients, since patient_index order is
    # itself reproducible from model.random_seed) additionally get their
    # full per-cycle state sequence recorded. Capped at n_patients so a
    # small model with a large sample-size setting doesn't error.
    trace_sample_size = min(max(model.patient_trace_sample_size, 0), n_patients)
    sample_patient_traces: list[list[str]] = [[] for _ in range(trace_sample_size)]
    # ------------------------------------------------------------------
    # Vectorised patient loop (2026-10).
    #
    # Every quantity the per-patient loop below computes for a patient in
    # a cycle is a pure function of that patient's CONTEXT -- (state,
    # tenure, Tracker values) at that cycle, since without per_patient /
    # per_cycle Distributions every patient shares the same run-level
    # variables -- plus ONE uniform draw from the patient's own RNG
    # sub-stream per sampled hop. So all patients are advanced together
    # one cycle at a time: patients are grouped by identical context, each
    # group's probabilities / cost / effect / Tracker updates are
    # evaluated ONCE with the very same _resolve() calls as before, and the
    # sampled transition is a vectorised inverse-CDF lookup
    # (searchsorted, exactly what numpy's Generator.choice(p=...) does) on
    # draws that were pre-generated PER PATIENT from that patient's own
    # SeedSequence child (rng.random(K) is the same stream, value for
    # value, as K successive rng.random() calls). Patient i therefore
    # still sees the identical draws regardless of N or of the strategy,
    # and every per-cycle sum is accumulated in patient order with a
    # sequential add so the result matches the old loop bit for bit.
    #
    # Models with PSM nodes, per_patient / per_cycle Distributions (extra
    # RNG consumption), stochastic functions inside formulas, loops among
    # non-real states or duplicate Tracker names keep using the original
    # per-patient loop below, untouched.
    # ------------------------------------------------------------------
    def _microsim_vector_ok() -> bool:
        if n_patients < 1 or model.psm_nodes:
            return False
        for definition in (distributions or {}).values():
            if str((definition or {}).get("resample") or "per_model") in ("per_patient", "per_cycle"):
                return False
        text = repr(model) + repr(variables) + repr(distributions)
        if any(token in text for token in ("DistSample", "DistToTime", "TableToTime", "DistTrim")):
            return False
        names = [tracker.name for tracker in model.trackers]
        if len(set(names)) != len(names):
            return False
        for transition in model.transitions:
            if transition.to_state not in states_by_id or transition.from_state not in states_by_id:
                return False
        if any(
            (state.is_real_state and state.id not in states_by_id) for state in states
        ):
            return False
        # Non-real subgraph must be acyclic (else the original loop raises
        # the proper error at the right moment).
        colour: dict[str, int] = {}

        def acyclic(node: str) -> bool:
            colour[node] = 1
            for to_state, _prob in grouped[node]:
                to_obj = states_by_id[to_state]
                if to_obj.is_real_state:
                    continue
                if colour.get(to_state) == 1:
                    return False
                if colour.get(to_state) is None and not acyclic(to_state):
                    return False
            colour[node] = 2
            return True

        for state in states:
            if not state.is_real_state and colour.get(state.id) is None and not acyclic(state.id):
                return False
        return _numpy_choice_matches()

    _vector_ok = _microsim_vector_ok()

    if _vector_ok:
        # Longest chain of draws a single patient-cycle can consume:
        # 1 (the cycle's own transition) + one per non-real node with
        # outgoing transitions visited along the path.
        depth_memo: dict[str, int] = {}

        def draw_depth(node: str) -> int:
            cached = depth_memo.get(node)
            if cached is not None:
                return cached
            if not grouped[node]:
                depth_memo[node] = 0
                return 0
            best = 0
            for to_state, _prob in grouped[node]:
                if not states_by_id[to_state].is_real_state:
                    best = max(best, draw_depth(to_state))
            depth_memo[node] = 1 + best
            return 1 + best

        extra_draws = max([0] + [draw_depth(s.id) for s in states if not s.is_real_state])
        draws_per_cycle = 1 + extra_draws
        total_draws = 1 + n_cycles * draws_per_cycle
        chunk_size = max(1, min(n_patients, 16_000_000 // total_draws))

        n_all = len(state_ids)
        sidx = {sid: i for i, sid in enumerate(state_ids)}
        tracker_list = list(model.trackers)
        n_trk = len(tracker_list)
        tracker_names = [tracker.name for tracker in tracker_list]
        text_all = repr(model) + repr(variables) + repr(distributions) + repr(tables)
        tenure_matters = "_tenure" in text_all
        ones_tenure_cap = 2  # tenure only matters through `tenure == 1` when not referenced

        initial_ids_arr = np.array([sidx[sid] for sid in initial_choice_ids], dtype=np.int64)
        initial_probs_v = _resolve_probability_group(
            [(state.id, state.initial_prob) for state in states if state.is_real_state],
            resolved_parameters,
            {**run_level_variables, "_cycle": 0, "_tenure": 1},
            distributions,
            tables,
        )
        weights_v = np.array([max(0.0, initial_probs_v.get(sid, 0.0)) for sid in initial_choice_ids], dtype=float)
        weight_total_v = weights_v.sum()
        weights_v = (
            weights_v / weight_total_v
            if weight_total_v > 0
            else np.full(len(initial_choice_ids), 1.0 / len(initial_choice_ids))
        )
        np.random.default_rng(0).choice(initial_choice_ids, p=weights_v)  # same validation as the loop
        initial_cdf = weights_v.cumsum()
        initial_cdf /= initial_cdf[-1]
        initial_tracker_values = [
            _resolve(
                tracker.initial_value, resolved_parameters,
                {**run_level_variables, "_cycle": 0, "_tenure": 1}, distributions, tables,
            )
            for tracker in tracker_list
        ]
        # cached cumulative distribution per distinct outgoing probability map
        choice_cache: dict[tuple, tuple[np.ndarray, list[str]]] = {}

        def prepare_choice(outgoing_map: Mapping[str, float]) -> tuple[np.ndarray, list[str]]:
            outcomes = list(outgoing_map.keys())
            probs = np.clip(np.array(list(outgoing_map.values()), dtype=float), 0.0, None)
            prob_total = probs.sum()
            probs = probs / prob_total if prob_total > 0 else np.full(len(outcomes), 1.0 / len(outcomes))
            key = (tuple(outcomes), probs.tobytes())
            cached = choice_cache.get(key)
            if cached is None:
                np.random.default_rng(0).choice(outcomes, p=probs)  # same validation as _choose_target
                cdf = probs.cumsum()
                cdf /= cdf[-1]
                if len(choice_cache) > 4096:
                    choice_cache.clear()
                cached = choice_cache[key] = (cdf, outcomes)
            return cached

        def row_groups(columns: list[np.ndarray]) -> tuple[np.ndarray, int]:
            """Dense group code per row for rows identical across all
            `columns` (int64 arrays; floats are passed as their bit
            patterns so grouping is exact)."""
            key = None
            count = 0
            for col in columns:
                uniq, inv = np.unique(col, return_inverse=True)
                inv = inv.reshape(-1).astype(np.int64)
                if key is None:
                    key, count = inv, len(uniq)
                else:
                    key = key * len(uniq) + inv
                    uniq2, key = np.unique(key, return_inverse=True)
                    key = key.reshape(-1).astype(np.int64)
                    count = len(uniq2)
            return key, count

        def first_of_groups(key: np.ndarray, count: int) -> np.ndarray:
            _unused, first = np.unique(key, return_index=True)
            return first

        def ordered_slices(key: np.ndarray, count: int):
            order = np.argsort(key, kind="stable")
            bounds = np.concatenate(([0], np.cumsum(np.bincount(key, minlength=count))))
            return order, bounds

        def tenure_key(tenure_arr: np.ndarray) -> np.ndarray:
            return tenure_arr if tenure_matters else np.minimum(tenure_arr, ones_tenure_cap)

        def ctx_for(state_name: str, tenure_value: int, tracker_row: np.ndarray, cycle_value: int) -> dict[str, Any]:
            return {
                **patient_variables_v,
                **{name: float(tracker_row[k]) for k, name in enumerate(tracker_names)},
                "_cycle": cycle_value,
                "_tenure": tenure_value,
            }

        patient_variables_v = dict(run_level_variables)
        S = n_all
        discount_cost_den = [(1.0 + discount_cost_per_cycle) ** cycle for cycle in range(n_cycles)]
        discount_effect_den = [(1.0 + discount_effect_per_cycle) ** cycle for cycle in range(n_cycles)]
        real_mask_all = np.array([states_by_id[sid].is_real_state for sid in state_ids])
        reset_state_mask = [
            np.array([sid in tracker.reset_rules for sid in state_ids]) for tracker in tracker_list
        ]

        for chunk_start in range(0, n_patients, chunk_size):
            chunk_stop = min(n_patients, chunk_start + chunk_size)
            n = chunk_stop - chunk_start
            uniforms = _patient_uniforms(model.random_seed, patient_seeds, chunk_start, n, total_draws)
            pos = np.ones(n, dtype=np.int64)
            row_ids = np.arange(n)

            chosen = initial_cdf.searchsorted(uniforms[:, 0], side="right")
            state_arr = initial_ids_arr[chosen]
            tenure_arr = np.ones(n, dtype=np.int64)
            trk = np.empty((n, n_trk))
            for k in range(n_trk):
                trk[:, k] = initial_tracker_values[k]

            counts0 = np.bincount(state_arr, minlength=S)
            for i in np.nonzero(counts0)[0]:
                occupancy_sum[0][state_ids[i]] += float(counts0[i])
            trace_n = max(0, min(trace_sample_size - chunk_start, n))
            for j in range(trace_n):
                sample_patient_traces[chunk_start + j].append(state_ids[state_arr[j]])

            tot_cost = np.zeros(n)
            tot_effect = np.zeros(n)
            first_cost = np.zeros(n)
            first_effect = np.zeros(n)

            for cycle in range(n_cycles):
                cols = [state_arr * (n_cycles + 3) + tenure_key(tenure_arr)]
                for k in range(n_trk):
                    cols.append(np.ascontiguousarray(trk[:, k]).view(np.int64))
                gkey, gcount = row_groups(cols)
                gfirst = first_of_groups(gkey, gcount)
                order, bounds = ordered_slices(gkey, gcount)

                g_cost = np.empty(gcount)
                g_effect = np.empty(gcount)
                g_ctx: list[dict[str, Any]] = []
                g_state: list[str] = []
                g_outgoing: list[dict[str, float]] = []
                g_newtrk = np.empty((gcount, n_trk))
                for g in range(gcount):
                    r = int(gfirst[g])
                    st = state_ids[state_arr[r]]
                    tn = int(tenure_arr[r])
                    ctx = ctx_for(st, tn, trk[r], cycle)
                    outgoing = outgoing_for_state(st, ctx)
                    current = states_by_id[st]
                    if not current.tenure_applied:
                        cost_g = _resolve(current.cost, resolved_parameters, ctx, distributions, tables)
                    else:
                        initial_cost = _resolve(current.initial_cost, resolved_parameters, ctx, distributions, tables)
                        recurring_cost = _resolve(
                            current.recurring_cost if current.recurring_cost is not None else current.cost,
                            resolved_parameters, ctx, distributions, tables,
                        )
                        terminal_cost = _resolve(current.terminal_cost, resolved_parameters, ctx, distributions, tables)
                        exit_probability = sum(
                            p for to_state, p in _collapsed_probabilities(st, ctx).items() if to_state != st
                        )
                        base_cost = initial_cost if tn == 1 else recurring_cost * (1.0 - exit_probability)
                        cost_g = base_cost + (0.0 if tn == 1 else terminal_cost * exit_probability)
                    effect_g = _resolve(current.effect, resolved_parameters, ctx, distributions, tables)
                    g_cost[g] = cost_g
                    g_effect[g] = effect_g
                    g_ctx.append(ctx)
                    g_state.append(st)
                    g_outgoing.append(outgoing)

                landed = np.empty(n, dtype=np.int64)
                path_cost = np.zeros(n) if has_path_rewards else None
                path_effect = np.zeros(n) if has_path_rewards else None

                def sample_from(g: int, from_id: str, pidx: np.ndarray, outgoing_map: Mapping[str, float]) -> None:
                    cdf, outs = prepare_choice(outgoing_map)
                    u = uniforms[pidx, pos[pidx]]
                    pos[pidx] += 1
                    loc = cdf.searchsorted(u, side="right")
                    for k in np.unique(loc):
                        sub = pidx[loc == k]
                        target_id = outs[int(k)]
                        if has_path_rewards:
                            transition = transition_by_pair.get((from_id, target_id))
                            if transition is not None:
                                if transition.cost_increment:
                                    path_cost[sub] += _resolve(
                                        transition.cost_increment, resolved_parameters, g_ctx[g], distributions, tables
                                    )
                                if transition.effect_increment:
                                    path_effect[sub] += _resolve(
                                        transition.effect_increment, resolved_parameters, g_ctx[g], distributions, tables
                                    )
                        target_obj = states_by_id.get(target_id)
                        if target_obj is None or target_obj.is_real_state:
                            landed[sub] = sidx[target_id]
                        elif not grouped[target_id]:
                            alias_id = _alias_target(target_obj)
                            if alias_id is None:
                                raise _unresolved_alias_error(target_id)
                            landed[sub] = sidx[alias_id]
                        else:
                            sample_from(g, target_id, sub, outgoing_for_state(target_id, g_ctx[g]))

                for g in range(gcount):
                    pidx = order[bounds[g]:bounds[g + 1]]
                    sample_from(g, g_state[g], pidx, g_outgoing[g])

                cost_p = g_cost[gkey]
                effect_p = g_effect[gkey]
                if has_path_rewards:
                    cost_p = cost_p + path_cost
                    effect_p = effect_p + path_effect
                disc_cost = cost_p / discount_cost_den[cycle]
                disc_effect = effect_p / discount_effect_den[cycle]
                cost_sum_per_cycle[cycle] = float(
                    np.add.accumulate(np.concatenate(([cost_sum_per_cycle[cycle]], disc_cost)))[-1]
                )
                effect_sum_per_cycle[cycle] = float(
                    np.add.accumulate(np.concatenate(([effect_sum_per_cycle[cycle]], disc_effect)))[-1]
                )
                raw_cost_sum_per_cycle[cycle] = float(
                    np.add.accumulate(np.concatenate(([raw_cost_sum_per_cycle[cycle]], cost_p)))[-1]
                )
                raw_effect_sum_per_cycle[cycle] = float(
                    np.add.accumulate(np.concatenate(([raw_effect_sum_per_cycle[cycle]], effect_p)))[-1]
                )
                tot_cost += disc_cost
                tot_effect += disc_effect
                if cycle == 0:
                    first_cost = disc_cost.copy()
                    first_effect = disc_effect.copy()

                pair_counts = np.bincount(state_arr * S + landed, minlength=S * S)
                for flat in np.nonzero(pair_counts)[0]:
                    transition_sum[cycle][state_ids[flat // S]][state_ids[flat % S]] += float(pair_counts[flat])
                entering = landed != state_arr

                # Tracker updates (formulas see the step-1 context).
                for g in range(gcount):
                    for k, tracker in enumerate(tracker_list):
                        formula = tracker.update_formula if tracker.update_formula is not None else tracker.name
                        g_newtrk[g, k] = _resolve(formula, resolved_parameters, g_ctx[g], distributions, tables)
                new_trk = g_newtrk[gkey] if n_trk else np.empty((n, 0))
                for k, tracker in enumerate(tracker_list):
                    if not tracker.reset_rules:
                        continue
                    if reset_on_entry:
                        mask = entering & reset_state_mask[k][landed]
                        key_state = landed
                    else:
                        mask = (~entering) & (tenure_arr == 1) & reset_state_mask[k][state_arr]
                        key_state = state_arr
                    idxs = np.nonzero(mask)[0]
                    if idxs.size == 0:
                        continue
                    cols2 = [gkey[idxs], key_state[idxs]] + [
                        np.ascontiguousarray(new_trk[idxs, j]).view(np.int64) for j in range(n_trk)
                    ]
                    key2, count2 = row_groups(cols2)
                    first2 = first_of_groups(key2, count2)
                    for g2 in range(count2):
                        rr = int(first2[g2])
                        r_global = int(idxs[rr])
                        ctx2 = {
                            **g_ctx[int(gkey[r_global])],
                            **{name: float(new_trk[r_global, j]) for j, name in enumerate(tracker_names)},
                        }
                        value = _resolve(
                            tracker.reset_rules[state_ids[key_state[r_global]]],
                            resolved_parameters, ctx2, distributions, tables,
                        )
                        new_trk[idxs[key2 == g2], k] = value
                trk = new_trk
                tenure_arr = np.where(entering, 1, tenure_arr + 1)
                state_arr = landed

                counts_n = np.bincount(state_arr, minlength=S)
                for i in np.nonzero(counts_n)[0]:
                    occupancy_sum[cycle + 1][state_ids[i]] += float(counts_n[i])
                for j in range(trace_n):
                    sample_patient_traces[chunk_start + j].append(state_ids[state_arr[j]])

            if model.half_cycle_correction and n_cycles > 0:
                cols = [state_arr * (n_cycles + 3) + tenure_key(tenure_arr)]
                for k in range(n_trk):
                    cols.append(np.ascontiguousarray(trk[:, k]).view(np.int64))
                gkey, gcount = row_groups(cols)
                gfirst = first_of_groups(gkey, gcount)
                f_cost = np.empty(gcount)
                f_effect = np.empty(gcount)
                for g in range(gcount):
                    r = int(gfirst[g])
                    final_state = states_by_id[state_ids[state_arr[r]]]
                    tn = int(tenure_arr[r])
                    final_ctx = {
                        **patient_variables_v,
                        **{name: float(trk[r, k]) for k, name in enumerate(tracker_names)},
                        "_cycle": n_cycles,
                        "_tenure": tn,
                    }
                    if not final_state.tenure_applied:
                        final_cost = _resolve(final_state.cost, resolved_parameters, final_ctx, distributions, tables)
                    else:
                        final_cost = _resolve(
                            final_state.initial_cost
                            if tn == 1
                            else (final_state.recurring_cost if final_state.recurring_cost is not None else final_state.cost),
                            resolved_parameters, final_ctx, distributions, tables,
                        )
                    final_effect = _resolve(final_state.effect, resolved_parameters, final_ctx, distributions, tables)
                    f_cost[g] = final_cost
                    f_effect[g] = final_effect
                disc_final_cost = f_cost[gkey] / (1.0 + discount_cost_per_cycle) ** n_cycles
                disc_final_effect = f_effect[gkey] / (1.0 + discount_effect_per_cycle) ** n_cycles
                tot_cost += 0.5 * (disc_final_cost - first_cost)
                tot_effect += 0.5 * (disc_final_effect - first_effect)

            patient_costs.extend(tot_cost.tolist())
            patient_effects.extend(tot_effect.tolist())
            for k, tracker in enumerate(tracker_list):
                if tracker.include_in_output:
                    tracker_finals[tracker.name].extend(trk[:, k].tolist())

    for patient_index in (range(n_patients) if not _vector_ok else ()):
        patient_rng = np.random.default_rng(patient_seeds[patient_index])
        patient_tier_values = _resample_distributions(
            distributions, "per_patient", patient_rng, resolved_parameters,
            {**run_level_variables, "_cycle": 0, "_tenure": 1}, tables,
        )
        patient_variables = {**run_level_variables, **patient_tier_values}

        initial_probs = _resolve_probability_group(
            [(state.id, state.initial_prob) for state in states if state.is_real_state]
            + [(node.id, node.initial_prob) for node in model.psm_nodes],
            resolved_parameters,
            {**patient_variables, "_cycle": 0, "_tenure": 1},
            distributions,
            tables,
        )
        weights = np.array([max(0.0, initial_probs.get(sid, 0.0)) for sid in initial_choice_ids], dtype=float)
        weight_total = weights.sum()
        weights = (
            weights / weight_total
            if weight_total > 0
            else np.full(len(initial_choice_ids), 1.0 / len(initial_choice_ids))
        )
        chosen_start = str(patient_rng.choice(initial_choice_ids, p=weights))
        # psm_ctx is None while occupying a real MarkovState; once set, it
        # holds {node_id, pfs_cycle, os_cycle, entry_cycle} for as long as
        # the patient stays inside that PSM node (Phase 1: forever, once
        # entered -- PSM roles are terminal, no onward transitions).
        psm_ctx: dict[str, Any] | None = None
        if chosen_start in psm_nodes_by_id:
            entry_node = psm_nodes_by_id[chosen_start]
            pfs_cycle, os_cycle = _sample_psm_entry_times(
                entry_node, patient_rng, resolved_parameters,
                {**patient_variables, "_cycle": 0, "_tenure": 1}, distributions, tables, cycles_per_year,
            )
            state = f"{entry_node.id}::{_psm_role_at(0.0, pfs_cycle, os_cycle)}"
            psm_ctx = {"node_id": entry_node.id, "pfs_cycle": pfs_cycle, "os_cycle": os_cycle, "entry_cycle": 0}
        else:
            state = chosen_start
        tenure = 1

        tracker_values: dict[str, float] = {}
        for tracker in model.trackers:
            tracker_values[tracker.name] = _resolve(
                tracker.initial_value, resolved_parameters,
                {**patient_variables, "_cycle": 0, "_tenure": 1}, distributions, tables,
            )

        occupancy_sum[0][state] += 1.0
        if patient_index < trace_sample_size:
            sample_patient_traces[patient_index].append(state)
        patient_total_cost = 0.0
        patient_total_effect = 0.0
        first_cycle_cost = 0.0
        first_cycle_effect = 0.0

        for cycle in range(n_cycles):
            cycle_tier_values = _resample_distributions(
                distributions, "per_cycle", patient_rng, resolved_parameters,
                {**patient_variables, "_cycle": cycle, "_tenure": tenure}, tables,
            )
            ctx_vars = {
                **patient_variables,
                **cycle_tier_values,
                **tracker_values,
                "_cycle": cycle,
                "_tenure": tenure,
            }

            if psm_ctx is not None:
                # Patient-level PSM node: role (and so cost/effect) is a
                # deterministic function of elapsed time since entry --
                # no probability resolution/sampling needed this cycle.
                psm_node = psm_nodes_by_id[psm_ctx["node_id"]]
                psm_role = _psm_role_at(cycle - psm_ctx["entry_cycle"], psm_ctx["pfs_cycle"], psm_ctx["os_cycle"])
                psm_role_outcomes = psm_node.roles[psm_role]
                cost = _resolve(psm_role_outcomes.cost, resolved_parameters, ctx_vars, distributions, tables)
                effect = _resolve(psm_role_outcomes.effect, resolved_parameters, ctx_vars, distributions, tables)
            else:
                outgoing = outgoing_for_state(state, ctx_vars)

                current = states_by_id[state]
                if not current.tenure_applied:
                    cost = _resolve(current.cost, resolved_parameters, ctx_vars, distributions, tables)
                else:
                    initial_cost = _resolve(current.initial_cost, resolved_parameters, ctx_vars, distributions, tables)
                    recurring_cost = _resolve(
                        current.recurring_cost if current.recurring_cost is not None else current.cost,
                        resolved_parameters, ctx_vars, distributions, tables,
                    )
                    terminal_cost = _resolve(current.terminal_cost, resolved_parameters, ctx_vars, distributions, tables)
                    exit_probability = sum(
                        p for to_state, p in _collapsed_probabilities(state, ctx_vars).items() if to_state != state
                    )
                    base_cost = initial_cost if tenure == 1 else recurring_cost * (1.0 - exit_probability)
                    cost = base_cost + (0.0 if tenure == 1 else terminal_cost * exit_probability)
                effect = _resolve(current.effect, resolved_parameters, ctx_vars, distributions, tables)

            if psm_ctx is not None:
                new_state = f"{psm_node.id}::{psm_role}"
            else:
                sampled_target = _choose_target(outgoing, patient_rng)
                if has_path_rewards:
                    # Follow any non-real branch/terminal nodes down to the
                    # real health state this cycle ends in, and add the
                    # transitions' one-time Incre_$/Incre_E to THIS cycle's
                    # cost/effect (same cycle the Markov cohort engine
                    # charges them in).
                    sampled_target, path_cost, path_effect = _follow_path(state, sampled_target, ctx_vars, patient_rng)
                    cost += path_cost
                    effect += path_effect
                if sampled_target in psm_nodes_by_id:
                    # Entering a PSM node for the first time: sample this
                    # patient's own pfs_time/os_time now (their own RNG
                    # substream, their own covariate context) and record
                    # the resulting *role*, not the bare node id, as the
                    # transition target -- the node itself is never an
                    # occupiable state, only its 3 roles are.
                    entry_node = psm_nodes_by_id[sampled_target]
                    pfs_cycle, os_cycle = _sample_psm_entry_times(
                        entry_node, patient_rng, resolved_parameters, ctx_vars, distributions, tables, cycles_per_year,
                    )
                    initial_role = _psm_role_at(0.0, pfs_cycle, os_cycle)
                    new_state = f"{entry_node.id}::{initial_role}"
                    psm_ctx = {
                        "node_id": entry_node.id, "pfs_cycle": pfs_cycle, "os_cycle": os_cycle,
                        "entry_cycle": cycle + 1,
                    }
                else:
                    new_state = sampled_target
            discounted_cost = cost / (1.0 + discount_cost_per_cycle) ** cycle
            discounted_effect = effect / (1.0 + discount_effect_per_cycle) ** cycle
            cost_sum_per_cycle[cycle] += discounted_cost
            effect_sum_per_cycle[cycle] += discounted_effect
            raw_cost_sum_per_cycle[cycle] += cost
            raw_effect_sum_per_cycle[cycle] += effect
            patient_total_cost += discounted_cost
            patient_total_effect += discounted_effect
            if cycle == 0:
                first_cycle_cost = discounted_cost
                first_cycle_effect = discounted_effect

            transition_sum[cycle][state][new_state] += 1.0
            entering_new_state = new_state != state
            tenure_this_cycle = tenure

            new_tracker_values: dict[str, float] = {}
            for tracker in model.trackers:
                formula = tracker.update_formula if tracker.update_formula is not None else tracker.name
                new_tracker_values[tracker.name] = _resolve(
                    formula, resolved_parameters, ctx_vars, distributions, tables,
                )

            for tracker in model.trackers:
                if reset_on_entry:
                    if entering_new_state and new_state in tracker.reset_rules:
                        new_tracker_values[tracker.name] = _resolve(
                            tracker.reset_rules[new_state], resolved_parameters,
                            {**ctx_vars, **new_tracker_values}, distributions, tables,
                        )
                else:
                    if not entering_new_state and tenure_this_cycle == 1 and state in tracker.reset_rules:
                        new_tracker_values[tracker.name] = _resolve(
                            tracker.reset_rules[state], resolved_parameters,
                            {**ctx_vars, **new_tracker_values}, distributions, tables,
                        )

            tracker_values = new_tracker_values
            if entering_new_state:
                state = new_state
                tenure = 1
            else:
                tenure += 1

            occupancy_sum[cycle + 1][state] += 1.0
            if patient_index < trace_sample_size:
                sample_patient_traces[patient_index].append(state)

        if model.half_cycle_correction and n_cycles > 0:
            final_ctx = {
                **patient_variables,
                **tracker_values,
                "_cycle": n_cycles,
                "_tenure": tenure,
            }
            if psm_ctx is not None:
                final_node = psm_nodes_by_id[psm_ctx["node_id"]]
                final_role = _psm_role_at(n_cycles - psm_ctx["entry_cycle"], psm_ctx["pfs_cycle"], psm_ctx["os_cycle"])
                final_role_outcomes = final_node.roles[final_role]
                final_cost = _resolve(final_role_outcomes.cost, resolved_parameters, final_ctx, distributions, tables)
                final_effect = _resolve(final_role_outcomes.effect, resolved_parameters, final_ctx, distributions, tables)
            else:
                final_state = states_by_id[state]
                if not final_state.tenure_applied:
                    final_cost = _resolve(final_state.cost, resolved_parameters, final_ctx, distributions, tables)
                else:
                    final_cost = _resolve(
                        final_state.initial_cost
                        if tenure == 1
                        else (final_state.recurring_cost if final_state.recurring_cost is not None else final_state.cost),
                        resolved_parameters, final_ctx, distributions, tables,
                    )
                final_effect = _resolve(final_state.effect, resolved_parameters, final_ctx, distributions, tables)
            disc_final_cost = final_cost / (1.0 + discount_cost_per_cycle) ** n_cycles
            disc_final_effect = final_effect / (1.0 + discount_effect_per_cycle) ** n_cycles
            patient_total_cost += 0.5 * (disc_final_cost - first_cycle_cost)
            patient_total_effect += 0.5 * (disc_final_effect - first_cycle_effect)

        patient_costs.append(patient_total_cost)
        patient_effects.append(patient_total_effect)
        for tracker in model.trackers:
            if tracker.include_in_output:
                tracker_finals[tracker.name].append(tracker_values.get(tracker.name, 0.0))

    trace = [
        {state_id: occupancy_sum[cycle][state_id] / n_patients for state_id in all_ids}
        for cycle in range(n_cycles + 1)
    ]
    transition_counts = [
        {
            from_state: {to_state: count / n_patients for to_state, count in tos.items()}
            for from_state, tos in transition_sum[cycle].items()
        }
        for cycle in range(n_cycles)
    ]
    cost_per_cycle = [total / n_patients for total in cost_sum_per_cycle]
    effect_per_cycle = [total / n_patients for total in effect_sum_per_cycle]
    raw_cost_per_cycle = [total / n_patients for total in raw_cost_sum_per_cycle]
    raw_effect_per_cycle = [total / n_patients for total in raw_effect_sum_per_cycle]

    tracker_summary: dict[str, dict[str, float]] = {}
    for name, values in tracker_finals.items():
        if not values:
            continue
        array = np.array(values, dtype=float)
        tracker_summary[name] = {
            "mean": float(array.mean()),
            "p10": float(np.percentile(array, 10)),
            "p50": float(np.percentile(array, 50)),
            "p90": float(np.percentile(array, 90)),
        }

    return MicrosimResult(
        total_cost=float(np.mean(patient_costs)) if patient_costs else 0.0,
        total_effect=float(np.mean(patient_effects)) if patient_effects else 0.0,
        cost_per_cycle=cost_per_cycle,
        effect_per_cycle=effect_per_cycle,
        raw_cost_per_cycle=raw_cost_per_cycle,
        raw_effect_per_cycle=raw_effect_per_cycle,
        trace=trace,
        transition_counts=transition_counts,
        patient_costs=patient_costs,
        patient_effects=patient_effects,
        tracker_summary=tracker_summary,
        sample_patient_traces=sample_patient_traces,
    )
