"""Cohort simulation for Markov models."""

from __future__ import annotations

import ast
import operator
import re
from collections.abc import Mapping
from typing import Any

import numpy as np

from .markov_models import MarkovModel, MarkovResult, MarkovTransition
from .markov_validation import validate_markov
from .models import NumericRef, Param, ValidationError
from .expressions import _Evaluator, _parse_expression, evaluate_expression

# Early-stop threshold for run_markov()'s "cohort has fully converged
# into permanently zero-reward absorbing states" check (see its own
# comment at the break site). 1e-9 of remaining cohort mass, times
# whatever cycles are left, times a per-cycle reward, is the absolute
# worst-case error this introduces into total_cost/total_effect: for a
# model with a 1,000,000-unit per-cycle cost and 1,000 cycles left
# when the threshold is crossed, that is at most 1e-9 * 1000 * 1e6 =
# 1e-3 -- a millionth of a typical total, far below any reporting
# precision (2-4 significant figures) a HEOR result is ever quoted to.
# Mass only decreases from here (structurally_absorbing_ids states
# never emit it back out), so this is a true worst case, not typical.
_EARLY_STOP_MASS_TOLERANCE = 1e-9
_CYCLE_PATTERN = re.compile(r"\b_cycle\b")
_TENURE_PATTERN = re.compile(r"\b_tenure\b")


# Set False (tests only) to run the original per-(state, tenure) reward loop instead of the numpy fast path.
FAST_REWARD = True

def _engine_variables(
    cycle: int,
    tenure: int = 1,
    variables: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {**(variables or {}), "_cycle": cycle, "_tenure": tenure}


def _resolve(
    value: NumericRef,
    parameters: Mapping[str, Param],
    variables: Mapping[str, Any] | None = None,
    distributions: Mapping[str, Mapping[str, Any]] | None = None,
    tables: Mapping[str, Mapping[str, Any]] | None = None,
) -> float:
    if isinstance(value, str):
        return evaluate_expression(value, parameters, variables, distributions, tables)
    return float(value)


def _resolve_probability_group(
    entries: list[tuple[str, NumericRef]],
    parameters: Mapping[str, Param],
    variables: Mapping[str, Any] | None = None,
    distributions: Mapping[str, Mapping[str, Any]] | None = None,
    tables: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, float]:
    """Resolve a group of probabilities that may include one '#'/'C' complement."""

    from .models import is_complement

    resolved: dict[str, float] = {}
    complement_key: str | None = None
    known_sum = 0.0
    for key, value in entries:
        if is_complement(value):
            complement_key = key
            continue
        probability = _resolve(value, parameters, variables, distributions, tables)
        resolved[key] = probability
        known_sum += probability
    if complement_key is not None:
        resolved[complement_key] = 1.0 - known_sum
    return resolved


def validate_markov_only(
    model: MarkovModel,
    parameters: Mapping[str, Param] | None = None,
    allow_negative_cost: bool = False,
    variables: Mapping[str, Any] | None = None,
    distributions: Mapping[str, Mapping[str, Any]] | None = None,
    tables: Mapping[str, Mapping[str, Any]] | None = None,
):
    """Run ONLY the validation step run_markov() performs before it
    simulates (same call, same arguments), and return its error list.

    The canvas needs the blocking errors of every strategy on every
    redraw, but never the simulation output. Getting them through
    run_markov() meant a full cohort simulation (seconds to minutes on a
    600-cycle, tenure-tracked model) just to highlight a node -- on every
    edit, including a change to 'Cycles'. Validation evaluates each
    expression once, so this is effectively instant.
    """

    return validate_markov(
        model,
        parameters or {},
        allow_negative_cost,
        variables=_engine_variables(0, variables=variables),
        distributions=distributions,
        tables=tables,
    )


def run_markov(
    model: MarkovModel,
    parameters: Mapping[str, Param] | None = None,
    discount_cost_per_cycle: float = 0.0,
    discount_effect_per_cycle: float = 0.0,
    allow_negative_cost: bool = False,
    variables: Mapping[str, Any] | None = None,
    distributions: Mapping[str, Mapping[str, Any]] | None = None,
    tables: Mapping[str, Mapping[str, Any]] | None = None,
) -> MarkovResult:
    """Validate and run a cohort simulation over the model's cycles.

    Rewards (cost/effect) are accrued for the distribution occupying each
    REAL state (MarkovState.is_real_state -- see its docstring) at the
    *start* of a cycle; the transition matrix then advances the cohort to
    the next cycle's distribution over real states. Discounting uses a
    per-cycle rate applied as (1 + rate) ** -t. When half_cycle_correction
    is set on the model, the trapezoidal correction (Briggs/Sculpher/
    Claxton) is applied: subtract half of cycle 0's reward and add half of
    an extra reward evaluated at the distribution after the final
    transition.

    A non-real state (a canvas-created child of another state -- an
    adverse-event branch, a decision-style sub-split of an existing
    pathway -- rather than a health state the cohort spends a whole cycle
    occupying) never appears in the simulated distribution or the trace:
    reaching one costs no simulated time. Whenever a real state's outgoing
    transitions lead into a chain of one or more non-real states, that
    whole chain is resolved in zero elapsed cycles, in the SAME cycle as
    the real state it started from -- multiplying probabilities and
    summing cost_increment/effect_increment/extra_increment (and each
    non-real state's own cost/effect/extra_outcomes, fired once as the
    cohort passes through, never as a recurring per-cycle charge) along
    the way -- until it lands on the next real state. A plain real ->
    real transition (the overwhelmingly common case, and the only case
    before MarkovState.is_real_state existed) is unaffected: it resolves
    in exactly one hop, identical to previous behavior.

    Extra Cost/Effect outcomes (Outcomes table rows beyond the primary
    pair, e.g. a second Effect "LY") are accrued the same way as the
    primary effect -- a flat per-cycle rate off MarkovState.extra_outcomes
    for a real state (or a one-off amount for a non-real one, per above),
    with no tenure initial/recurring/terminal split (see the field's own
    docstring) -- plus a per-transition MarkovTransition.extra_increment,
    mirroring cost_increment/effect_increment. Each extra outcome is
    discounted at discount_cost_per_cycle if its Outcomes-table type is
    "cost", or discount_effect_per_cycle if "effect" (mirrors
    abm_engine.py's extra_outcomes convention). model.extra_outcomes is
    empty unless the document's Outcomes table has such a row, so this is
    a no-op for every existing model.
    """

    # --- Speed: share ONE expression evaluator per (cycle, tenure) --------
    # The module-level _resolve() builds a brand-new _Evaluator for every
    # single expression, so its per-evaluator variable cache was thrown
    # away each time: every state cost / transition probability re-ran the
    # whole variable chain (incl. constructing a scipy distribution object
    # for DistToProb) from scratch. Within one (cycle, tenure) pair the
    # variable namespace ({**variables, _cycle, _tenure}) is identical for
    # every expression, so sharing the evaluator is numerically identical
    # -- each variable is just computed once per (cycle, tenure) instead of
    # once per expression. Entries are dropped when the cycle advances.
    #
    # --- Speed (2026-10): tiered sharing across cycles / tenures ------------
    # An expression (following variable / distribution references
    # transitively) that cannot see `_cycle` evaluates identically in every
    # cycle, and one that cannot see `_tenure` identically for every
    # tenure. Each evaluator is therefore keyed by ONLY the dimensions its
    # expression actually depends on -- (None, None) for a constant formula,
    # (cycle, None) for time-varying but tenure-free ones, (None, tenure)
    # for tenure-only ones, (cycle, tenure) otherwise -- so a value is
    # computed once and reused, with the exact same arithmetic as before
    # (numerically identical). Final expression values are memoised the same
    # way. Disabled (old per-(cycle, tenure) behaviour) whenever the model
    # uses a stochastic function (DistSample...), whose draws must not be
    # shared across cycles.
    _evaluator_cache: dict[tuple[Any, Any], _Evaluator] = {}
    _result_cache: dict[tuple[Any, Any, str], float] = {}
    _stochastic_tokens = ("DistSample", "DistSampleCond", "DistToTime", "TableToTime", "DistTrim")
    _model_text = repr(model) + repr(variables) + repr(distributions)
    _share_across_dims = not any(token in _model_text for token in _stochastic_tokens)
    _dep_token_re = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
    _name_dep_memo: dict[str, tuple[bool, bool]] = {}
    _expr_dep_memo: dict[str, tuple[bool, bool]] = {}

    def _name_deps(name: str, stack: frozenset[str]) -> tuple[bool, bool]:
        if name == "_cycle":
            return (True, False)
        if name == "_tenure":
            return (False, True)
        memo = _name_dep_memo.get(name)
        if memo is not None:
            return memo
        if name in stack:
            return (False, False)
        cyc = ten = False
        if variables is not None and name in variables:
            cyc, ten = _expr_deps(variables[name], stack | {name})
        if distributions is not None and name in distributions:
            for v in (distributions[name].get("parameters") or {}).values():
                c2, t2 = _expr_deps(v, stack | {name})
                cyc = cyc or c2
                ten = ten or t2
        _name_dep_memo[name] = (cyc, ten)
        return (cyc, ten)

    def _expr_deps(value: Any, stack: frozenset[str] = frozenset()) -> tuple[bool, bool]:
        if not isinstance(value, str):
            return (False, False)
        cyc = ten = False
        for name in _dep_token_re.findall(value):
            c2, t2 = _name_deps(name, stack)
            cyc = cyc or c2
            ten = ten or t2
            if cyc and ten:
                break
        return (cyc, ten)

    # --- Speed: tenure-vectorised evaluation ---------------------------------
    # An expression that DOES depend on _tenure (e.g. cost = 'if(_tenure==1,
    # a, b) + cSub') used to be re-interpreted once per (cycle, tenure) --
    # hundreds of thousands of AST walks on a tenure-tracked model. Here it
    # is evaluated ONCE per cycle for every tenure 1..T at once with numpy.
    # Only +, -, *, /, unary +/-, single comparisons and if(...) are
    # vectorised (IEEE float64 element-wise ops == Python float ops, so
    # values are bit-identical); any tenure-free sub-expression is evaluated
    # by the ordinary scalar evaluator. Anything else, any exception and any
    # non-finite result silently falls back to the old scalar path.
    _tenure_len = max([1] + [max(1, st.tenure_count) for st in model.states if st.is_real_state])
    _tenure_axis = np.arange(1, _tenure_len + 1, dtype=float)
    _fast_reward_ok = FAST_REWARD  # module switch: False forces the per-entry reward loop (parity tests)
    _vec_cache: dict[tuple[Any, str], Any] = {}
    _vec_var_cache: dict[tuple[Any, str], Any] = {}
    _node_info: dict[int, tuple[str, tuple[bool, bool], ast.AST]] = {}
    _vec_binops = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}
    _vec_cmps = {
        ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt,
        ast.LtE: operator.le, ast.Gt: operator.gt, ast.GtE: operator.ge,
    }

    class _NoVec(Exception):
        pass

    def _vec_numeric(x):
        if isinstance(x, np.ndarray):
            if x.dtype != np.float64:
                raise _NoVec
            return x
        if type(x) is float:
            return x
        raise _NoVec

    def _vec_variable(name: str, cycle: int):
        cyc_dep = _name_deps(name, frozenset())[0]
        key = (cycle if cyc_dep else None, name)
        cached = _vec_var_cache.get(key)
        if cached is not None:
            return cached
        definition = variables[name]
        if not isinstance(definition, str):
            raise _NoVec
        text = definition.strip()
        if not text:
            raise _NoVec
        if text in (parameters or {}):
            result = float(parameters[text].base)
        elif variables is not None and text in variables:
            result = _vec_variable(text, cycle)
        else:
            result = _vec_node(_parse_expression(text).body, cycle)
        _vec_var_cache[key] = result
        return result

    def _vec_node(node, cycle: int):
        info = _node_info.get(id(node))
        if info is None:
            text = ast.unparse(node)
            info = (text, _expr_deps(text), node)  # keep node alive: id() key
            _node_info[id(node)] = info
        text, deps, _keep = info
        if not deps[1]:
            return _resolve(text, resolved_parameters, _engine_variables(cycle, 1, variables), distributions, tables)
        if isinstance(node, ast.Name):
            name = node.id
            if name == "_tenure":
                return _tenure_axis
            if name in (parameters or {}):
                return float(parameters[name].base)
            if variables is not None and name in variables:
                return _vec_variable(name, cycle)
            raise _NoVec
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            operand = _vec_numeric(_vec_node(node.operand, cycle))
            return -operand if isinstance(node.op, ast.USub) else +operand
        if isinstance(node, ast.BinOp) and type(node.op) in _vec_binops:
            left = _vec_numeric(_vec_node(node.left, cycle))
            right = _vec_numeric(_vec_node(node.right, cycle))
            return _vec_binops[type(node.op)](left, right)
        if isinstance(node, ast.Compare) and len(node.ops) == 1 and type(node.ops[0]) in _vec_cmps:
            left = _vec_numeric(_vec_node(node.left, cycle))
            right = _vec_numeric(_vec_node(node.comparators[0], cycle))
            return _vec_cmps[type(node.ops[0])](left, right)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "if_"
            and len(node.args) == 3
            and not node.keywords
        ):
            cond = _vec_node(node.args[0], cycle)
            if isinstance(cond, np.ndarray):
                if cond.dtype != np.bool_:
                    raise _NoVec
                a = _vec_numeric(_vec_node(node.args[1], cycle))
                b = _vec_numeric(_vec_node(node.args[2], cycle))
                return np.where(cond, a, b)
            if isinstance(cond, (bool, np.bool_, float)):
                return _vec_node(node.args[1] if bool(cond) else node.args[2], cycle)
            raise _NoVec
        raise _NoVec

    def _tenure_vector(value: str, deps: tuple[bool, bool], cycle: int):
        key = (cycle if deps[0] else None, value)
        vec = _vec_cache.get(key)
        if vec is None:
            vec = False
            try:
                with np.errstate(all="ignore"):
                    text = value.strip()
                    if not text:
                        raise _NoVec
                    if text in (parameters or {}):
                        raise _NoVec
                    if variables is not None and text in variables:
                        res = _vec_variable(text, cycle)
                    else:
                        res = _vec_node(_parse_expression(text).body, cycle)
                    if type(res) is float:
                        if res == res and abs(res) != float("inf"):
                            vec = [res] * _tenure_len
                    elif isinstance(res, np.ndarray) and res.dtype == np.float64 and res.shape == (_tenure_len,):
                        if np.isfinite(res).all():
                            vec = res.tolist()
            except Exception:  # any problem -> scalar fallback (identical to old behaviour)
                vec = False
            _vec_cache[key] = vec
        return vec

    def _resolve(value, parameters_, variables_=None, distributions_=None, tables_=None):
        if isinstance(value, str):
            if _share_across_dims:
                deps = _expr_dep_memo.get(value)
                if deps is None:
                    deps = _expr_deps(value)
                    _expr_dep_memo[value] = deps
                key = (
                    variables_["_cycle"] if deps[0] else None,
                    variables_["_tenure"] if deps[1] else None,
                )
            else:
                key = (variables_["_cycle"], variables_["_tenure"])
            result_key = (key[0], key[1], value)
            cached = _result_cache.get(result_key)
            if cached is not None:
                return cached
            if _share_across_dims and deps[1] and key[1] <= _tenure_len:
                vec = _tenure_vector(value, deps, variables_["_cycle"])
                if vec:
                    result = vec[key[1] - 1]
                    _result_cache[result_key] = result
                    return result
            evaluator = _evaluator_cache.get(key)
            if evaluator is None:
                evaluator = _Evaluator(parameters_, variables_, distributions_, tables_)
                _evaluator_cache[key] = evaluator
            result = evaluator.evaluate(value)
            _result_cache[result_key] = result
            return result
        return float(value)

    def _resolve_probability_group(entries, parameters_, variables_=None, distributions_=None, tables_=None):
        from .models import is_complement

        resolved: dict[str, float] = {}
        complement_key: str | None = None
        known_sum = 0.0
        for key, value in entries:
            if is_complement(value):
                complement_key = key
                continue
            probability = _resolve(value, parameters_, variables_, distributions_, tables_)
            resolved[key] = probability
            known_sum += probability
        if complement_key is not None:
            resolved[complement_key] = 1.0 - known_sum
        return resolved

    initial_variables = _engine_variables(0, variables=variables)
    resolved_parameters = parameters or {}
    errors = validate_markov(
        model,
        resolved_parameters,
        allow_negative_cost,
        variables=initial_variables,
        distributions=distributions,
        tables=tables,
    )
    if errors:
        return MarkovResult(errors=errors)

    states = model.states
    state_ids = [state.id for state in states]
    real_state_ids = [state.id for state in states if state.is_real_state]
    states_by_id = {state.id: state for state in states}
    # Speed (vectorized advance -- see advance()'s own docstring): a
    # fixed integer index per real state, and each one's tenure-array
    # length, so the cohort's mass can be represented as one numpy
    # array per state instead of a Python dict keyed by (state_id,
    # tenure). tenure_count defaults to 1 (MarkovState.tenure_count)
    # even for a state where tenure_applied is False, matching
    # advance()'s own `max(1, ...)` floor below.
    real_state_index = {state_id: i for i, state_id in enumerate(real_state_ids)}
    real_state_tenure_counts = [
        max(1, states_by_id[state_id].tenure_count) for state_id in real_state_ids
    ]
    # Fallback for resolving a non-real dead-end leaf into the real state
    # it represents when it has no explicit alias_of (models saved before
    # that field existed): match by display name, since "relabel as
    # terminal" always copies the reference state's name onto the leaf.
    real_state_by_name = {state.name: state.id for state in states if state.is_real_state}
    initial_distribution = _resolve_probability_group(
        [(state.id, state.initial_prob) for state in states if state.is_real_state],
        resolved_parameters,
        initial_variables,
        distributions,
        tables,
    )
    distribution = {(state_id, 1): probability for state_id, probability in initial_distribution.items()}

    grouped: dict[str, list[tuple[str, NumericRef]]] = {state_id: [] for state_id in state_ids}
    transition_by_pair: dict[tuple[str, str], MarkovTransition] = {}
    for transition in model.transitions:
        grouped[transition.from_state].append((transition.to_state, transition.probability))
        transition_by_pair[(transition.from_state, transition.to_state)] = transition

    # Early-stop setup (TreeAge/Amua finish in seconds on a model
    # that takes this engine minutes -- once age 100 is reached the
    # cohort is 100% dead and every later cycle would add nothing, so
    # there is no reason to keep simulating out to cycles=1000).
    #
    # structurally_absorbing_ids: real states with NO declared outgoing
    # transitions at all -- outgoing_for_state()'s only fallback for
    # these is the hardcoded {state_id: 1.0} self-loop, never anything
    # expression-derived, so once the cohort is inside one of these it
    # can NEVER leave, for any cycle/tenure, independent of every other
    # transition's probability formula elsewhere in the model.
    #
    # _reward_fields_reference_dynamic_var: True if ANY state's cost/
    # effect/initial_cost/recurring_cost/terminal_cost/extra_outcomes,
    # or any transition's cost_increment/effect_increment/
    # extra_increment, textually references _cycle or _tenure anywhere
    # in the whole model. When True, early-stop is disabled entirely
    # (falls back to simulating every cycle, exactly as before) because
    # a reward that is 0 this cycle is not guaranteed to stay 0 later.
    # When False, every reward field is either a plain number or an
    # expression whose value cannot depend on which cycle/tenure it is
    # evaluated at, so "0 this cycle" means "0 forever".
    def _mentions_dynamic_variable(value: Any) -> bool:
        # Word-boundary match, not a plain substring search: a
        # parameter can legitimately be named e.g. "cPF_cycle1" (cost
        # of PF, cycle 1) without ever referencing the engine's own
        # dynamic _cycle variable -- "_cycle" in "cPF_cycle1" would be
        # a false positive that disables early-stop for every model
        # with a parameter name like that (found on a real
        # model). _CYCLE/_TENURE_PATTERN require _cycle/_tenure to
        # stand alone as their own identifier.
        return isinstance(value, str) and bool(
            _CYCLE_PATTERN.search(value) or _TENURE_PATTERN.search(value)
        )

    _reward_fields_reference_dynamic_var = False
    for _state in states:
        for _value in (
            _state.cost, _state.effect, _state.initial_cost,
            _state.recurring_cost, _state.terminal_cost,
        ):
            if _mentions_dynamic_variable(_value):
                _reward_fields_reference_dynamic_var = True
                break
        if _reward_fields_reference_dynamic_var:
            break
        for _rate in _state.extra_outcomes.values():
            if _mentions_dynamic_variable(_rate):
                _reward_fields_reference_dynamic_var = True
                break
        if _reward_fields_reference_dynamic_var:
            break
    if not _reward_fields_reference_dynamic_var:
        for _transition in model.transitions:
            if _mentions_dynamic_variable(_transition.cost_increment) or _mentions_dynamic_variable(
                _transition.effect_increment
            ):
                _reward_fields_reference_dynamic_var = True
                break
            for _value in _transition.extra_increment.values():
                if _mentions_dynamic_variable(_value):
                    _reward_fields_reference_dynamic_var = True
                    break
            if _reward_fields_reference_dynamic_var:
                break

    structurally_absorbing_ids = {
        state_id for state_id in real_state_ids if not grouped[state_id]
    }
    early_stop_eligible = (
        not _reward_fields_reference_dynamic_var and bool(structurally_absorbing_ids)
    )

    extra_outcome_ids = [row["id"] for row in model.extra_outcomes]
    extra_outcome_types = {row["id"]: row["type"] for row in model.extra_outcomes}
    extra_discount_per_cycle = {
        oid: (discount_cost_per_cycle if extra_outcome_types[oid] == "cost" else discount_effect_per_cycle)
        for oid in extra_outcome_ids
    }

    # --- Speed: collapse the tenure dimension of the TRANSITION walk -------
    # _collapse(state, cycle, tenure) only uses `tenure` to fill the
    # `_tenure` variable while evaluating transition probabilities and
    # cost/effect increments. If none of those expressions (following
    # variable references transitively) can see `_tenure`, the result is
    # identical for every tenure, so compute it once per (state, cycle)
    # instead of once per (state, cycle, tenure) -- with tenure_count=1000
    # that was up to ~1000x redundant work per cycle. Reward expressions
    # (state cost/effect) are NOT touched: they still see the real tenure.
    import re as _re

    _token_re = _re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
    _tenure_dep_memo: dict[str, bool] = {}

    def _expr_sees_tenure(value: Any, _stack: frozenset[str] = frozenset()) -> bool:
        if not isinstance(value, str):
            return False
        for name in _token_re.findall(value):
            if name == "_tenure":
                return True
            if name in _stack:
                continue
            cached_dep = _tenure_dep_memo.get(name)
            if cached_dep is None:
                cached_dep = False
                if variables is not None and name in variables:
                    cached_dep = _expr_sees_tenure(variables[name], _stack | {name})
                elif distributions is not None and name in distributions:
                    cached_dep = any(
                        _expr_sees_tenure(v, _stack | {name})
                        for v in (distributions[name].get("parameters") or {}).values()
                    )
                _tenure_dep_memo[name] = cached_dep
            if cached_dep:
                return True
        return False

    _transitions_see_tenure = False
    for _transition in model.transitions:
        if (
            _expr_sees_tenure(_transition.probability)
            or _expr_sees_tenure(_transition.cost_increment)
            or _expr_sees_tenure(_transition.effect_increment)
            or any(_expr_sees_tenure(v) for v in _transition.extra_increment.values())
        ):
            _transitions_see_tenure = True
            break
    if not _transitions_see_tenure:
        for _state in states:
            if not _state.is_real_state and (
                _expr_sees_tenure(_state.cost)
                or _expr_sees_tenure(_state.effect)
                or any(_expr_sees_tenure(v) for v in _state.extra_outcomes.values())
            ):
                _transitions_see_tenure = True
                break

    def outgoing_for_state(
        state_id: str, cycle: int, tenure: int
    ) -> dict[str, float]:
        entries = grouped[state_id]
        if not entries:
            # Absorbing state with no declared transitions: implicit self-loop.
            return {state_id: 1.0}
        return _resolve_probability_group(
            entries,
            resolved_parameters,
            _engine_variables(cycle, tenure, variables),
            distributions,
            tables,
        )

    # {(state_id, cycle, tenure): {to_real_state_id: (probability,
    # cost_sum, effect_sum, extra_sum)}} -- see _collapse()'s docstring.
    # reward() and advance() each need the same collapse for the same
    # (state_id, cycle, tenure), so this avoids doing the walk twice.
    _collapse_cache: dict[tuple[str, int, int], dict[str, tuple[float, float, float, dict[str, float]]]] = {}

    def _collapse(
        state_id: str, cycle: int, tenure: int
    ) -> dict[str, tuple[float, float, float, dict[str, float]]]:
        """Resolve state_id's one-cycle outgoing transitions all the way
        through any chain of non-real (canvas-child) states, in zero
        elapsed simulation time, landing only on real states. Returns
        {to_real_state_id: (probability, cost_sum, effect_sum, extra_sum)}
        where, for every path from state_id to that real destination this
        cycle: probability is the summed conditional probability of
        taking it, and cost_sum/effect_sum/extra_sum are each the SUM
        over those paths of (that path's own conditional probability *
        that path's cumulative one-time cost/effect/extra) -- i.e.
        already the exact quantity to multiply by this state's current
        mass and add straight into a cost/effect total, with no further
        multiplication by probability needed. A single real -> real hop
        degenerates to exactly {to_state: (probability, probability *
        cost_increment, probability * effect_increment, ...)} -- the
        same value the old one-hop-only version of this engine used."""

        if not _transitions_see_tenure:
            tenure = 1  # result provably identical for every tenure
        cache_key = (state_id, cycle, tenure)
        cached = _collapse_cache.get(cache_key)
        if cached is not None:
            return cached

        state_variables = _engine_variables(cycle, tenure, variables)
        accumulated: dict[str, list[Any]] = {}

        def walk(
            current_id: str,
            prob_so_far: float,
            cost_so_far: float,
            effect_so_far: float,
            extra_so_far: dict[str, float],
            visited: frozenset[str],
        ) -> None:
            for to_state, probability in outgoing_for_state(current_id, cycle, tenure).items():
                if probability == 0.0:
                    continue
                path_prob = prob_so_far * probability
                path_cost = cost_so_far
                path_effect = effect_so_far
                path_extra = extra_so_far
                transition = transition_by_pair.get((current_id, to_state))
                if transition is not None:
                    if transition.cost_increment:
                        path_cost += _resolve(
                            transition.cost_increment, resolved_parameters, state_variables, distributions, tables
                        )
                    if transition.effect_increment:
                        path_effect += _resolve(
                            transition.effect_increment, resolved_parameters, state_variables, distributions, tables
                        )
                    if transition.extra_increment:
                        path_extra = dict(path_extra)
                        for oid, incre_value in transition.extra_increment.items():
                            if incre_value in (None, 0, 0.0):
                                continue
                            path_extra[oid] = path_extra.get(oid, 0.0) + _resolve(
                                incre_value, resolved_parameters, state_variables, distributions, tables
                            )

                to_state_obj = states_by_id[to_state]
                if to_state_obj.is_real_state:
                    entry = accumulated.get(to_state)
                    if entry is None:
                        entry = [0.0, 0.0, 0.0, {}]
                        accumulated[to_state] = entry
                    entry[0] += path_prob
                    entry[1] += path_prob * path_cost
                    entry[2] += path_prob * path_effect
                    for oid, value in path_extra.items():
                        entry[3][oid] = entry[3].get(oid, 0.0) + path_prob * value
                    continue

                # to_state is a non-real (canvas-child) state: no elapsed
                # cycle to get here, so keep walking its own outgoing
                # transitions within this SAME cycle. Its own cost/effect/
                # extra_outcomes (if any) fire once, right now, same as a
                # transition's own increment above -- see
                # MarkovState.is_real_state's docstring.
                if to_state in visited:
                    raise ValueError(
                        f"Markov model has a loop entirely among non-real (canvas-child) "
                        f"states -- '{to_state}' is reached again without ever passing "
                        f"through a real state. A branch/decision chain must eventually "
                        f"reach a real state."
                    )
                node_cost = path_cost
                node_effect = path_effect
                node_extra = path_extra
                if to_state_obj.cost:
                    node_cost = node_cost + _resolve(
                        to_state_obj.cost, resolved_parameters, state_variables, distributions, tables
                    )
                if to_state_obj.effect:
                    node_effect = node_effect + _resolve(
                        to_state_obj.effect, resolved_parameters, state_variables, distributions, tables
                    )
                if to_state_obj.extra_outcomes:
                    node_extra = dict(node_extra)
                    for oid, rate in to_state_obj.extra_outcomes.items():
                        if rate in (None, 0, 0.0):
                            continue
                        node_extra[oid] = node_extra.get(oid, 0.0) + _resolve(
                            rate, resolved_parameters, state_variables, distributions, tables
                        )

                if not grouped[to_state]:
                    # Dead-end leaf: no declared outgoing transitions, and
                    # (per validate_markov()) a non-real state is never
                    # required to be flagged is_absorbing for this. Per
                    # the Markov principle, reaching a
                    # terminal-looking leaf just means the cohort already
                    # occupies whatever real state it represents -- so
                    # resolve it straight into that real state (zero
                    # further elapsed cycles) instead of recursing into
                    # outgoing_for_state's implicit self-loop, which would
                    # otherwise misfire the "loop among non-real states"
                    # check below on every single dead-end leaf.
                    alias_target = to_state_obj.alias_of or real_state_by_name.get(to_state_obj.name)
                    alias_state = states_by_id.get(alias_target) if alias_target else None
                    if alias_state is None or not alias_state.is_real_state:
                        raise ValueError(
                            f"State '{to_state}' has no outgoing transitions and doesn't "
                            f"match any real state by alias or name -- use 'relabel as "
                            f"terminal' on it to link it to the real state it represents, "
                            f"or give it its own outgoing transition(s)."
                        )
                    entry = accumulated.get(alias_target)
                    if entry is None:
                        entry = [0.0, 0.0, 0.0, {}]
                        accumulated[alias_target] = entry
                    entry[0] += path_prob
                    entry[1] += path_prob * node_cost
                    entry[2] += path_prob * node_effect
                    for oid, value in node_extra.items():
                        entry[3][oid] = entry[3].get(oid, 0.0) + path_prob * value
                    continue

                walk(to_state, path_prob, node_cost, node_effect, node_extra, visited | {to_state})

        walk(state_id, 1.0, 0.0, 0.0, {}, frozenset({state_id}))
        result = {
            to_state: (entry[0], entry[1], entry[2], entry[3])
            for to_state, entry in accumulated.items()
        }
        _collapse_cache[cache_key] = result
        return result

    def reward(
        dist: Mapping[tuple[str, int], float], cycle: int, apply_terminal: bool
    ) -> tuple[float, float, dict[str, float]]:
        total_cost = 0.0
        total_effect = 0.0
        total_extra = {oid: 0.0 for oid in extra_outcome_ids}
        # Speed: a reward field is resolved ONCE per (state, field, cycle)
        # into (0, scalar) for a tenure-independent field, (1, per-tenure
        # list) for a tenure-vectorised one (see _tenure_vector), or
        # (2, None) = "scalar fallback" (old per-entry evaluation). Built
        # lazily at the first non-zero entry of the state, in the same
        # field order as before, so errors surface in the same place.
        contexts: dict[str, list[Any]] = {}
        exit_cache: dict[int, float] = {}
        collapsed_by_state: dict[str, Any] = {}
        collapse_once = not _transitions_see_tenure

        def make(value: Any, tenure: int) -> tuple[int, Any]:
            if not isinstance(value, str):
                return (0, float(value))
            if not _share_across_dims:
                return (2, None)
            deps = _expr_dep_memo.get(value)
            if deps is None:
                deps = _expr_deps(value)
                _expr_dep_memo[value] = deps
            if not deps[1]:
                return (0, _resolve(
                    value, resolved_parameters,
                    _engine_variables(cycle, tenure, variables), distributions, tables,
                ))
            vec = _tenure_vector(value, deps, cycle) if tenure <= _tenure_len else False
            return (1, vec) if vec else (2, None)

        def slow(value: Any, tenure: int) -> float:
            return _resolve(
                value, resolved_parameters, _engine_variables(cycle, tenure, variables), distributions, tables
            )

        # Fast path (numpy over the tenure axis): when `dist` is the array the vectorised advance just built and
        # nothing per-patient/stochastic or per-tenure-in-transitions is involved, each state's reward is a dot
        # product of its tenure masses with its per-tenure cost/effect vector instead of one Python iteration per
        # (state, tenure) -- 1000 tenure buckets x 2 states x every cycle in the capivasertib model.  A state whose
        # fields cannot be vectorised (kind 2) falls back to the per-entry loop below.  Sums are reordered, so
        # totals agree with the per-entry loop to ~1e-15 relative, not bit for bit.
        only_states: set[str] | None = None
        if _fast_reward_ok and not extra_outcome_ids and collapse_once and dist is _built["dist"]:
            only_states = set()
            arr = np.asarray(_built["values"], dtype=float)

            def value_at(entry: tuple[int, Any], idx: Any, m_nz: Any) -> float:
                kind, data = entry
                if kind == 0:
                    return float(data) * float(m_nz.sum())
                return float(np.dot(m_nz, np.asarray(data, dtype=float)[idx]))

            def per_tenure(entry: tuple[int, Any], idx: Any) -> Any:
                kind, data = entry
                return float(data) if kind == 0 else np.asarray(data, dtype=float)[idx]

            for i, state_id in enumerate(real_state_ids):
                a, b = _state_slices[i]
                m = arr[a:b]
                idx = np.flatnonzero(m)
                if not len(idx):
                    continue
                m_nz = m[idx]
                first_tenure = int(idx[0]) + 1
                state = states_by_id[state_id]
                entries = [make(state.effect, first_tenure)]
                if not state.tenure_applied:
                    entries.append(make(state.cost, first_tenure))
                else:
                    recurring_value = state.recurring_cost if state.recurring_cost is not None else state.cost
                    entries.append(make(state.initial_cost, first_tenure))
                    entries.append(make(recurring_value, first_tenure))
                    if apply_terminal:
                        entries.append(make(state.terminal_cost, first_tenure))
                if any(kind == 2 or (kind == 1 and len(data) < b - a) for kind, data in entries):
                    only_states.add(state_id)
                    continue
                total_effect += value_at(entries[0], idx, m_nz)
                collapsed = _collapse(state_id, cycle, first_tenure) if apply_terminal else None
                if not state.tenure_applied:
                    total_cost += value_at(entries[1], idx, m_nz)
                else:
                    initial_entry, recurring_entry = entries[1], entries[2]
                    first_cost = float(initial_entry[1] if initial_entry[0] == 0 else initial_entry[1][0])
                    recurring = per_tenure(recurring_entry, idx)
                    if not apply_terminal:
                        cost_vec = np.broadcast_to(np.asarray(recurring, dtype=float), idx.shape).copy()
                    else:
                        exit_probability = sum(
                            probability for to_state, (probability, _c, _e, _x) in collapsed.items() if to_state != state_id
                        )
                        terminal = per_tenure(entries[3], idx)
                        cost_vec = np.asarray(recurring * (1.0 - exit_probability) + terminal * exit_probability, dtype=float)
                        cost_vec = np.broadcast_to(cost_vec, idx.shape).copy()
                    cost_vec[idx == 0] = first_cost
                    total_cost += float(np.dot(m_nz, cost_vec))
                if collapsed is not None:
                    mass_sum = float(m_nz.sum())
                    for _to_state, (_probability, cost_sum, effect_sum, _extra_sum) in collapsed.items():
                        if cost_sum:
                            total_cost += mass_sum * cost_sum
                        if effect_sum:
                            total_effect += mass_sum * effect_sum
            if not only_states:
                return total_cost, total_effect, total_extra

        current_id = None
        state = None
        ctx: list[Any] = []
        for (state_id, tenure), mass in dist.items():
            if only_states is not None and state_id not in only_states:
                continue
            if mass == 0.0:
                continue
            if state_id != current_id:
                current_id = state_id
                state = states_by_id[state_id]
                ctx = contexts.get(state_id)
                if ctx is None:
                    ctx = contexts[state_id] = [None, None, None, None, None]
            entry = ctx[0]
            if entry is None:
                entry = ctx[0] = make(state.effect, tenure)
            kind, data = entry
            total_effect += mass * (data if kind == 0 else data[tenure - 1] if kind == 1 else slow(state.effect, tenure))
            if extra_outcome_ids:
                state_variables = _engine_variables(cycle, tenure, variables)
                for oid in extra_outcome_ids:
                    rate = state.extra_outcomes.get(oid)
                    if rate is None:
                        continue
                    total_extra[oid] += mass * _resolve(
                        rate, resolved_parameters, state_variables, distributions, tables
                    )
            collapsed: dict[str, tuple[float, float, float, dict[str, float]]] | None = None
            if apply_terminal:
                if collapse_once:
                    collapsed = collapsed_by_state.get(state_id)
                    if collapsed is None:
                        collapsed = collapsed_by_state[state_id] = _collapse(state_id, cycle, tenure)
                else:
                    collapsed = _collapse(state_id, cycle, tenure)

            if not state.tenure_applied:
                entry = ctx[1]
                if entry is None:
                    entry = ctx[1] = make(state.cost, tenure)
                kind, data = entry
                total_cost += mass * (data if kind == 0 else data[tenure - 1] if kind == 1 else slow(state.cost, tenure))
            else:
                entry = ctx[2]
                if entry is None:
                    entry = ctx[2] = make(state.initial_cost, tenure)
                kind, data = entry
                initial_cost = data if kind == 0 else data[tenure - 1] if kind == 1 else slow(state.initial_cost, tenure)
                recurring_value = state.recurring_cost if state.recurring_cost is not None else state.cost
                entry = ctx[3]
                if entry is None:
                    entry = ctx[3] = make(recurring_value, tenure)
                kind, data = entry
                recurring_cost = data if kind == 0 else data[tenure - 1] if kind == 1 else slow(recurring_value, tenure)
                if not apply_terminal:
                    total_cost += mass * (initial_cost if tenure == 1 else recurring_cost)
                else:
                    entry = ctx[4]
                    if entry is None:
                        entry = ctx[4] = make(state.terminal_cost, tenure)
                    kind, data = entry
                    terminal_cost = data if kind == 0 else data[tenure - 1] if kind == 1 else slow(state.terminal_cost, tenure)
                    exit_probability = exit_cache.get(id(collapsed))
                    if exit_probability is None:
                        exit_probability = sum(
                            probability
                            for to_state, (probability, _cost, _effect, _extra) in collapsed.items()
                            if to_state != state_id
                        )
                        exit_cache[id(collapsed)] = exit_probability
                    base_cost = initial_cost if tenure == 1 else recurring_cost * (1.0 - exit_probability)
                    total_cost += mass * (base_cost + (0.0 if tenure == 1 else terminal_cost * exit_probability))

            # Per-destination one-time reward (see _collapse()). Only when a
            # transition is actually occurring.
            if collapsed is not None:
                for to_state, (_probability, cost_sum, effect_sum, extra_sum) in collapsed.items():
                    if cost_sum:
                        total_cost += mass * cost_sum
                    if effect_sum:
                        total_effect += mass * effect_sum
                    for oid, value in extra_sum.items():
                        if oid not in total_extra or value in (None, 0, 0.0):
                            continue
                        total_extra[oid] += mass * value
        return total_cost, total_effect, total_extra

    def _advance_scalar(
        dist: Mapping[tuple[str, int], float], cycle: int
    ) -> dict[tuple[str, int], float]:
        """Original, per-(state, tenure) Python-loop implementation --
        the only correct path when transition probabilities (or the
        cost/effect of any non-real state along the way) actually vary
        by tenure (_transitions_see_tenure True), since then the
        collapsed result out of (from_state, tenure) genuinely differs
        across tenure and can't be captured by one shared matrix. See
        _advance_vectorized below for the fast path used otherwise."""
        next_dist = {
            (state_id, tenure): 0.0
            for state_id in real_state_ids
            for tenure in range(1, max(1, states_by_id[state_id].tenure_count) + 1)
        }
        for (from_state, tenure), mass in dist.items():
            if mass == 0.0:
                continue
            state = states_by_id[from_state]
            next_tenure = min(tenure + 1, state.tenure_count) if state.tenure_applied else 1
            for to_state, (probability, _cost, _effect, _extra) in _collapse(from_state, cycle, tenure).items():
                target = (from_state, next_tenure) if to_state == from_state else (to_state, 1)
                next_dist[target] = next_dist.get(target, 0.0) + mass * probability
        return next_dist

    # Layout of the vectorised advance's output dict (state-major, tenure
    # ascending) and a handle on the last dict it built, so the next cycle
    # can read the values back without per-key dict lookups.
    _all_keys = [
        (state_id, tenure)
        for i, state_id in enumerate(real_state_ids)
        for tenure in range(1, real_state_tenure_counts[i] + 1)
    ]
    _state_slices: list[tuple[int, int]] = []
    _offset = 0
    for _count in real_state_tenure_counts:
        _state_slices.append((_offset, _offset + _count))
        _offset += _count
    _built: dict[str, Any] = {"dist": None, "values": None}

    def _advance_vectorized(
        dist: Mapping[tuple[str, int], float], cycle: int
    ) -> dict[tuple[str, int], float]:
        """numpy-vectorized equivalent of _advance_scalar, valid ONLY
        when not _transitions_see_tenure (the `advance` dispatch right
        below checks this once, outside the cycle loop -- never call
        this directly on a model where the flag is True).

        When nothing along any transition varies by tenure,
        _collapse(state_id, cycle, tenure=1) already gives the SAME
        result for every tenure bucket of state_id (see _collapse's own
        tenure-forcing-to-1 branch, gated by this same flag), so the
        whole per-tenure Python loop in _advance_scalar collapses into:

          1. One n_real_state x n_real_state matrix T built once per
             cycle -- T[i, j] is the probability that a cohort member
             anywhere in real_state_ids[i] ends this cycle in
             real_state_ids[j], summed over every intervening non-real
             state (exactly what _collapse already computes; here it's
             just read into a matrix instead of looked up per tenure).
          2. For j != i (moved to a DIFFERENT real state): the result
             lands at tenure 1 of j no matter which tenure bucket of i
             it left from, so only the TOTAL mass of i matters --
             T[i, j] * sum(v_i) is one scalar added into new tenure-1
             mass of j (summed across every i != j via a matrix-vector
             product, not a Python loop over tenure).
          3. For i == j (stayed in the SAME real state): each tenure
             bucket's mass is scaled by T[i, i], then the whole array
             shifts up by one tenure (bucket k -> k+1), with the last
             bucket capped exactly as _advance_scalar's own
             `min(tenure + 1, tenure_count)` -- done for the entire
             array in one numpy assignment instead of one Python
             addition per tenure value.

        Produces the identical dict (same keys; float values matching
        _advance_scalar to the same tolerance any reordered
        floating-point sum already carries) -- verified against
        real saved autosave models, see the ad hoc parity
        check run when this was introduced (2026-09-29)."""
        n = len(real_state_ids)
        if dist is _built["dist"]:
            values = _built["values"]
            vectors = [np.array(values[a:b], dtype=float) for a, b in _state_slices]
        else:
            vectors = [
                np.array(
                    [dist.get((state_id, tenure), 0.0) for tenure in range(1, real_state_tenure_counts[i] + 1)],
                    dtype=float,
                )
                for i, state_id in enumerate(real_state_ids)
            ]
        totals = np.array([vector.sum() for vector in vectors])

        transition_matrix = np.zeros((n, n))
        for i, state_id in enumerate(real_state_ids):
            for to_state, (probability, _cost, _effect, _extra) in _collapse(state_id, cycle, 1).items():
                transition_matrix[i, real_state_index[to_state]] += probability

        incoming_from_others = transition_matrix.T @ totals - np.diag(transition_matrix) * totals

        new_values: list[float] = []
        for j, state_id in enumerate(real_state_ids):
            tenure_count = real_state_tenure_counts[j]
            self_scaled = transition_matrix[j, j] * vectors[j]
            shifted = np.zeros(tenure_count)
            if tenure_count == 1:
                shifted[0] = self_scaled[0]
            else:
                shifted[1:] = self_scaled[:-1]
                shifted[-1] += self_scaled[-1]
            shifted[0] += incoming_from_others[j]
            new_values.extend(shifted.tolist())
        next_dist = dict(zip(_all_keys, new_values))
        _built["dist"] = next_dist
        _built["values"] = new_values
        return next_dist

    # Dispatch once, outside the cycle loop -- _transitions_see_tenure is
    # a fixed property of the model's own expressions (computed above),
    # never changes cycle to cycle, so there is no per-cycle cost to
    # picking the right implementation once here.
    advance = _advance_scalar if _transitions_see_tenure else _advance_vectorized

    def trace_entry(dist: Mapping[tuple[str, int], float]) -> dict[str, float]:
        if dist is _built["dist"]:
            values = _built["values"]
            return {
                state_id: sum(values[a:b]) for state_id, (a, b) in zip(real_state_ids, _state_slices)
            }
        # Single pass grouping (same per-state summation order as before).
        groups: dict[str, list[float]] = {state_id: [] for state_id in real_state_ids}
        for (current_state, _), mass in dist.items():
            group = groups.get(current_state)
            if group is not None:
                group.append(mass)
        return {state_id: sum(group) for state_id, group in groups.items()}

    try:
        trace: list[dict[str, float]] = [trace_entry(distribution)]
        tenure_trace: list[dict[tuple[str, int], float]] = [dict(distribution)]
        cost_per_cycle: list[float] = []
        effect_per_cycle: list[float] = []
        raw_cost_per_cycle: list[float] = []
        raw_effect_per_cycle: list[float] = []
        raw_extra_per_cycle: dict[str, list[float]] = {oid: [] for oid in extra_outcome_ids}
        extra_per_cycle: dict[str, list[float]] = {oid: [] for oid in extra_outcome_ids}
        current = distribution
        for t in range(model.cycles):
            # Per-cycle caches are never reused by a later cycle -- drop them
            # so memory stays O(states x tenures) instead of growing with
            # cycles x tenures.
            for _stale in [k for k in _evaluator_cache if k[0] is not None]:
                del _evaluator_cache[_stale]
            for _stale in [k for k in _result_cache if k[0] is not None]:
                del _result_cache[_stale]
            for _stale in [k for k in _vec_cache if k[0] is not None]:
                del _vec_cache[_stale]
            for _stale in [k for k in _vec_var_cache if k[0] is not None]:
                del _vec_var_cache[_stale]
            _collapse_cache.clear()
            raw_cost, raw_effect, raw_extra = reward(current, t, apply_terminal=True)
            raw_cost_per_cycle.append(raw_cost)
            raw_effect_per_cycle.append(raw_effect)
            cost_per_cycle.append(raw_cost / (1.0 + discount_cost_per_cycle) ** t)
            effect_per_cycle.append(raw_effect / (1.0 + discount_effect_per_cycle) ** t)
            for oid in extra_outcome_ids:
                raw_extra_per_cycle[oid].append(raw_extra[oid])
                extra_per_cycle[oid].append(raw_extra[oid] / (1.0 + extra_discount_per_cycle[oid]) ** t)
            current = advance(current, t)
            trace.append(trace_entry(current))
            tenure_trace.append(dict(current))

            if early_stop_eligible:
                if current is _built["dist"]:
                    _values = _built["values"]
                    _outside: list[float] = []
                    for _sid, (_a, _b) in zip(real_state_ids, _state_slices):
                        if _sid not in structurally_absorbing_ids:
                            _outside.extend(_values[_a:_b])
                    outside_mass = sum(_outside)
                else:
                    outside_mass = sum(
                        mass
                        for (state_id, _tenure), mass in current.items()
                        if state_id not in structurally_absorbing_ids
                    )
                if outside_mass < _EARLY_STOP_MASS_TOLERANCE:
                    # The cohort has (to within a completely negligible
                    # tolerance -- see _EARLY_STOP_MASS_TOLERANCE's own
                    # comment for the error-bound math) funneled into
                    # states that can never emit mass again
                    # (structurally_absorbing_ids) and whose reward is
                    # proven cycle/tenure-independent (early_stop_
                    # eligible), so every remaining cycle would compute
                    # essentially this same (~0, ~0, {~0, ...}) reward
                    # and an unchanging distribution -- pad the rest with
                    # that fixed point instead of paying for the
                    # collapse/advance work all the way out to
                    # model.cycles. A pure exponential-decay model (e.g.
                    # an age-based mortality curve with no hard cliff to
                    # exactly 100%) never hits outside_mass == 0.0
                    # exactly in floating point, which is why this is a
                    # tolerance rather than an equality check -- this is
                    # the same "stop once the cohort is fully absorbed"
                    # idea TreeAge/Amua use: once
                    # age 100 means ~100% dead, later cycles add nothing.
                    remaining = model.cycles - (t + 1)
                    if remaining > 0:
                        raw_cost_per_cycle.extend([0.0] * remaining)
                        raw_effect_per_cycle.extend([0.0] * remaining)
                        cost_per_cycle.extend([0.0] * remaining)
                        effect_per_cycle.extend([0.0] * remaining)
                        for oid in extra_outcome_ids:
                            raw_extra_per_cycle[oid].extend([0.0] * remaining)
                            extra_per_cycle[oid].extend([0.0] * remaining)
                        final_trace_entry = trace_entry(current)
                        trace.extend([final_trace_entry] * remaining)
                        tenure_trace.extend([dict(current)] * remaining)
                    break

        total_cost = sum(cost_per_cycle)
        total_effect = sum(effect_per_cycle)
        extra_totals = {oid: sum(values) for oid, values in extra_per_cycle.items()}
        extra_totals_undiscounted = {oid: sum(values) for oid, values in raw_extra_per_cycle.items()}

        if model.half_cycle_correction and model.cycles > 0:
            final_cost, final_effect, final_extra = reward(current, model.cycles, apply_terminal=False)
            disc_final_cost = final_cost / (1.0 + discount_cost_per_cycle) ** model.cycles
            disc_final_effect = final_effect / (1.0 + discount_effect_per_cycle) ** model.cycles
            total_cost += 0.5 * (disc_final_cost - cost_per_cycle[0])
            total_effect += 0.5 * (disc_final_effect - effect_per_cycle[0])
            for oid in extra_outcome_ids:
                disc_final_extra = final_extra[oid] / (1.0 + extra_discount_per_cycle[oid]) ** model.cycles
                extra_totals[oid] += 0.5 * (disc_final_extra - extra_per_cycle[oid][0])
    except ValueError as exc:
        # Only the walk's own "loop among non-real states" check deserves
        # that code. Any other ValueError raised mid-simulation (a table
        # lookup with no matching row, a bad distribution argument, ...) is
        # an expression problem that only shows up at some later cycle, and
        # used to be mislabelled as a non-real-state loop -- send the
        # person hunting in the wrong place. Say which cycle it happened at.
        message = str(exc)
        if "loop entirely among non-real" in message:
            return MarkovResult(errors=[ValidationError(code="non_real_state_loop", message=message)])
        failed_cycle = locals().get("t")
        where = f" (while simulating cycle {failed_cycle})" if isinstance(failed_cycle, int) else ""
        return MarkovResult(errors=[ValidationError(code="expression_error", message=message + where)])

    return MarkovResult(
        total_cost=total_cost,
        total_effect=total_effect,
        cost_per_cycle=cost_per_cycle,
        effect_per_cycle=effect_per_cycle,
        raw_cost_per_cycle=raw_cost_per_cycle,
        raw_effect_per_cycle=raw_effect_per_cycle,
        extra_totals=extra_totals,
        extra_totals_undiscounted=extra_totals_undiscounted,
        trace=trace,
        tenure_trace=tenure_trace,
    )
