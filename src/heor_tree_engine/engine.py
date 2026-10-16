"""Rollback calculations for decision trees."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .models import (
    Branch,
    ChanceNode,
    DecisionNode,
    DecisionResult,
    DESTreeNode,
    MarkovTreeNode,
    Node,
    NumericRef,
    Param,
    PSM_STATE_ROLES,
    PSMNode,
    Result,
    SurvivalCurve,
    TerminalNode,
    ValidationError,
    is_complement,
)
from .validation import validate
from .expressions import evaluate_expression, resolve_variables


def _engine_variables(
    variables: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {**(variables or {}), "_cycle": 0, "_tenure": 1}


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


def _curve_survival(
    curve: SurvivalCurve,
    time_years: float,
    parameters: Mapping[str, Param],
    variables: Mapping[str, Any] | None = None,
    distributions: Mapping[str, Mapping[str, Any]] | None = None,
    tables: Mapping[str, Mapping[str, Any]] | None = None,
) -> float:
    """Evaluate a PSMNode's survival curve at elapsed time in years. Exponential/Weibull are the
    original two formulas (also kept in the standalone psm_engine.py's
    _survival() for the legacy PSM screen until it's retired); log_normal/
    log_logistic/gompertz/generalized_gamma were added so the PSM curve picker
    covers all 6 NICE DSU TSD 14 distributions already in the general
    Distributions system. The first three use the same
    parameter names as that system's scipy adapters; PSM generalized gamma
    uses Prentice mu/sigma/Q and converts internally, while general
    Distributions retains a/c/scale. Every formula but generalized_gamma
    is a closed form (plain `math`, no new hard dependency -- matches this
    module's existing zero-scipy-required philosophy for exponential/
    weibull); generalized_gamma's survival function needs the regularized
    incomplete gamma function, which stdlib `math` doesn't provide, so it
    imports scipy lazily and raises the same "requires scipy" error the
    general Distributions system already raises when scipy is missing --
    resolved through this engine's own _resolve() throughout so curve
    params can be parameter references/expressions like every other
    numeric field in a tree, not just literal numbers."""

    from .des_models import DES_TTE_PARAMS
    from .des_sampling import survival
    from .models import curve_time_in_native_unit

    names = DES_TTE_PARAMS.get(curve.distribution)
    if names is None or curve.distribution in ("fixed", "expression", "table_to_time", "named"):
        raise ValueError(f"Unsupported survival distribution '{curve.distribution}'")
    # One survival formula for PSM curves and DES time-to-event
    # (des_sampling.survival) -- same math, one implementation.
    if curve.distribution == "generalized_gamma" and "mu" in curve.params:
        names = ("mu", "sigma", "Q")
    params = {name: _resolve(curve.params[name], parameters, variables, distributions, tables) for name in names}
    # time_years is elapsed time in YEARS (cycle / cycles_per_year); convert
    # to the curve's own time_unit (default "year", a no-op) so params can
    # be typed exactly as published -- see curve_time_in_native_unit().
    return survival(curve.distribution, params, curve_time_in_native_unit(curve, time_years))


def _branch_cost(
    branch: Branch,
    parameters: Mapping[str, Param],
    variables: Mapping[str, Any] | None = None,
    distributions: Mapping[str, Mapping[str, Any]] | None = None,
    tables: Mapping[str, Mapping[str, Any]] | None = None,
) -> float:
    if branch.cost_increment is None:
        return 0.0
    return _resolve(branch.cost_increment, parameters, variables, distributions, tables)


def _branch_effect(
    branch: Branch,
    parameters: Mapping[str, Param],
    variables: Mapping[str, Any] | None = None,
    distributions: Mapping[str, Mapping[str, Any]] | None = None,
    tables: Mapping[str, Mapping[str, Any]] | None = None,
) -> float:
    if branch.effect_increment is None:
        return 0.0
    return _resolve(branch.effect_increment, parameters, variables, distributions, tables)


def _branch_extra_outcomes(
    branch: Branch,
    outcome_kinds: Mapping[str, str],
    parameters: Mapping[str, Param],
    variables: Mapping[str, Any] | None = None,
    distributions: Mapping[str, Mapping[str, Any]] | None = None,
    tables: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, float]:
    """Resolve a branch's extra Cost/Effect outcomes (beyond the primary
    pair) for every outcome id the model defines. See
    Branch.extra_outcomes / Engine.run()'s outcome_kinds parameter."""

    if not outcome_kinds or not branch.extra_outcomes:
        return {}
    return {
        outcome_id: _resolve(branch.extra_outcomes[outcome_id], parameters, variables, distributions, tables)
        for outcome_id in outcome_kinds
        if outcome_id in branch.extra_outcomes
    }


def _discount_extra_totals(
    totals: Mapping[str, float],
    year: float,
    outcome_kinds: Mapping[str, str],
    cost_discount_rate: float,
    effect_discount_rate: float,
) -> dict[str, float]:
    """Apply the same per-year discounting used for the primary Cost/
    Effect pair to every extra outcome, using the cost or effect discount
    rate depending on that outcome's own kind ("cost" or "effect")."""

    if not year:
        return dict(totals)
    discounted: dict[str, float] = {}
    for outcome_id, value in totals.items():
        rate = cost_discount_rate if outcome_kinds.get(outcome_id) == "cost" else effect_discount_rate
        discounted[outcome_id] = value / (1.0 + rate) ** year if rate else value
    return discounted


def _prefix_paths(prefix: list[str], paths: list[list[str]]) -> list[list[str]]:
    return [prefix + path for path in paths]


def _evaluate_node(
    node: Node,
    parameters: Mapping[str, Param],
    variables: Mapping[str, Any] | None = None,
    distributions: Mapping[str, Mapping[str, Any]] | None = None,
    cost_discount_rate: float = 0.0,
    effect_discount_rate: float = 0.0,
    outcome_kinds: Mapping[str, str] | None = None,
    tables: Mapping[str, Mapping[str, Any]] | None = None,
) -> Result:
    outcome_kinds = outcome_kinds or {}
    if isinstance(node, TerminalNode):
        cost = _resolve(node.cost, parameters, variables, distributions, tables)
        effect = _resolve(node.effect, parameters, variables, distributions, tables)
        year = _resolve(node.year, parameters, variables, distributions, tables)
        extra_totals = {
            outcome_id: _resolve(node.extra_outcomes[outcome_id], parameters, variables, distributions, tables)
            for outcome_id in outcome_kinds
            if outcome_id in node.extra_outcomes
        }
        if year:
            if cost_discount_rate:
                cost /= (1.0 + cost_discount_rate) ** year
            if effect_discount_rate:
                effect /= (1.0 + effect_discount_rate) ** year
            extra_totals = _discount_extra_totals(
                extra_totals, year, outcome_kinds, cost_discount_rate, effect_discount_rate
            )
        return Result(
            cost=cost,
            effect=effect,
            extra_totals=extra_totals,
            paths=[[node.id]],
        )

    if isinstance(node, ChanceNode):
        total_cost = 0.0
        total_effect = 0.0
        total_extras: dict[str, float] = {outcome_id: 0.0 for outcome_id in outcome_kinds}
        paths: list[list[str]] = []
        resolved: dict[int, float] = {}
        complement_index: int | None = None
        known_sum = 0.0
        for index, branch in enumerate(node.branches):
            if branch.probability is None or branch.child is None:
                raise ValueError(f"Chance branch '{branch.label}' is incomplete")
            if is_complement(branch.probability):
                complement_index = index
                continue
            probability = _resolve(branch.probability, parameters, variables, distributions, tables)
            resolved[index] = probability
            known_sum += probability
        if complement_index is not None:
            # "#"/"C" means: whatever probability is left so the siblings sum to 1.
            resolved[complement_index] = 1.0 - known_sum
        for index, branch in enumerate(node.branches):
            probability = resolved[index]
            child_result = _evaluate_node(
                branch.child, parameters, variables, distributions,
                cost_discount_rate, effect_discount_rate, outcome_kinds, tables,
            )
            total_cost += probability * (
                _branch_cost(branch, parameters, variables, distributions, tables) + child_result.cost
            )
            total_effect += probability * (
                _branch_effect(branch, parameters, variables, distributions, tables) + child_result.effect
            )
            if outcome_kinds:
                branch_extras = _branch_extra_outcomes(
                    branch, outcome_kinds, parameters, variables, distributions, tables
                )
                for outcome_id in outcome_kinds:
                    total_extras[outcome_id] += probability * (
                        branch_extras.get(outcome_id, 0.0) + child_result.extra_totals.get(outcome_id, 0.0)
                    )
            paths.extend(
                _prefix_paths([node.id, branch.label], child_result.paths)
            )
        return Result(cost=total_cost, effect=total_effect, extra_totals=total_extras, paths=paths)

    if isinstance(node, PSMNode):
        branches_by_role: dict[str, Branch] = {}
        for branch in node.branches:
            if branch.role not in PSM_STATE_ROLES:
                raise ValueError(f"PSM node '{node.id}' branch '{branch.label}' has an invalid or missing role")
            if branch.child is None:
                raise ValueError(f"PSM branch '{branch.label}' is incomplete")
            branches_by_role[branch.role] = branch
        missing_roles = set(PSM_STATE_ROLES) - set(branches_by_role)
        if missing_roles:
            raise ValueError(f"PSM node '{node.id}' is missing branch role(s): {', '.join(sorted(missing_roles))}")

        # A PSM branch's (branch cost_increment/effect_increment + its
        # Terminal child's own cost/effect) -- exactly the same quantity
        # a Chance branch sums (see _branch_cost/_branch_effect below) and
        # exactly what the canvas's edge-editable $=/E= rows already
        # write to, so editing a PSM branch's reward reuses Tree's
        # existing per-edge editing UI with no new form needed -- is read
        # as a PER-CYCLE rate integrated over that role's occupancy across
        # cycles, NOT the one-off lump sum the same quantity means
        # everywhere else in the tree. The child's own `year` field is
        # ignored (there is no single "year" a continuous occupancy
        # integral happens at).
        # rate_cost/rate_effect used to be resolved ONCE here, before the
        # per-cycle loop below, using _cycle frozen at 0 by the
        # module-level _engine_variables() above -- so a user-typed
        # if(_cycle<=4, X, Y) cost/effect formula on a PSM branch always
        # picked the _cycle==0 branch, no matter which cycle the
        # occupancy integral below was actually summing (this was a reported
        # bug). Fixed by resolving rate_cost/
        # rate_effect fresh for every cycle, with the REAL cycle index
        # injected into `variables` -- mirrors markov_engine.py's own
        # per-cycle variable injection (its _engine_variables(cycle,
        # tenure, variables) == {**variables, "_cycle": cycle, "_tenure":
        # tenure}, rebuilt every cycle/tenure), so "_cycle" means the
        # same thing -- the current cycle index -- everywhere in DECIDE,
        # Tree/PSM and Markov alike (TreeAge's PSM equivalent is its
        # "_stage" variable). PSM still has no tenure concept (occupancy
        # is read straight off the survival curves at absolute time t),
        # so _tenure
        # stays fixed at 1, same as before this fix.
        def cycle_parameters(cycle: int) -> Mapping[str, Param]:
            # `parameters` (the outer argument) already has "_cycle"/
            # "_tenure" baked in as FROZEN Param objects (base=0.0/1.0) --
            # resolve_variables(), called once at the top of
            # expected_cost()/evaluate_decision()/etc. before _evaluate_node
            # ever runs, absorbs every key of _engine_variables() (that
            # includes these two) into the resolved parameters dict. And
            # _Evaluator._resolve_name() checks self.parameters BEFORE
            # self.variables (see expressions.py) -- so overriding
            # `variables["_cycle"]` alone is not enough, that frozen
            # parameters-dict entry always wins first. Must override the
            # PARAMETER too, per cycle, for a user-typed _cycle reference to
            # see the real cycle index.
            return {
                **parameters,
                "_cycle": Param(name="_cycle", base=float(cycle)),
                "_tenure": Param(name="_tenure", base=1.0),
            }

        def cycle_variables(cycle: int) -> dict[str, Any]:
            return {**(variables or {}), "_cycle": cycle, "_tenure": 1}

        def branch_rewards(cycle: int) -> tuple[dict[str, float], dict[str, float]]:
            cycle_params = cycle_parameters(cycle)
            cycle_vars = cycle_variables(cycle)
            rate_cost: dict[str, float] = {}
            rate_effect: dict[str, float] = {}
            for role, branch in branches_by_role.items():
                child = branch.child
                if not isinstance(child, TerminalNode):
                    raise ValueError(
                        f"PSM node '{node.id}' branch '{branch.label}' must lead to a Terminal node "
                        "(the only child kind a PSM branch supports today)"
                    )
                rate_cost[role] = _branch_cost(branch, cycle_params, cycle_vars, distributions, tables) + _resolve(
                    child.cost, cycle_params, cycle_vars, distributions, tables
                )
                rate_effect[role] = _branch_effect(branch, cycle_params, cycle_vars, distributions, tables) + _resolve(
                    child.effect, cycle_params, cycle_vars, distributions, tables
                )
            return rate_cost, rate_effect

        cycles_per_year = node.cycles_per_year if node.cycles_per_year > 0 else 1.0

        def occupancy(cycle: float) -> dict[str, float]:
            cycle_time_years = cycle / cycles_per_year
            s_pfs = _curve_survival(node.pfs_curve, cycle_time_years, parameters, variables, distributions, tables)
            s_os = _curve_survival(node.os_curve, cycle_time_years, parameters, variables, distributions, tables)
            return {
                "pre_progression": s_pfs,
                "post_progression": max(0.0, s_os - s_pfs),
                "dead": 1.0 - s_os,
            }

        def cycle_reward(cycle: int) -> tuple[float, float]:
            occ = occupancy(cycle)
            rate_cost, rate_effect = branch_rewards(cycle)
            return (
                sum(occ[role] * rate_cost[role] for role in PSM_STATE_ROLES),
                sum(occ[role] * rate_effect[role] for role in PSM_STATE_ROLES),
            )

        cost_per_cycle: list[float] = []
        effect_per_cycle: list[float] = []
        for _cycle in range(node.cycles):
            cycle_time_years = _cycle / cycles_per_year
            raw_cost, raw_effect = cycle_reward(_cycle)
            # Discount rates and survival curves use elapsed years; rewards
            # remain per-cycle values.
            cost_per_cycle.append(raw_cost / (1.0 + cost_discount_rate) ** cycle_time_years)
            effect_per_cycle.append(raw_effect / (1.0 + effect_discount_rate) ** cycle_time_years)

        total_cost = sum(cost_per_cycle)
        total_effect = sum(effect_per_cycle)
        if node.half_cycle_correction and node.cycles > 0:
            final_cycle = node.cycles
            final_time_years = final_cycle / cycles_per_year
            final_cost, final_effect = cycle_reward(final_cycle)
            disc_final_cost = final_cost / (1.0 + cost_discount_rate) ** final_time_years
            disc_final_effect = final_effect / (1.0 + effect_discount_rate) ** final_time_years
            total_cost += 0.5 * (disc_final_cost - cost_per_cycle[0])
            total_effect += 0.5 * (disc_final_effect - effect_per_cycle[0])

        paths = [
            [node.id, branches_by_role[role].label, branches_by_role[role].child.id]
            for role in PSM_STATE_ROLES
        ]
        return Result(
            cost=total_cost,
            effect=total_effect,
            extra_totals={outcome_id: 0.0 for outcome_id in outcome_kinds},
            paths=paths,
        )

    if isinstance(node, DESTreeNode):
        return _evaluate_des_node(
            node, parameters, variables, distributions,
            cost_discount_rate, effect_discount_rate, outcome_kinds, tables,
        )

    if isinstance(node, MarkovTreeNode):
        return _evaluate_markov_node(
            node, parameters, variables, distributions,
            cost_discount_rate, effect_discount_rate, outcome_kinds, tables,
        )

    if isinstance(node, DecisionNode):
        raise TypeError("Use evaluate_decision() for a DecisionNode")

    raise TypeError(f"Unsupported node type: {type(node).__name__}")


def _per_cycle_rate(annual_rate: float, cycles_per_year: float) -> float:
    """Annual discount rate -> equivalent per-cycle rate."""

    if not annual_rate:
        return 0.0
    return (1.0 + annual_rate) ** (1.0 / max(cycles_per_year, 1e-9)) - 1.0


def _evaluate_markov_node(
    node: MarkovTreeNode,
    parameters: Mapping[str, Param],
    variables: Mapping[str, Any] | None,
    distributions: Mapping[str, Mapping[str, Any]] | None,
    cost_discount_rate: float,
    effect_discount_rate: float,
    outcome_kinds: Mapping[str, str],
    tables: Mapping[str, Mapping[str, Any]] | None,
) -> Result:
    """Run the node's cohort Markov model and roll it back as one node: its
    value is the discounted total cost/effect of the cohort (TreeAge's
    Markov node). The tree's annual discount rates are converted to
    per-cycle rates using the node's cycles_per_year."""

    import dataclasses

    from .markov_engine import run_markov

    markov_variables = {**node.variables, **(variables or {})}
    # Variables already folded into `parameters` (evaluated once) must not
    # be passed twice: the Markov evaluator would report a name clash. Keep
    # them as raw formulas so per-cycle ones (_cycle/_tenure) still work.
    base_parameters = {name: value for name, value in parameters.items() if name not in markov_variables}
    model = node.model
    if outcome_kinds:
        model = dataclasses.replace(
            model,
            extra_outcomes=[{"id": oid, "type": kind, "label": oid} for oid, kind in outcome_kinds.items()],
        )
    result = run_markov(
        model, base_parameters,
        _per_cycle_rate(cost_discount_rate, node.cycles_per_year),
        _per_cycle_rate(effect_discount_rate, node.cycles_per_year),
        variables=markov_variables, distributions=distributions, tables=tables,
    )
    if result.errors:
        first = result.errors[0]
        where = f" ({first.node_id})" if first.node_id else ""
        raise DESRunError(f"Markov node '{node.id}': {first.message}{where}", first.code, first.node_id or node.id)
    return Result(
        cost=result.total_cost,
        effect=result.total_effect,
        extra_totals={oid: result.extra_totals.get(oid, 0.0) for oid in outcome_kinds},
        paths=[[node.id]],
    )


class DESRunError(ValueError):
    """A DES node could not be simulated (e.g. an unknown name in a formula
    evaluated per patient). Engine.run() turns it into a ValidationError
    instead of letting it crash the caller."""

    def __init__(self, message: str, code: str = "des_runtime_error", node_id: str | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.node_id = node_id


def _evaluate_des_node(
    node: DESTreeNode,
    parameters: Mapping[str, Param],
    variables: Mapping[str, Any] | None,
    distributions: Mapping[str, Mapping[str, Any]] | None,
    cost_discount_rate: float,
    effect_discount_rate: float,
    outcome_kinds: Mapping[str, str],
    tables: Mapping[str, Mapping[str, Any]] | None,
) -> Result:
    """Run the DES subtree and roll it back as one node: its value is the
    mean discounted cost/effect across the simulated patients (TreeAge
    DES output). The discount rates are the tree's own annual rates."""

    from .des_engine import run_des

    des_variables = {**node.variables, **(variables or {})}
    result = run_des(
        node.model, parameters, cost_discount_rate, effect_discount_rate,
        variables=des_variables, distributions=distributions, tables=tables,
        outcome_kinds=outcome_kinds,
    )
    if result.errors:
        first = result.errors[0]
        where = f" ({first.node_id})" if first.node_id else ""
        raise DESRunError(f"DES node '{node.id}': {first.message}{where}", first.code, first.node_id or node.id)
    return Result(
        cost=result.total_cost,
        effect=result.total_effect,
        extra_totals={outcome_id: result.extra_totals.get(outcome_id, 0.0) for outcome_id in outcome_kinds},
        paths=[[node.id]],
    )


def expected_cost(
    node: Node,
    parameters: Mapping[str, Param] | None = None,
    variables: Mapping[str, Any] | None = None,
    distributions: Mapping[str, Mapping[str, Any]] | None = None,
    cost_discount_rate: float = 0.0,
    effect_discount_rate: float = 0.0,
    tables: Mapping[str, Mapping[str, Any]] | None = None,
) -> float:
    """Return the rollback expected cost for a non-decision node."""

    engine_variables = _engine_variables(variables)
    resolved_parameters = resolve_variables(parameters or {}, engine_variables, distributions, tables)
    return _evaluate_node(
        node, resolved_parameters, engine_variables, distributions,
        cost_discount_rate, effect_discount_rate, tables=tables,
    ).cost


def expected_utility(
    node: Node,
    parameters: Mapping[str, Param] | None = None,
    variables: Mapping[str, Any] | None = None,
    distributions: Mapping[str, Mapping[str, Any]] | None = None,
    cost_discount_rate: float = 0.0,
    effect_discount_rate: float = 0.0,
    tables: Mapping[str, Mapping[str, Any]] | None = None,
) -> float:
    """Return the rollback expected QALY effect for a non-decision node."""

    engine_variables = _engine_variables(variables)
    resolved_parameters = resolve_variables(parameters or {}, engine_variables, distributions, tables)
    return _evaluate_node(
        node, resolved_parameters, engine_variables, distributions,
        cost_discount_rate, effect_discount_rate, tables=tables,
    ).effect


def evaluate_decision(
    node: DecisionNode,
    parameters: Mapping[str, Param] | None = None,
    variables: Mapping[str, Any] | None = None,
    distributions: Mapping[str, Mapping[str, Any]] | None = None,
    cost_discount_rate: float = 0.0,
    effect_discount_rate: float = 0.0,
    outcome_kinds: Mapping[str, str] | None = None,
    tables: Mapping[str, Mapping[str, Any]] | None = None,
) -> DecisionResult:
    """Return a separate rollback result for every strategy branch."""

    outcome_kinds = outcome_kinds or {}
    results: dict[str, Result] = {}
    engine_variables = _engine_variables(variables)
    resolved_parameters = resolve_variables(parameters or {}, engine_variables, distributions, tables)
    for branch in node.branches:
        if branch.child is None:
            raise ValueError(f"Decision branch '{branch.label}' is incomplete")
        child_result = _evaluate_node(
            branch.child, resolved_parameters, engine_variables, distributions,
            cost_discount_rate, effect_discount_rate, outcome_kinds, tables,
        )
        branch_cost = _branch_cost(branch, resolved_parameters, engine_variables, distributions, tables)
        branch_effect = _branch_effect(branch, resolved_parameters, engine_variables, distributions, tables)
        extra_totals = dict(child_result.extra_totals)
        if outcome_kinds:
            branch_extras = _branch_extra_outcomes(
                branch, outcome_kinds, resolved_parameters, engine_variables, distributions, tables
            )
            for outcome_id in outcome_kinds:
                extra_totals[outcome_id] = branch_extras.get(outcome_id, 0.0) + extra_totals.get(outcome_id, 0.0)
        results[branch.label] = Result(
            cost=branch_cost + child_result.cost,
            effect=branch_effect + child_result.effect,
            extra_totals=extra_totals,
            paths=_prefix_paths([node.id, branch.label], child_result.paths),
        )
    return DecisionResult(strategies=results)


def _node_terminal_probabilities(
    node: Node,
    parameters: Mapping[str, Param],
    variables: Mapping[str, Any] | None = None,
    distributions: Mapping[str, Mapping[str, Any]] | None = None,
    tables: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, float]:
    """Recursive helper for terminal_probabilities(): the probability of
    reaching each TerminalNode under `node`, mirroring the branch-
    probability resolution in _evaluate_node() (including "#"/complement
    handling) but tracking reach-probability instead of cost/effect."""

    if isinstance(node, TerminalNode):
        return {node.id: 1.0}

    if isinstance(node, ChanceNode):
        resolved: dict[int, float] = {}
        complement_index: int | None = None
        known_sum = 0.0
        for index, branch in enumerate(node.branches):
            if branch.probability is None or branch.child is None:
                raise ValueError(f"Chance branch '{branch.label}' is incomplete")
            if is_complement(branch.probability):
                complement_index = index
                continue
            probability = _resolve(branch.probability, parameters, variables, distributions, tables)
            resolved[index] = probability
            known_sum += probability
        if complement_index is not None:
            resolved[complement_index] = 1.0 - known_sum
        totals: dict[str, float] = {}
        for index, branch in enumerate(node.branches):
            probability = resolved[index]
            child_probabilities = _node_terminal_probabilities(
                branch.child, parameters, variables, distributions, tables
            )
            for terminal_id, child_probability in child_probabilities.items():
                totals[terminal_id] = totals.get(terminal_id, 0.0) + probability * child_probability
        return totals

    if isinstance(node, PSMNode):
        # No single branch probability exists for a PSM node -- the closest
        # analogue is the cohort's occupancy split at the END of the
        # horizon (which, unlike the per-cycle occupancy used for
        # cost/effect, genuinely sums to 1 and answers "what fraction of
        # the cohort ends up in each of the 3 states").
        final_time_years = node.cycles / (node.cycles_per_year if node.cycles_per_year > 0 else 1.0)
        final_pfs = _curve_survival(node.pfs_curve, final_time_years, parameters, variables, distributions, tables)
        final_os = _curve_survival(node.os_curve, final_time_years, parameters, variables, distributions, tables)
        final_occupancy = {
            "pre_progression": final_pfs,
            "post_progression": max(0.0, final_os - final_pfs),
            "dead": 1.0 - final_os,
        }
        totals: dict[str, float] = {}
        for branch in node.branches:
            if branch.role not in final_occupancy or branch.child is None:
                continue
            child_probabilities = _node_terminal_probabilities(
                branch.child, parameters, variables, distributions, tables
            )
            weight = final_occupancy[branch.role]
            for terminal_id, child_probability in child_probabilities.items():
                totals[terminal_id] = totals.get(terminal_id, 0.0) + weight * child_probability
        return totals

    if isinstance(node, MarkovTreeNode):
        # Like a DES node: no Tree terminals below a Markov node.
        return {}

    if isinstance(node, DESTreeNode):
        # A DES node has no Tree terminals below it (its Terminals live in
        # the DES subtree and are reached over time, not with one probability).
        return {}

    if isinstance(node, DecisionNode):
        raise TypeError("Use terminal_probabilities() with a DecisionNode root, not _node_terminal_probabilities()")

    raise TypeError(f"Unsupported node type: {type(node).__name__}")


def terminal_probabilities(
    root: Node,
    strategy_label: str,
    parameters: Mapping[str, Param] | None = None,
    variables: Mapping[str, Any] | None = None,
    distributions: Mapping[str, Mapping[str, Any]] | None = None,
    tables: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, float]:
    """Return {terminal_node_id: probability of reaching it}, for one
    strategy branch (a top-level branch of the DecisionNode `root`, keyed
    by its label -- the same "strategy_id" used throughout evaluate_decision()
    / evaluate_tree_scenario()).

    Unlike Markov/Microsim's per-cycle state occupancy, a decision tree has
    no loops, so every terminal node has exactly one path down from a
    strategy's root branch, and its reach-probability is simply the
    product of the branch probabilities along that path (summed across
    branches only in the unusual case where two different branches happen
    to point at the same terminal node id).

    Raises TypeError if `root` isn't a DecisionNode, or ValueError if
    `strategy_label` doesn't match any of its branches -- both signal a
    caller bug rather than an expected "no result" case, so callers that
    need a soft failure (e.g. calibration.py's residual function, scoring
    an optimizer's candidate) should catch these themselves.

    Main use case: calibration.py's Tree "terminal_probability" target
    metric -- e.g. "the real-world complication rate is 12%; find branch
    probabilities that reproduce that".
    """

    if not isinstance(root, DecisionNode):
        raise TypeError("terminal_probabilities() requires a DecisionNode root")
    engine_variables = _engine_variables(variables)
    resolved_parameters = resolve_variables(parameters or {}, engine_variables, distributions, tables)
    for branch in root.branches:
        if branch.label == strategy_label:
            if branch.child is None:
                raise ValueError(f"Decision branch '{branch.label}' is incomplete")
            return _node_terminal_probabilities(
                branch.child, resolved_parameters, engine_variables, distributions, tables
            )
    raise ValueError(f"Unknown strategy '{strategy_label}'")


def des_strategy_runs(
    root: Node,
    strategy_label: str,
    parameters: Mapping[str, Param] | None = None,
    variables: Mapping[str, Any] | None = None,
    distributions: Mapping[str, Mapping[str, Any]] | None = None,
    tables: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[tuple[float, Any, dict[str, str]]]:
    """Every DES node under one strategy, with the probability of reaching
    it (product of Chance probabilities on the way) and its simulation
    result: [(weight, DESResult, {node id: state name}), ...]. Used by
    calibration's "des_state_share" target (e.g. % alive at year 5)."""

    from .des_engine import run_des
    from .des_models import DESTerminal, TimeNode as _DESTimeNode

    if not isinstance(root, DecisionNode):
        raise TypeError("des_strategy_runs() requires a DecisionNode root")
    engine_variables = _engine_variables(variables)
    resolved = resolve_variables(parameters or {}, engine_variables, distributions, tables)
    runs: list[tuple[float, Any, dict[str, str]]] = []

    def walk(node: Node, weight: float) -> None:
        if isinstance(node, DESTreeNode):
            result = run_des(
                node.model, resolved, variables={**node.variables, **engine_variables},
                distributions=distributions, tables=tables,
            )
            if result.errors:
                first = result.errors[0]
                raise DESRunError(f"DES node '{node.id}': {first.message}", first.code, first.node_id or node.id)
            names = {
                node_id: str(getattr(model_node, "name", "") or node_id)
                for node_id, model_node in node.model.nodes.items()
                if isinstance(model_node, (_DESTimeNode, DESTerminal))
            }
            runs.append((weight, result, names))
            return
        if isinstance(node, ChanceNode):
            resolved_p: dict[int, float] = {}
            complement: int | None = None
            known = 0.0
            for index, branch in enumerate(node.branches):
                if is_complement(branch.probability):
                    complement = index
                    continue
                resolved_p[index] = _resolve(branch.probability, resolved, engine_variables, distributions, tables)
                known += resolved_p[index]
            if complement is not None:
                resolved_p[complement] = 1.0 - known
            for index, branch in enumerate(node.branches):
                if branch.child is not None:
                    walk(branch.child, weight * resolved_p[index])

    for branch in root.branches:
        if branch.label == strategy_label:
            if branch.child is not None:
                walk(branch.child, 1.0)
            return runs
    raise ValueError(f"Unknown strategy '{strategy_label}'")


def des_state_share(runs: list[tuple[float, Any, dict[str, str]]], state_name: str, time: float) -> float:
    """Share of the strategy's cohort in the state called `state_name`
    (a Time node's or absorbing Terminal's name, e.g. "Severe" or "Dead",
    or its id) at `time`, read from each DES run's trace at the nearest
    grid point and weighted by the probability of reaching that DES node."""

    total = 0.0
    for weight, result, names in runs:
        if not result.trace_times:
            continue
        index = min(range(len(result.trace_times)), key=lambda i: abs(result.trace_times[i] - time))
        row = result.trace[index]
        total += weight * sum(share for key, share in row.items() if key == state_name or names.get(key) == state_name)
    return total


class Engine:
    """Validate and evaluate a decision tree in one explicit operation."""

    def run(
        self,
        tree: Node,
        parameters: Mapping[str, Param] | None = None,
        allow_negative_cost: bool = False,
        variables: Mapping[str, Any] | None = None,
        distributions: Mapping[str, Mapping[str, Any]] | None = None,
        cost_discount_rate: float = 0.0,
        effect_discount_rate: float = 0.0,
        outcome_kinds: Mapping[str, str] | None = None,
        tables: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> DecisionResult:
        """Return strategy results, or structured errors without executing rollback.

        ``cost_discount_rate``/``effect_discount_rate`` are annual discount
        rates (e.g. 0.03 for 3%/year) applied to each TerminalNode's payoff
        based on its ``year`` field. Both default to 0.0, which reproduces
        the exact previous behaviour for every model that doesn't set a
        terminal node's year.

        ``outcome_kinds`` maps the id of every *extra* Cost/Effect outcome
        (beyond the primary pair) to "cost" or "effect" -- see
        document["outcomes"] in app.py. Each extra outcome is summed
        exactly like the primary cost/effect and discounted with the same
        rate as its kind. Omit it (or pass {}) for a model with no extra
        outcomes, which reproduces the exact previous behaviour.
        """

        engine_variables = _engine_variables(variables)
        resolved_parameters = resolve_variables(parameters or {}, engine_variables, distributions, tables)
        errors = validate(
            tree,
            parameters=resolved_parameters,
            allow_negative_cost=allow_negative_cost,
            variables=engine_variables,
            distributions=distributions,
            tables=tables,
        )
        if errors:
            return DecisionResult(errors=errors)
        if not isinstance(tree, DecisionNode):
            return DecisionResult(
                errors=[
                    ValidationError(
                        code="root_not_decision",
                        message="Engine.run requires a DecisionNode root",
                        node_id=tree.id,
                    )
                ]
            )
        try:
            return evaluate_decision(
                tree, parameters, variables, distributions,
                cost_discount_rate, effect_discount_rate, outcome_kinds, tables,
            )
        except DESRunError as exc:
            return DecisionResult(errors=[ValidationError(code=exc.code, message=str(exc), node_id=exc.node_id)])