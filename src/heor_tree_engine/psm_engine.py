"""Cohort partitioning for parametric partitioned survival models."""

from __future__ import annotations

from collections.abc import Mapping

from .des_sampling import survival
from .models import NumericRef, Param
from .psm_models import PartitionedSurvivalModel, PSMResult, SurvivalCurve
from .psm_validation import curve_param_names, validate_psm


def _resolve(value: NumericRef, parameters: Mapping[str, Param]) -> float:
    if isinstance(value, str):
        return parameters[value].base
    return float(value)


def _survival(curve: SurvivalCurve, _cycle: float, parameters: Mapping[str, Param]) -> float:
    names = curve_param_names(curve)
    if not names:
        raise ValueError(f"Unsupported distribution '{curve.distribution}'")
    resolved_params = {
        name: _resolve(curve.params[name], parameters)
        for name in names
    }
    return survival(curve.distribution, resolved_params, _cycle)


def run_psm(
    model: PartitionedSurvivalModel,
    parameters: Mapping[str, Param] | None = None,
    discount_cost_per_cycle: float = 0.0,
    discount_effect_per_cycle: float = 0.0,
    allow_negative_cost: bool = False,
) -> PSMResult:
    """Validate and partition the cohort over the model's cycles.

    At each integer `_cycle`, the cohort is split into pre-progression
    (S_pfs(_cycle)), post-progression (S_os(_cycle) - S_pfs(_cycle), floored at 0), and
    dead (1 - S_os(_cycle)). Rewards are discounted per cycle and, when
    half_cycle_correction is set, the same trapezoidal correction used by
    the Markov engine is applied (half of cycle 0's reward is subtracted,
    half of an extra reward at the final cycle is added).
    """

    resolved_parameters = parameters or {}
    errors = validate_psm(model, resolved_parameters, allow_negative_cost)
    if errors:
        return PSMResult(errors=errors)

    cost = {state.role: _resolve(state.cost, resolved_parameters) for state in model.states}
    effect = {state.role: _resolve(state.effect, resolved_parameters) for state in model.states}

    def occupancy(_cycle: float) -> dict[str, float]:
        s_pfs = _survival(model.pfs_curve, _cycle, resolved_parameters)
        s_os = _survival(model.os_curve, _cycle, resolved_parameters)
        pre = s_pfs
        post = max(0.0, s_os - s_pfs)
        dead = 1.0 - s_os
        return {"pre_progression": pre, "post_progression": post, "dead": dead}

    def reward(occ: dict[str, float]) -> tuple[float, float]:
        return (
            sum(occ[role] * cost[role] for role in occ),
            sum(occ[role] * effect[role] for role in occ),
        )

    trace: list[dict[str, float]] = []
    cost_per_cycle: list[float] = []
    effect_per_cycle: list[float] = []
    for _cycle in range(model.cycles):
        occ = occupancy(_cycle)
        trace.append(occ)
        raw_cost, raw_effect = reward(occ)
        cost_per_cycle.append(raw_cost / (1.0 + discount_cost_per_cycle) ** _cycle)
        effect_per_cycle.append(raw_effect / (1.0 + discount_effect_per_cycle) ** _cycle)

    final_cycle = model.cycles
    final_occ = occupancy(final_cycle)
    trace.append(final_occ)

    total_cost = sum(cost_per_cycle)
    total_effect = sum(effect_per_cycle)

    if model.half_cycle_correction and model.cycles > 0:
        final_cost, final_effect = reward(final_occ)
        disc_final_cost = final_cost / (1.0 + discount_cost_per_cycle) ** final_cycle
        disc_final_effect = final_effect / (1.0 + discount_effect_per_cycle) ** final_cycle
        total_cost += 0.5 * (disc_final_cost - cost_per_cycle[0])
        total_effect += 0.5 * (disc_final_effect - effect_per_cycle[0])

    return PSMResult(
        total_cost=total_cost,
        total_effect=total_effect,
        cost_per_cycle=cost_per_cycle,
        effect_per_cycle=effect_per_cycle,
        trace=trace,
    )
