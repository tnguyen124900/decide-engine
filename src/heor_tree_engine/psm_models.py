"""Core data models for parametric partitioned survival models (PSM).

A PSM is defined by two parametric survival curves - progression-free
survival (PFS) and overall survival (OS) - which partition a cohort into
three health states at every cycle: pre-progression (alive, PFS(_cycle)),
post-progression (alive, OS(_cycle) - PFS(_cycle)), and dead (1 - OS(_cycle)). This is
the standard "area under the curve" partitioned-survival convention used
in oncology HEOR models (as opposed to a state-transition Markov model).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .models import NumericRef, ValidationError  # noqa: F401 (re-exported)

PSM_STATE_ROLES = ("pre_progression", "post_progression", "dead")


@dataclass
class SurvivalCurve:
    """A parametric survival function S(_cycle) = P(event has not occurred by _cycle).

    Supported distributions are exponential, Weibull, log-normal,
    log-logistic, Gompertz, and generalized gamma.
    """

    distribution: str
    params: dict[str, NumericRef] = field(default_factory=dict)


@dataclass
class PSMState:
    """Cost/effect for one of the three fixed partitioned-survival roles."""

    role: str  # one of PSM_STATE_ROLES
    name: str
    cost: NumericRef = 0.0
    effect: NumericRef = 0.0


@dataclass
class PartitionedSurvivalModel:
    """A partitioned survival model: two curves, three states, a horizon."""

    id: str
    name: str
    pfs_curve: SurvivalCurve
    os_curve: SurvivalCurve
    states: list[PSMState] = field(default_factory=list)
    cycles: int = 10
    half_cycle_correction: bool = True


@dataclass
class PSMResult:
    """Cohort partition output: totals, per-cycle rewards, and the trace."""

    total_cost: float = 0.0
    total_effect: float = 0.0
    cost_per_cycle: list[float] = field(default_factory=list)
    effect_per_cycle: list[float] = field(default_factory=list)
    # One dict of {role: occupancy} per cycle, cycle 0 .. cycles (inclusive).
    trace: list[dict[str, float]] = field(default_factory=list)
    errors: list[ValidationError] = field(default_factory=list)
