"""Core data models for Markov cohort models.

Kept in a separate module from the decision-tree models (states/transitions
are a different shape from nodes/branches), but shares the same
NumericRef/Param/ValidationError building blocks so a future decision-tree
terminal node can point at a MarkovModel (a "Markov as terminal" hybrid
model, common in HEOR practice).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .models import NumericRef, ValidationError  # noqa: F401 (re-exported)


@dataclass
class MarkovState:
    """A health state a cohort member can occupy during a cycle."""

    id: str
    name: str
    cost: NumericRef = 0.0
    tenure_applied: bool = False
    tenure_count: int = 1
    initial_cost: NumericRef = 0.0
    recurring_cost: NumericRef | None = None
    terminal_cost: NumericRef = 0.0
    effect: NumericRef = 0.0
    initial_prob: NumericRef = 0.0
    # Convenience marker for a state with no way out (e.g. "Death"): if it has
    # zero declared outgoing transitions, the engine treats it as an implicit
    # self-loop with probability 1 instead of raising a validation error.
    is_absorbing: bool = False
    # Extra Cost/Effect outcomes (Outcomes table rows beyond the primary
    # "cost"/"effect" pair, e.g. a second Effect "LY" or a societal cost),
    # keyed by outcome id -> a per-cycle rate, same NumericRef convention
    # as `cost`/`effect`. Mirrors Branch.extra_outcomes (Tree/PSM) and
    # DTMCompartment.outcome_rates -- same idea, per-state instead of
    # per-branch/per-compartment. NOT combined with tenure_applied's
    # initial/recurring/terminal split (that only applies to the primary
    # cost) -- an extra outcome is always a flat per-cycle rate.
    extra_outcomes: dict[str, NumericRef] = field(default_factory=dict)
    # False for a state that is a canvas-created CHILD of another state
    # (the DECIDE app's Markov canvas: a double-click "quick branch" on an
    # existing state, not on the strategy's own root) rather than a direct
    # child of the strategy root -- i.e. a momentary decision/chance point
    # (an adverse-event branch, a sub-split of an existing pathway) instead
    # of a health state the cohort actually occupies for a whole cycle.
    # True (the default) is every ordinary state, including every model
    # built before this field existed -- run_markov() only takes a cycle
    # to move THROUGH a real (True) state; it resolves a whole chain of
    # non-real states in zero elapsed time within that same cycle (see
    # run_markov()'s own docstring), so cost/effect on a non-real state is
    # a one-off contribution the first time the cohort passes through it,
    # never a recurring per-cycle charge.
    is_real_state: bool = True
    # Explicit link from a non-real dead-end leaf (created via the DECIDE
    # app's "relabel as terminal" canvas action) to the id of the real
    # state it displays as / is economically identical to. When a
    # non-real state has zero outgoing transitions, the engine resolves
    # it straight into this real state (zero elapsed cycles) instead of
    # requiring an explicit is_absorbing flag or a self-loop transition --
    # matching the Markov principle that reaching any terminal-looking
    # leaf simply means the cohort is already occupying whatever real
    # health state that leaf represents. None (the default) means either
    # this state needs its own is_absorbing flag (if it's a real state),
    # or the engine falls back to matching by `name` against a real
    # state (for models saved before this field existed).
    alias_of: str | None = None


@dataclass
class MarkovTransition:
    """A directed transition probability from one state to another.

    cost_increment/effect_increment are OPTIONAL, per-transition rewards
    (mirrors Branch.cost_increment/effect_increment on the Decision Tree
    side): added on top of the destination state's own Initial/Recurring/
    Terminal cost when this specific transition fires, so two different
    transitions into the same shared state (e.g. two different paths into
    "Dead") can carry different costs without the engine forcing them to
    match. Default 0 -- existing models/callers (incl. Microsim/ABM, which
    reuse this same dataclass) are unaffected unless they set a value."""

    from_state: str
    to_state: str
    probability: NumericRef
    cost_increment: NumericRef = 0.0
    effect_increment: NumericRef = 0.0
    extra_increment: dict[str, NumericRef] = field(default_factory=dict)


@dataclass
class MarkovModel:
    """A cohort Markov model: states, transitions, and a time horizon."""

    id: str
    name: str
    states: list[MarkovState] = field(default_factory=list)
    transitions: list[MarkovTransition] = field(default_factory=list)
    cycles: int = 10
    half_cycle_correction: bool = True
    # Outcomes-table rows beyond the primary Cost/Effect pair (e.g. a
    # second Effect "LY"), as [{"id":..., "type":..., "label":...}, ...].
    # Mirrors ABMModel.extra_outcomes; see run_markov()'s docstring.
    extra_outcomes: list[dict[str, str]] = field(default_factory=list)


@dataclass
class MarkovResult:
    """Cohort simulation output: totals, per-cycle rewards, and the trace."""

    total_cost: float = 0.0
    total_effect: float = 0.0
    # cost_per_cycle/effect_per_cycle are DISCOUNTED (see run_markov()):
    # cost_per_cycle[t] = raw_cost_per_cycle[t] / (1 + discount_rate) ** t.
    # sum(cost_per_cycle) == total_cost EXCEPT when half_cycle_correction
    # is on, where total_cost additionally includes a final half-cycle
    # adjustment term that is not attributed to any single cycle here.
    cost_per_cycle: list[float] = field(default_factory=list)
    effect_per_cycle: list[float] = field(default_factory=list)
    # Undiscounted counterparts of the two lists above -- same length,
    # same cycle indexing -- so cohort-trace reporting can show cost and
    # effect alongside their discounted values, matching what the
    # discount rate actually did to each cycle's reward.
    raw_cost_per_cycle: list[float] = field(default_factory=list)
    raw_effect_per_cycle: list[float] = field(default_factory=list)
    # Discounted / undiscounted totals for each extra Outcomes-table
    # outcome (e.g. "ly" -> 12.4), keyed by outcome id. Empty when the
    # model has no extra outcomes. Mirrors ABMResult's fields of the
    # same name.
    extra_totals: dict[str, float] = field(default_factory=dict)
    extra_totals_undiscounted: dict[str, float] = field(default_factory=dict)
    # One dict of {state_id: probability} per cycle, cycle 0 .. cycles (inclusive).
    trace: list[dict[str, float]] = field(default_factory=list)
    # The same per-cycle occupancy as ``trace``, but keyed by (state_id,
    # tenure) instead of collapsed to state_id -- i.e. exactly the engine's
    # own internal cohort distribution before it's summed across tenures
    # for ``trace``. Exists so a caller (the Excel transparency export,
    # ``markov_excel_export.py``) can build a live "by state x tenure"
    # Cohort Trace sheet, and cross-check its own resolved cost/effect
    # formulas against this module's own numbers, without re-implementing
    # ``run_markov()``'s transition/reward math itself. Same length as
    # ``trace`` (cycle 0 .. cycles inclusive); for a state that isn't
    # ``tenure_applied`` every entry lives under tenure 1.
    tenure_trace: list[dict[tuple[str, int], float]] = field(default_factory=list)
    errors: list[ValidationError] = field(default_factory=list)
