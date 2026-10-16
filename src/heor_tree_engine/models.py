"""Core data models for the HEOR decision-tree engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TypeAlias


NumericRef: TypeAlias = float | str


COMPLEMENT_MARKERS = {"#", "c"}


def is_complement(value: NumericRef) -> bool:
    """Return True if value is the reserved marker meaning "the remaining
    probability needed so all sibling branches of a chance node sum to 1"
    (accepts "#" or "C"/"c", TreeAge's complement-probability convention).
    """

    return isinstance(value, str) and value.strip().lower() in COMPLEMENT_MARKERS


@dataclass(frozen=True)
class Param:
    """A model parameter with a required base value and optional uncertainty data."""

    name: str
    base: float
    low: float | None = None
    high: float | None = None
    dist: str | None = None
    dist_args: dict[str, float] | None = None
    # Free-text citation/source for this parameter's value (e.g. "Smith 2023,
    # Table 2"). Optional and purely documentary -- never read by any engine
    # calculation -- but surfaced in the transparency exports (Excel/Technical
    # Documentation Report) so a reviewer can trace every input back to
    # evidence. None for existing models/callers that predate this field.
    source: str | None = None


@dataclass
class Branch:
    """An edge from a decision or chance node to a child node."""

    label: str
    child: Node | None = None
    child_id: str | None = None
    probability: NumericRef | None = None
    cost_increment: NumericRef | None = None
    effect_increment: NumericRef | None = None
    # Extra Cost/Effect outcomes beyond the primary pair above, keyed by
    # the outcome's id from document["outcomes"] (e.g. "cost_2"). Empty
    # for every model that hasn't added extra outcomes -- see
    # Engine.run()'s outcome_kinds parameter for how these are summed.
    extra_outcomes: dict[str, NumericRef] = field(default_factory=dict)
    # Set only on a PSMNode's 3 fixed branches: "pre_progression" |
    # "post_progression" | "dead" (see PSM_STATE_ROLES). None (the
    # default) for every Decision/Chance branch, which is every branch
    # that existed before PSM nodes did -- fully backward compatible.
    role: str | None = None


@dataclass
class Node:
    """Base class for all decision-tree nodes."""

    id: str


@dataclass
class DecisionNode(Node):
    """A node representing mutually exclusive strategies."""

    branches: list[Branch] = field(default_factory=list)


@dataclass
class ChanceNode(Node):
    """A node whose branches are selected according to probabilities."""

    branches: list[Branch] = field(default_factory=list)


@dataclass
class TerminalNode(Node):
    """A terminal outcome with cost and QALY effect. ``year`` is the number
    of years from the decision point at which this payoff is realized, used
    to discount it (0.0 by default, meaning "no discounting" -- the
    behaviour every model had before discounting existed)."""

    cost: NumericRef = 0.0
    effect: NumericRef = 0.0
    year: NumericRef = 0.0
    # Extra Cost/Effect outcomes beyond the primary pair above, keyed by
    # the outcome's id from document["outcomes"] (e.g. "cost_2"). Empty
    # for every model that hasn't added extra outcomes.
    extra_outcomes: dict[str, NumericRef] = field(default_factory=dict)


PSM_STATE_ROLES = ("pre_progression", "post_progression", "dead")

# How many of each unit fit in one year -- PSMNode's occupancy loop always
# works in elapsed years (cycle / cycles_per_year), so a curve's own raw
# parameters (as published) get multiplied by this factor before being
# handed to the distribution formula. Design priority: type a paper's
# shape/scale/mu exactly as published, in whatever unit THAT paper fit
# them in, with no hand conversion to years first (2026-10-01, after the
# KEYNOTE-189/Cai et al. 2021 validation hit exactly this -- the paper's
# curves are fit in months, DECIDE always ran its curve axis in years).
PSM_CURVE_TIME_UNITS: dict[str, float] = {
    "year": 1.0,
    "month": 12.0,
    "week": 365.25 / 7.0,
    "day": 365.25,
}
PSM_CURVE_TIME_UNIT_DEFAULT = "year"


def curve_time_in_native_unit(curve: "SurvivalCurve", time_years: float) -> float:
    """Elapsed time in years -> elapsed time in `curve`'s own time_unit, the
    unit its distribution parameters were fit/published in. Unknown/missing
    units fall back to "year" (the engine's original, only-ever behaviour),
    so every PSM model saved before this field existed keeps running
    unchanged."""

    factor = PSM_CURVE_TIME_UNITS.get(curve.time_unit, 1.0)
    return time_years * factor


@dataclass
class SurvivalCurve:
    """A parametric survival function S(t) = P(event has not occurred by
    t), used by a PSMNode's pfs_curve/os_curve. `t` is in `time_unit`
    (default "year", see PSM_CURVE_TIME_UNITS) -- the engine's own
    occupancy loop always tracks elapsed time in years and converts via
    curve_time_in_native_unit() right before evaluating the curve, so
    params can be typed exactly as a paper publishes them (e.g. a
    log-logistic scale fit in months) without hand-converting to years.
    Mirrors psm_models.py's SurvivalCurve (kept there too, for the legacy
    standalone PSM engine until it's retired -- that copy is untouched by
    this field, since it isn't reachable from the live PSM Editor UI)."""

    distribution: str  # "exponential" | "weibull" for now (see Phase 3)
    params: dict[str, NumericRef] = field(default_factory=dict)
    time_unit: str = PSM_CURVE_TIME_UNIT_DEFAULT


@dataclass
class PSMNode(Node):
    """A partitioned-survival node: behaves like a ChanceNode, but its
    exactly-3 branches (roles pre_progression/post_progression/dead, see
    PSM_STATE_ROLES) are split by occupancy under pfs_curve/os_curve at
    each cycle instead of by typed probabilities. Each branch's child is a
    TerminalNode whose cost/effect are read as a PER-CYCLE rate integrated
    over that occupancy (not the one-off lump sum a Terminal means
    elsewhere in the tree -- see engine.py's PSMNode evaluation)."""

    pfs_curve: SurvivalCurve = field(default_factory=lambda: SurvivalCurve("weibull", {"shape": 1.5, "scale": 8.0}))
    os_curve: SurvivalCurve = field(default_factory=lambda: SurvivalCurve("weibull", {"shape": 1.8, "scale": 12.0}))
    cycles: int = 10
    # How many cycles make up one year (e.g. 12 for monthly cycles) --
    # cycle index t is converted to t / cycles_per_year before being fed
    # to pfs_curve/os_curve (a survival curve's t is always in years),
    # so cost/effect stay a per-CYCLE rate while the curves themselves
    # are unaffected by cycle length. Defaults to 1.0 (1 cycle = 1 year),
    # this node's behavior before cycles_per_year existed.
    cycles_per_year: float = 1.0
    half_cycle_correction: bool = True
    branches: list[Branch] = field(default_factory=list)


@dataclass
class DESTreeNode(Node):
    """A Discrete Event Simulation model used as ONE node of a decision
    tree (same architecture as PSMNode):
    it can sit after a Decision branch (one DES per strategy) or after a
    Chance node, and rolls back like any other node, returning the mean
    (discounted) cost/effect across its simulated patients.

    ``model`` is the des_models.DESModel parsed from the flat document
    (the DES node plus its Time/Chance/Terminal subtree, which therefore
    do NOT appear as ordinary Tree nodes). ``variables`` keeps the
    document's raw variable formulas so per-patient ones (e.g.
    "age = startAge + _time") are evaluated inside the simulation.
    Typed as Any to avoid a models <-> des_models import cycle."""

    model: Any = None
    variables: dict[str, Any] = field(default_factory=dict)


@dataclass
class MarkovTreeNode(Node):
    """A cohort Markov model used as ONE node of a decision tree (TreeAge's
    "Markov node"). It sits after a Decision branch or a Chance node, owns
    its own states/transitions (same schema as one strategy of the Markov
    editor) and rolls back like a terminal: its value is the discounted
    cost/effect of the cohort over ``model.cycles``. The tree's parameters
    and variables feed the Markov formulas; the tree's ANNUAL discount
    rates are converted to per-cycle rates with ``cycles_per_year``.
    ``variables`` keeps the document's raw variable formulas so ones that
    use _cycle/_tenure are evaluated inside the cohort run. ``model`` is a
    markov_models.MarkovModel (typed Any to avoid an import cycle)."""

    model: Any = None
    cycles_per_year: float = 1.0
    variables: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ValidationError:
    """A structured validation problem found in a model."""

    code: str
    message: str
    node_id: str | None = None
    field: str | None = None


@dataclass
class Result:
    """A value returned by an operation together with errors and path metadata."""

    cost: float | None = None
    effect: float | None = None
    # Totals for any extra Cost/Effect outcomes beyond the primary pair,
    # keyed by outcome id. Empty unless the model defines extra outcomes.
    extra_totals: dict[str, float] = field(default_factory=dict)
    errors: list[ValidationError] = field(default_factory=list)
    paths: list[list[str]] = field(default_factory=list)


@dataclass
class DecisionResult:
    """Rollback results for every strategy available at a decision node."""

    strategies: dict[str, Result] = field(default_factory=dict)
    errors: list[ValidationError] = field(default_factory=list)
