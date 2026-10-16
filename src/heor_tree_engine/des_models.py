"""Core data models for Discrete Event Simulation (DES) models.

Node vocabulary:

    Decision (strategies) -> DES -> Time -> [event branch] -> Chance* -> Terminal

* ``DESNode``   -- start of one DES model (one per strategy). Holds the run
  settings (time horizon, N patients, seed, trackers). Its branches are the
  possible STARTING states (a probability each, like a Markov node's
  initial distribution) and lead to a TimeNode or an absorbing Terminal.
* ``TimeNode``  -- one health state. Its branches are COMPETING EVENTS:
  each carries a time-to-event (``TimeToEvent``) instead of a probability,
  and the patient follows the event with the shortest sampled time
  (TreeAge "Time node", 43.4).
* ``ChanceNode`` (reused as-is from the Tree engine) -- a probabilistic
  consequence after an event.
* ``TerminalNode`` (DES flavour, ``DESTerminal``) -- ends an event path:
  ``jump_to`` a TimeNode (back into the model, TreeAge "return to state")
  or, with no ``jump_to``, absorbing (e.g. Dead) -- the patient stops.

Rewards (TreeAge 43.5 "Accumulating value -- time and event rewards"):

* time rewards  -- ``TimeNode.cost_rate``/``effect_rate``: value per ONE
  time unit, multiplied by the time actually spent at that node;
* event rewards -- a branch's ``cost_increment``/``effect_increment`` and a
  Terminal's ``cost``/``effect``: accrued once, no time adjustment;
* startup rewards -- ``TimeNode.startup_cost``/``startup_effect``: accrued
  once at time 0, only in the state the patient starts in.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .microsim_models import Tracker
from .models import NumericRef, ValidationError  # noqa: F401 (re-exported)

# Time-to-event distributions. The first 5 share parameter names with PSM
# curves; DES generalized gamma retains SciPy/Stacy's a/c/scale form while
# PSM uses the publication-standard Prentice mu/sigma/Q parameterization.
DES_TTE_PARAMS: dict[str, tuple[str, ...]] = {
    "exponential": ("rate",),
    "weibull": ("shape", "scale"),
    "log_normal": ("meanlog", "sdlog"),
    "log_logistic": ("shape", "scale"),
    "gompertz": ("shape", "rate"),
    "generalized_gamma": ("a", "c", "scale"),
    # A Distribution defined (with a name) in the document's Distributions
    # section: params {"name": ...}. The preferred form -- the event only
    # stores the distribution's NAME, its type and
    # parameters live in one place and can be reused, varied in SA/PSA, etc.
    "named": ("name",),
    # A constant time: params {"time": ...}.
    "fixed": ("time",),
    # Any DECIDE expression evaluated in the patient's current context
    # (trackers, _time, _event_time, ...): params {"value": "..."}. This is
    # TreeAge's "time-to-event expression" form, e.g.
    # "t_Age_BackgroundDeath - age".
    "expression": ("value",),
    # TableToTime = TreeAge's "TableProb" distribution (43.8), renamed so the
    # name says what it returns (a time), like DistToTime/DistToProb: sample from a table (e.g. a
    # life table). params {"table", "index_column", "value_column"}; see
    # TimeToEvent.prob_type / interpolate below.
    "table_to_time": ("table", "index_column", "value_column"),
}

DES_CLOCKS = ("reset", "forward")

# Distributions-section types sampled with DECIDE's own closed-form
# survival functions (no scipy): type -> (time-to-event key, parameter names).
# Any other continuous type falls back to scipy's sf/isf.
NAMED_DISTRIBUTION_CLOSED_FORMS: dict[str, tuple[str, tuple[str, ...]]] = {
    "Exponential": ("exponential", ("rate",)),
    "Weibull": ("weibull", ("shape", "scale")),
    "Lognormal": ("log_normal", ("meanlog", "sdlog")),
    "Log-logistic": ("log_logistic", ("shape", "scale")),
    "Gompertz": ("gompertz", ("shape", "rate")),
    "Generalized Gamma": ("generalized_gamma", ("a", "c", "scale")),
}
NAMED_DISTRIBUTION_UNSUITABLE = {
    "Bernoulli", "Binomial", "Categorical", "Dirichlet", "Discrete Uniform", "Geometric",
    "Hypergeometric", "Multinomial", "Multivariate Normal", "Negative Binomial", "Poisson",
}
TABLE_TO_TIME_TYPES = ("probability", "cumulative", "conditional")


@dataclass
class TimeToEvent:
    """How long until an event happens, sampled per patient.

    clock:
      * "reset"   -- time measured from entering the current Time node
        (semi-Markov; a fresh draw on every visit). Default.
      * "forward" -- time measured on the patient's model clock (+
        ``time_offset``), sampled CONDITIONAL on the event not having
        happened yet: TreeAge's ``DistSampleCond(dist; lower)`` (43.9).
        With ``time_offset = "startAge"`` a mortality table indexed by age
        is sampled from the patient's current age, and the time-to-event
        is ``sampled_age - current_age`` -- exactly TreeAge's
        ``Time_BackgroundDeath = t_Age_BackgroundDeath - age``.

    sample_once: draw the (absolute) event time once per patient and reuse
      it every time the patient is back at a Time node carrying this event
      (TreeAge's advice to store DistSampleCond's result in a tracker so it
      is "only called once per person"). Implies clock="forward".
    """

    distribution: str
    params: dict[str, NumericRef] = field(default_factory=dict)
    clock: str = "reset"
    time_offset: NumericRef = 0.0
    sample_once: bool = False
    # table_to_time only
    prob_type: str = "conditional"
    interpolate: bool = True


@dataclass
class DESBranch:
    """An edge in a DES subtree. Which fields matter depends on the parent:
    DESNode branch -> ``probability`` (starting-state share); TimeNode
    branch -> ``time_to_event`` (a competing event); ChanceNode branch ->
    ``probability``. ``cost_increment``/``effect_increment`` are EVENT
    rewards on every kind. ``tracker_updates`` ({tracker: formula}) run
    when a patient traverses the branch (e.g. {"n_relapse": "n_relapse + 1"})."""

    label: str
    child_id: str
    probability: NumericRef | None = None
    time_to_event: TimeToEvent | None = None
    # Logic node branches only: a true/false formula (e.g. "n_relapse >= 3").
    # The FIRST branch whose condition is true is taken; "#" (or "else")
    # marks the default branch taken when no other condition holds.
    condition: NumericRef | None = None
    cost_increment: NumericRef | None = None
    effect_increment: NumericRef | None = None
    tracker_updates: dict[str, NumericRef] = field(default_factory=dict)
    # Event rewards for EXTRA Cost/Effect outcomes (Outcomes table "+"),
    # keyed by outcome id, like the Tree's Branch.extra_outcomes.
    extra_outcomes: dict[str, NumericRef] = field(default_factory=dict)


@dataclass
class DESNode:
    id: str
    name: str = "DES"
    branches: list[DESBranch] = field(default_factory=list)
    time_horizon: NumericRef = 40.0
    n_patients: int = 1000
    random_seed: int = 1
    trackers: list[Tracker] = field(default_factory=list)
    # Number of patients (first N by index) whose full event history is
    # kept in DESResult.event_log_sample (Patient Timeline view).
    event_log_sample_size: int = 0
    # Occupancy trace grid spacing (time units) for DESResult.trace.
    trace_step: float = 1.0
    # Safety stop against a model that never reaches an absorbing state or
    # the horizon (e.g. a zero time-to-event loop).
    max_events_per_patient: int = 10_000
    # Optional stop rule (TreeAge "Termination condition"), e.g.
    # "startAge + _time >= 100". A patient stops as soon as it is true --
    # checked on entering every Time node and, between events, located
    # exactly in time (the only thing that changes between two events is
    # _time), so time rewards stop at the right moment. The time horizon
    # still applies on top.
    termination_condition: NumericRef | None = None


@dataclass
class TimeNode:
    id: str
    name: str = ""
    branches: list[DESBranch] = field(default_factory=list)
    cost_rate: NumericRef = 0.0
    effect_rate: NumericRef = 0.0
    startup_cost: NumericRef = 0.0
    startup_effect: NumericRef = 0.0
    # Extra Cost/Effect outcomes (Outcomes table "+"), keyed by outcome id:
    # per-year time rewards and one-off startup rewards.
    extra_rates: dict[str, NumericRef] = field(default_factory=dict)
    extra_startup: dict[str, NumericRef] = field(default_factory=dict)
    # Tracker updates applied every time a patient ENTERS this state
    # (TreeAge "tracker modification" at a node), e.g. {"n_severe": "n_severe + 1"}.
    tracker_updates: dict[str, NumericRef] = field(default_factory=dict)


@dataclass
class DESChance:
    id: str
    branches: list[DESBranch] = field(default_factory=list)


@dataclass
class DESTerminal:
    id: str
    name: str = ""
    cost: NumericRef = 0.0
    effect: NumericRef = 0.0
    jump_to: str | None = None  # a TimeNode id; None = absorbing
    extra_outcomes: dict[str, NumericRef] = field(default_factory=dict)
    # Tracker updates applied when a patient reaches this Terminal.
    tracker_updates: dict[str, NumericRef] = field(default_factory=dict)


@dataclass
class DESLogic:
    """TreeAge Logic node: branches carry true/false conditions instead of
    probabilities; the first true one is followed ("#"/"else" = default).
    Used after an event, like a Chance node, e.g. to send patients with
    n_relapse >= 3 to a different treatment line."""

    id: str
    branches: list[DESBranch] = field(default_factory=list)


DESNodeType = DESNode | TimeNode | DESChance | DESLogic | DESTerminal

ELSE_MARKERS = {"#", "else", "otherwise"}


@dataclass
class DESModel:
    """One strategy's DES subtree, flattened: the DESNode plus every node
    reachable from it, by id."""

    des: DESNode
    nodes: dict[str, DESNodeType] = field(default_factory=dict)
    # Outcomes table: Cost/Effect outcome id -> display symbol (e.g.
    # "effect_2" -> "QALY"), for the _total_<symbol>/_stage_<symbol>
    # keywords.
    outcome_symbols: dict[str, str] = field(default_factory=dict)


@dataclass
class DESEventRecord:
    time: float
    from_state: str
    event: str
    to_state: str | None  # None = absorbed/ended


@dataclass
class DESResult:
    """Patient-level DES output; totals are means across patients."""

    total_cost: float = 0.0
    total_effect: float = 0.0
    total_cost_undiscounted: float = 0.0
    total_effect_undiscounted: float = 0.0
    # Means across patients for every extra Cost/Effect outcome id.
    extra_totals: dict[str, float] = field(default_factory=dict)
    extra_totals_undiscounted: dict[str, float] = field(default_factory=dict)
    patient_costs: list[float] = field(default_factory=list)
    patient_effects: list[float] = field(default_factory=list)
    # Mean number of times each event (TimeNode branch, keyed
    # "<time node id>:<branch label>") fired per patient.
    event_counts: dict[str, float] = field(default_factory=dict)
    # Mean undiscounted time spent at each TimeNode per patient.
    time_in_state: dict[str, float] = field(default_factory=dict)
    # Share of patients at each TimeNode (or absorbed, keyed by the
    # absorbing Terminal's id) at t = 0, trace_step, 2*trace_step, ... .
    trace_times: list[float] = field(default_factory=list)
    trace: list[dict[str, float]] = field(default_factory=list)
    mean_time_simulated: float = 0.0
    tracker_summary: dict[str, dict[str, float]] = field(default_factory=dict)
    event_log_sample: list[list[DESEventRecord]] = field(default_factory=list)
    # One row per patient (TreeAge trial output): patient, cost, effect
    # (discounted + undiscounted), extra outcomes by id, time simulated,
    # number of events, final state id (or "Terminated") and final tracker
    # values.
    patient_rows: list[dict[str, Any]] = field(default_factory=list)
    # Step-by-step log of one patient (run_des(trace_patient=k)).
    calculation_trace: list[dict[str, Any]] = field(default_factory=list)
    # Per competing event ("<time node id>:<label>"): (entry, exit, happened)
    # on the event's own time axis, for the simulated-vs-input survival check.
    event_exposures: dict[str, list[tuple[float, float, bool]]] = field(default_factory=dict)
    errors: list[ValidationError] = field(default_factory=list)
