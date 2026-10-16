"""Core data models for patient-level (microsimulation) HEOR models.

States and transitions are structurally identical to a Markov cohort
model (see markov_models.py) -- a Decision (root) fans into one root per
strategy which fans into States, the same canvas node vocabulary
confirmed reused for Microsimulation -- so this module reuses
MarkovState/MarkovTransition rather than duplicating them. The only
genuinely new concept is the Tracker: a per-patient value carried across
cycles.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .markov_models import MarkovState, MarkovTransition
from .models import PSM_STATE_ROLES, NumericRef, SurvivalCurve, ValidationError  # noqa: F401 (re-exported)

# Values accepted for MicrosimModel.tracker_reset_timing:
# "on_entry" applies a Tracker's reset_rules value starting the very
# cycle the patient enters the new state; "next_cycle" lets the normal
# update_formula run once more (using the old state's Tracker value)
# before the reset takes effect, one cycle later.
TRACKER_RESET_TIMINGS = ("on_entry", "next_cycle")

# Values accepted for a Distribution row's "resample" setting.
# "per_psa" is accepted for forward compatibility with the outer
# PSA harness, but within a single run_microsim() call (no PSA loop yet)
# it behaves exactly like "per_model" -- see
# microsim_engine._resample_distributions.
DISTRIBUTION_RESAMPLE_TIERS = ("per_model", "per_psa", "per_patient", "per_cycle")


@dataclass
class Tracker:
    """A per-patient value carried across cycles -- the architectural
    difference between a patient-level model and a Markov cohort, which
    only has the built-in `_cycle`/`_tenure`."""

    name: str
    initial_value: NumericRef = 0.0
    # A formula evaluated once per cycle, in the same context as
    # cost/effect and transition probabilities that cycle (see
    # microsim_engine.run_microsim's execution order docstring). Defaults
    # to the tracker's own name (i.e. a no-op that carries the previous
    # value forward unchanged) when not given.
    update_formula: NumericRef | None = None
    # state_id -> value/formula to use instead of the update_formula's
    # result when the patient enters that state -- see
    # MicrosimModel.tracker_reset_timing for *when* this takes effect.
    reset_rules: dict[str, NumericRef] = field(default_factory=dict)
    # Whether this tracker's per-patient final value is summarised in
    # MicrosimResult.tracker_summary (subgroup analysis / breakdown).
    include_in_output: bool = True


@dataclass
class PSMRoleOutcomes:
    """Cost/effect for one of a MicrosimPSMNode's 3 roles (PSM_STATE_ROLES)
    -- same per-cycle-rate convention as MarkovState.cost/effect (a PSM
    role, like a PSM Editor branch, is a rate integrated over time, not a
    lump sum), plus extra_outcomes for parity with
    MarkovState.extra_outcomes."""

    cost: NumericRef = 0.0
    effect: NumericRef = 0.0
    extra_outcomes: dict[str, NumericRef] = field(default_factory=dict)


@dataclass
class MicrosimPSMNode:
    """A patient-level partitioned-survival node: when a patient enters
    it, ONE pfs_time and ONE os_time are drawn for that patient alone
    (inverse-transform sampling from pfs_curve/os_curve -- see
    microsim_engine.run_microsim's PSM handling), letting the curves
    reference per-patient covariates the way no cohort-level PSM
    (PSMNode/PSM Editor) can. From then on the patient's role among
    PSM_STATE_ROLES is a deterministic function of elapsed time since
    entry versus those 2 sampled times -- no further transition draws
    for as long as they stay in this node."""

    id: str
    name: str
    pfs_curve: SurvivalCurve
    os_curve: SurvivalCurve
    roles: dict[str, PSMRoleOutcomes] = field(default_factory=dict)  # keyed by PSM_STATE_ROLES
    # Same meaning as MarkovState.initial_prob -- can a patient START
    # inside this PSM node at cycle 0 (role pre_progression)?
    initial_prob: NumericRef = 0.0


@dataclass
class MicrosimModel:
    """A patient-level (microsimulation) model: states, transitions, and
    trackers, simulated for `n_patients` independent patients."""

    id: str
    name: str
    states: list[MarkovState] = field(default_factory=list)
    transitions: list[MarkovTransition] = field(default_factory=list)
    cycles: int = 10
    half_cycle_correction: bool = True
    n_patients: int = 1000
    random_seed: int = 1
    tracker_reset_timing: str = "on_entry"
    trackers: list[Tracker] = field(default_factory=list)
    # How many of the n_patients (the first `patient_trace_sample_size` by
    # patient index, 0 = none) to additionally record a full per-cycle
    # state sequence for -- the "Patient Trace" Patient Journey view (see
    # the 2026-09-23 design mockup's Concept B). Capped independent of
    # n_patients so this stays cheap even when n_patients is large; 0
    # keeps run_microsim() exactly as before (no sequences kept).
    patient_trace_sample_size: int = 0
    # PSM nodes available as transition targets alongside `states` (a
    # MarkovTransition.to_state may name either one). Empty for every model that
    # hasn't added one, in which case run_microsim() is byte-identical to
    # before this field existed.
    psm_nodes: list[MicrosimPSMNode] = field(default_factory=list)


@dataclass
class MicrosimResult:
    """Patient-level simulation output. total_cost/total_effect/
    cost_per_cycle/effect_per_cycle/trace are all means across patients,
    so existing cohort-style result rendering can plot them unchanged --
    plus the per-patient samples and Tracker summaries a microsimulation
    additionally makes possible."""

    total_cost: float = 0.0
    total_effect: float = 0.0
    # cost_per_cycle/effect_per_cycle are means-across-patients of each
    # patient's DISCOUNTED cost/effect for that cycle (see run_microsim()).
    cost_per_cycle: list[float] = field(default_factory=list)
    effect_per_cycle: list[float] = field(default_factory=list)
    # Undiscounted counterparts of the two lists above -- same length,
    # same cycle indexing, also averaged across patients -- so
    # cohort-trace reporting can show cost and effect alongside their
    # discounted values, matching MarkovResult's equivalent fields.
    raw_cost_per_cycle: list[float] = field(default_factory=list)
    raw_effect_per_cycle: list[float] = field(default_factory=list)
    # One dict of {state_id: fraction of patients occupying it} per cycle,
    # cycle 0 .. cycles (inclusive) -- the patient-level analogue of
    # MarkovResult.trace.
    trace: list[dict[str, float]] = field(default_factory=list)
    # One dict of {from_state: {to_state: fraction of patients moving
    # from_state -> to_state}} per cycle (length `cycles`, one entry per
    # cycle -> cycle+1 step, unlike `trace` which has cycles+1 snapshots).
    # Includes self-loops (patients who stayed), so summing every to_state
    # for a from_state reproduces that from_state's occupancy at `trace[cycle]`.
    # Cheap to compute (a per-cycle tally already available in the patient
    # loop, no per-patient identity kept) -- powers the "Population flow"
    # Patient Journey view (straight-edged flow bands between cycle
    # columns) without needing per-patient trajectories.
    transition_counts: list[dict[str, dict[str, float]]] = field(default_factory=list)
    # Per-patient totals (post half-cycle-correction), for percentile/CI
    # reporting beyond the population mean above.
    patient_costs: list[float] = field(default_factory=list)
    patient_effects: list[float] = field(default_factory=list)
    # tracker_name -> {"mean", "p10", "p50", "p90"} of that tracker's
    # final per-patient value, for trackers with include_in_output=True.
    tracker_summary: dict[str, dict[str, float]] = field(default_factory=dict)
    # Full per-cycle state sequence (length cycles+1, same indexing as
    # `trace`) for the first `model.patient_trace_sample_size` patients,
    # in patient-index order -- empty unless that setting is >0. Powers
    # the "Patient Trace" Patient Journey view; deliberately NOT kept for
    # every patient (memory), only this bounded sample.
    sample_patient_traces: list[list[str]] = field(default_factory=list)
    errors: list[ValidationError] = field(default_factory=list)
