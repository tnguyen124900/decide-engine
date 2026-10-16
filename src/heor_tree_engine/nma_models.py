"""Core data models for network meta-analysis (NMA) of randomised trials.

An NMA pools direct and indirect comparisons from a connected network of
RCTs to estimate the relative effect of every treatment against a common
reference. Phase 1 of DECIDE's NMA engine covers RCT-only networks with:

- arm-level binary data (events / total)      -> log odds ratio, log risk
  ratio or risk difference
- arm-level continuous data (mean, sd, n)     -> mean difference or
  standardised mean difference (Hedges' g)
- arm-level count data (events / person-time) -> log rate ratio
- contrast-level data (estimate + se)         -> any measure, e.g. log HR

Relative effects are always estimated on the *analysis scale* (log scale
for ratio measures, identity for MD); results also carry the natural scale.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .models import ValidationError  # noqa: F401 (re-exported)

OUTCOME_TYPES = ("binary", "continuous", "rate", "contrast")

# Default effect measure per outcome type (arm-level data).
DEFAULT_MEASURE = {
    "binary": "OR",
    "continuous": "MD",
    "rate": "RATE_RATIO",
}

# Measures analysed on the log scale (results exponentiated for reporting).
RATIO_MEASURES = ("OR", "RR", "HR", "RATE_RATIO")
DIFFERENCE_MEASURES = ("MD", "SMD", "RD")
MEASURES = RATIO_MEASURES + DIFFERENCE_MEASURES

# Which measures each arm-level outcome type can produce in Phase 1.
ARM_LEVEL_MEASURES = {
    "binary": ("OR", "RR", "RD"),
    "continuous": ("MD", "SMD"),
    "rate": ("RATE_RATIO",),
}

SMALL_VALUES = ("good", "bad")


def is_ratio_measure(measure: str) -> bool:
    return measure in RATIO_MEASURES


@dataclass
class Treatment:
    """A node of the network. `aliases` map spelling variants onto `name`."""

    name: str
    aliases: list[str] = field(default_factory=list)
    treatment_class: str | None = None
    color: str | None = None


@dataclass
class Arm:
    """One randomised arm of a study (arm-level data).

    Which fields are required depends on the outcome type:
    binary -> events, n; continuous -> mean, sd, n; rate -> events, exposure.
    """

    treatment: str
    n: float | None = None
    events: float | None = None
    mean: float | None = None
    sd: float | None = None
    exposure: float | None = None
    source: str | None = None


@dataclass
class Contrast:
    """A reported relative effect of `treatment` versus `versus`.

    `estimate` and `se` are on the analysis scale (log scale for ratio
    measures). Use nma_parser to convert a point estimate + 95% CI.
    """

    treatment: str
    versus: str
    estimate: float
    se: float
    source: str | None = None


@dataclass
class Study:
    """A randomised trial: either arm-level `arms` or contrast-level `contrasts`.

    For a contrast-level multi-arm study (> 1 contrast), every contrast must
    share the same `versus` arm and `baseline_se` - the standard error of the
    outcome in that shared arm - is required to build the correct covariance.
    """

    id: str
    arms: list[Arm] = field(default_factory=list)
    contrasts: list[Contrast] = field(default_factory=list)
    baseline_se: float | None = None
    source: str | None = None
    characteristics: dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0  # evidence weight in (0, 1]: variances are divided by it (down-weighting, e.g. non-randomised studies)

    @property
    def is_contrast_level(self) -> bool:
        return bool(self.contrasts) and not self.arms

    @property
    def treatments(self) -> list[str]:
        if self.is_contrast_level:
            names: list[str] = []
            for contrast in self.contrasts:
                for name in (contrast.versus, contrast.treatment):
                    if name not in names:
                        names.append(name)
            return names
        return [arm.treatment for arm in self.arms]


@dataclass
class NMASpec:
    """Analysis settings."""

    outcome: str = "binary"  # one of OUTCOME_TYPES
    measure: str = "OR"  # one of MEASURES
    reference: str | None = None  # defaults to the first treatment
    small_values: str = "good"  # are lower effects better? ("good" | "bad")
    zero_correction: float = 0.5  # added to every cell of a study with a zero cell
    ci_level: float = 0.95


@dataclass
class NetworkMetaAnalysis:
    """A complete NMA document: treatments, studies and settings."""

    id: str
    name: str
    treatments: list[Treatment] = field(default_factory=list)
    studies: list[Study] = field(default_factory=list)
    spec: NMASpec = field(default_factory=NMASpec)

    @property
    def treatment_names(self) -> list[str]:
        return [treatment.name for treatment in self.treatments]

    @property
    def reference(self) -> str | None:
        if self.spec.reference:
            return self.spec.reference
        return self.treatments[0].name if self.treatments else None


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------


@dataclass
class ContrastObservation:
    """One study-level contrast actually entered into the model."""

    study_id: str
    treatment: str
    versus: str
    estimate: float  # analysis scale
    variance: float


@dataclass
class RelativeEffect:
    """Effect of `treatment` versus `versus` (analysis + natural scale)."""

    treatment: str
    versus: str
    estimate: float
    se: float
    lower: float
    upper: float
    natural: float
    natural_lower: float
    natural_upper: float
    # Random-effects prediction interval (where a new study's true effect is
    # expected to lie); None for fixed-effect fits or when df < 1.
    pred_lower: float | None = None
    pred_upper: float | None = None
    natural_pred_lower: float | None = None
    natural_pred_upper: float | None = None


@dataclass
class NMAFit:
    """One fitted model (fixed or random effects)."""

    model: str  # "fixed" | "random"
    tau2: float
    # Effects of every treatment versus the reference (reference excluded).
    effects: list[RelativeEffect] = field(default_factory=list)
    # Every ordered pair (treatment, versus), treatment != versus.
    league: dict[tuple[str, str], RelativeEffect] = field(default_factory=dict)
    # Frequentist analogue of SUCRA (Ruecker & Schwarzer 2015), in [0, 1].
    p_scores: dict[str, float] = field(default_factory=dict)
    # Variance-covariance of the basic parameters d (analysis scale), in the
    # order of `parameter_names`.
    parameter_names: list[str] = field(default_factory=list)
    covariance: list[list[float]] = field(default_factory=list)


@dataclass
class Heterogeneity:
    q: float = 0.0
    df: int = 0
    p_value: float | None = None
    i2: float | None = None
    tau2: float = 0.0
    tau: float = 0.0
    # Degrees of freedom of the t distribution used for prediction
    # intervals: studies - treatments (Riley et al. 2011's k - 2 for a
    # pairwise meta-analysis, generalised). None when < 1.
    prediction_df: int | None = None


@dataclass
class NMAResult:
    reference: str | None = None
    measure: str = "OR"
    treatments: list[str] = field(default_factory=list)
    fixed: NMAFit | None = None
    random: NMAFit | None = None
    heterogeneity: Heterogeneity = field(default_factory=Heterogeneity)
    observations: list[ContrastObservation] = field(default_factory=list)
    excluded_studies: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[ValidationError] = field(default_factory=list)
