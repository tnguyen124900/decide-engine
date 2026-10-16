"""Build a NetworkMetaAnalysis from JSON or from flat CSV-style rows.

JSON document shape::

    {
      "id": "nma1",
      "name": "Smoking cessation",
      "spec": {"outcome": "binary", "measure": "OR", "reference": "Placebo"},
      "treatments": [{"name": "Placebo", "aliases": ["PBO"], "class": "Control"}],
      "studies": [
        {"id": "S1", "arms": [{"treatment": "Placebo", "events": 18, "n": 120},
                               {"treatment": "Drug A", "events": 11, "n": 118}]},
        {"id": "S9", "contrasts": [{"treatment": "Drug Z", "versus": "Placebo",
                                     "effect": 0.70, "lower": 0.55, "upper": 0.89}]}
      ]
    }

Treatment names in studies are matched to `treatments` (and their
aliases) case-insensitively, ignoring spaces, hyphens and underscores. When
`treatments` is omitted the list is built from the data in order of first
appearance. Names that cannot be matched are kept as written so that
validation can report them (with a "did you mean" suggestion).
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping
from typing import Any

from .nma_models import (
    DEFAULT_MEASURE,
    Arm,
    Contrast,
    NetworkMetaAnalysis,
    NMASpec,
    Study,
    Treatment,
    is_ratio_measure,
)

Z_95 = 1.959963984540054


def normalize_name(name: str) -> str:
    """Key used to match spelling variants: 'Drug-A ' == 'drug a' == 'DRUG_A'."""

    return re.sub(r"[\s\-_]+", "", str(name)).casefold()


def build_alias_map(treatments: Iterable[Treatment]) -> dict[str, str]:
    alias_map: dict[str, str] = {}
    for treatment in treatments:
        for variant in [treatment.name, *treatment.aliases]:
            alias_map.setdefault(normalize_name(variant), treatment.name)
    return alias_map


def _optional_float(raw: Mapping[str, Any], key: str) -> float | None:
    value = raw.get(key)
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise ValueError(f"'{key}' must be a number")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"'{key}' must be a number (got {value!r})") from exc


def se_from_ci(lower: float, upper: float, ratio: bool, level: float = 0.95) -> float:
    """Standard error on the analysis scale from a symmetric CI.

    For ratio measures the CI is on the natural scale, so logs are taken:
    se = (ln(upper) - ln(lower)) / (2 * z).
    """

    z = Z_95 if abs(level - 0.95) < 1e-12 else _normal_quantile(0.5 + level / 2)
    if ratio:
        if lower <= 0 or upper <= 0:
            raise ValueError("Confidence limits of a ratio measure must be > 0")
        return (math.log(upper) - math.log(lower)) / (2 * z)
    return (upper - lower) / (2 * z)


def _normal_quantile(p: float) -> float:
    # Acklam's rational approximation; accurate to ~1e-9, enough for CI levels.
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00]
    if not 0 < p < 1:
        raise ValueError("p must be in (0, 1)")
    if p < 0.02425:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > 1 - 0.02425:
        return -_normal_quantile(1 - p)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (
        ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def _parse_spec(raw: Mapping[str, Any]) -> NMASpec:
    outcome = str(raw.get("outcome", "binary"))
    measure = str(raw.get("measure", DEFAULT_MEASURE.get(outcome, "OR"))).upper()
    return NMASpec(
        outcome=outcome,
        measure=measure,
        reference=(str(raw["reference"]) if raw.get("reference") not in (None, "") else None),
        small_values=str(raw.get("small_values", "good")),
        zero_correction=float(raw.get("zero_correction", 0.5)),
        ci_level=float(raw.get("ci_level", 0.95)),
    )


def _parse_treatment(raw: Mapping[str, Any] | str) -> Treatment:
    if isinstance(raw, str):
        return Treatment(name=raw)
    if "name" not in raw:
        raise ValueError("A treatment requires a 'name'")
    return Treatment(
        name=str(raw["name"]).strip(),
        aliases=[str(alias) for alias in raw.get("aliases", [])],
        treatment_class=(str(raw["class"]) if raw.get("class") else None),
        color=(str(raw["color"]) if raw.get("color") else None),
    )


def _parse_arm(raw: Mapping[str, Any]) -> Arm:
    if not raw.get("treatment"):
        raise ValueError("An arm requires a 'treatment'")
    return Arm(
        treatment=str(raw["treatment"]).strip(),
        n=_optional_float(raw, "n"),
        events=_optional_float(raw, "events"),
        mean=_optional_float(raw, "mean"),
        sd=_optional_float(raw, "sd"),
        exposure=_optional_float(raw, "exposure"),
        source=(str(raw["source"]) if raw.get("source") else None),
    )


def _parse_contrast(raw: Mapping[str, Any], spec: NMASpec) -> Contrast:
    """Accepts either `estimate` + `se` (analysis scale) or `effect` +
    `lower` + `upper` (natural scale, e.g. HR 0.70 [0.55, 0.89])."""

    for key in ("treatment", "versus"):
        if not raw.get(key):
            raise ValueError(f"A contrast requires '{key}'")
    ratio = is_ratio_measure(spec.measure)
    estimate = _optional_float(raw, "estimate")
    se = _optional_float(raw, "se")
    if estimate is None:
        effect = _optional_float(raw, "effect")
        if effect is None:
            raise ValueError("A contrast requires 'estimate' + 'se' or 'effect' + 'lower' + 'upper'")
        if ratio:
            if effect <= 0:
                raise ValueError("A ratio effect must be > 0")
            estimate = math.log(effect)
        else:
            estimate = effect
    if se is None:
        lower = _optional_float(raw, "lower")
        upper = _optional_float(raw, "upper")
        if lower is None or upper is None:
            raise ValueError("A contrast requires 'se' or both 'lower' and 'upper'")
        se = se_from_ci(lower, upper, ratio, spec.ci_level)
    return Contrast(
        treatment=str(raw["treatment"]).strip(),
        versus=str(raw["versus"]).strip(),
        estimate=estimate,
        se=se,
        source=(str(raw["source"]) if raw.get("source") else None),
    )


def _parse_study(raw: Mapping[str, Any], spec: NMASpec, index: int) -> Study:
    return Study(
        id=str(raw.get("id", f"study_{index + 1}")),
        arms=[_parse_arm(arm) for arm in raw.get("arms", [])],
        contrasts=[_parse_contrast(contrast, spec) for contrast in raw.get("contrasts", [])],
        baseline_se=_optional_float(raw, "baseline_se"),
        source=(str(raw["source"]) if raw.get("source") else None),
        characteristics=dict(raw.get("characteristics", {})),
    )


def canonicalize_treatments(model: NetworkMetaAnalysis) -> NetworkMetaAnalysis:
    """Rename study treatments to their canonical names (in place) and, when
    the treatment list is empty, build it from the data."""

    if not model.treatments:
        seen: dict[str, str] = {}
        for study in model.studies:
            for name in study.treatments:
                seen.setdefault(normalize_name(name), name)
        model.treatments = [Treatment(name=name) for name in seen.values()]
    alias_map = build_alias_map(model.treatments)

    def canonical(name: str) -> str:
        return alias_map.get(normalize_name(name), name)

    for study in model.studies:
        for arm in study.arms:
            arm.treatment = canonical(arm.treatment)
        for contrast in study.contrasts:
            contrast.treatment = canonical(contrast.treatment)
            contrast.versus = canonical(contrast.versus)
    if model.spec.reference:
        model.spec.reference = canonical(model.spec.reference)
    return model


def resolve_nma(document: Mapping[str, Any]) -> NetworkMetaAnalysis:
    """Parse an NMA JSON document into a NetworkMetaAnalysis."""

    if not isinstance(document.get("studies"), list):
        raise ValueError("NMA document requires a 'studies' list")
    spec = _parse_spec(document.get("spec", {}))
    model = NetworkMetaAnalysis(
        id=str(document.get("id", "nma")),
        name=str(document.get("name", "Network meta-analysis")),
        treatments=[_parse_treatment(raw) for raw in document.get("treatments", [])],
        studies=[_parse_study(raw, spec, index) for index, raw in enumerate(document["studies"])],
        spec=spec,
    )
    return canonicalize_treatments(model)


# --------------------------------------------------------------------------
# Flat rows (CSV / Excel import): one row per arm or per contrast
# --------------------------------------------------------------------------


def studies_from_arm_rows(rows: Iterable[Mapping[str, Any]]) -> list[Study]:
    """Group arm rows (`study`, `treatment`, `events`, `n`, ...) into studies,
    keeping the order in which studies first appear."""

    studies: dict[str, Study] = {}
    for index, row in enumerate(rows):
        study_id = str(row.get("study", "")).strip()
        if not study_id:
            raise ValueError(f"Row {index + 1}: 'study' is required")
        study = studies.setdefault(study_id, Study(id=study_id))
        study.arms.append(_parse_arm(row))
    return list(studies.values())


def studies_from_contrast_rows(rows: Iterable[Mapping[str, Any]], spec: NMASpec) -> list[Study]:
    """Group contrast rows (`study`, `treatment`, `versus` [or `comparator`],
    `estimate`/`se` or `effect`/`lower`/`upper`, optional `baseline_se`)."""

    studies: dict[str, Study] = {}
    for index, row in enumerate(rows):
        study_id = str(row.get("study", "")).strip()
        if not study_id:
            raise ValueError(f"Row {index + 1}: 'study' is required")
        normalized = dict(row)
        if "versus" not in normalized and "comparator" in normalized:
            normalized["versus"] = normalized["comparator"]
        study = studies.setdefault(study_id, Study(id=study_id))
        study.contrasts.append(_parse_contrast(normalized, spec))
        baseline_se = _optional_float(normalized, "baseline_se")
        if baseline_se is not None:
            study.baseline_se = baseline_se
    return list(studies.values())
