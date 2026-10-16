"""Import NMA study data from external systematic-review tools.

Two sources are supported, both producing the same arm-level shape the NMA
editor's study table uses (``studies[].arms[]`` with binary ``events/n`` or
continuous ``mean/sd/n``):

- **RevMan** -- a ``.rm5`` file (RevMan 5's XML export) or the CSV "data and
  analyses" export. Both carry dichotomous (events/total per arm) and
  continuous (mean/SD/total per arm) outcomes; multi-arm studies are split
  one row per arm under the study's own id.
- **Covidence** -- the data-extraction CSV export (one row per study x arm
  with columns for events/n or mean/SD/n).

Everything here is pure parsing into plain dicts -- no Streamlit, no
session state -- so the engine tests can exercise it directly and the UI
layer just hands the uploaded bytes over.
"""

from __future__ import annotations

import csv
import io
import re
import xml.etree.ElementTree as ET
from typing import Any

__all__ = [
    "NMAImportError",
    "detect_import_format",
    "parse_revman_rm5",
    "parse_revman_csv",
    "parse_covidence_csv",
    "studies_to_document_rows",
]


class NMAImportError(ValueError):
    """Raised when an uploaded file can't be parsed as the claimed source.

    Raised at parse time, before any document is touched, so a malformed
    upload is reported immediately with the specific column/element at
    fault rather than surfacing later as a confusing NMA error."""


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------


def detect_import_format(filename: str, content: bytes) -> str:
    """Best-effort source detection: "rm5", "covidence", "revman_csv".

    Used when the user picks a file without saying where it came from. The
    caller can still override with an explicit source choice; this just
    makes the common case zero-click.
    """

    name = filename.lower()
    if name.endswith(".rm5"):
        return "rm5"
    text = content.decode("utf-8-sig", errors="replace")
    if text.lstrip().startswith("<?xml") or "<rm5" in text[:400].lower():
        return "rm5"
    header = text.split("\n", 1)[0].lower()
    # Covidence extraction CSVs carry a stable set of columns.
    if "covidence" in name or "intervention" in header and "outcome" in header:
        return "covidence"
    # RevMan's own CSV export names its columns differently from Covidence's.
    if "study" in header and ("subgroup" in header or "experimental" in header):
        return "revman_csv"
    raise NMAImportError(
        "Couldn't tell whether this file is RevMan (.rm5 / CSV) or Covidence "
        "CSV -- choose the source explicitly."
    )


# ---------------------------------------------------------------------------
# Shared row assembly
# ---------------------------------------------------------------------------

# Canonical arm dict keys, matching the editor's arm-level table columns.
_ARM_BINARY = ("events", "n")
_ARM_CONTINUOUS = ("mean", "sd", "n")


def _arm(treatment: str, **fields: Any) -> dict[str, Any]:
    arm: dict[str, Any] = {"treatment": treatment}
    for key in ("n", "events", "mean", "sd"):
        value = fields.get(key)
        if value is not None:
            arm[key] = float(value)
    return arm


def studies_to_document_rows(studies: list[dict[str, Any]]) -> dict[str, str]:
    """Summarise parsed studies for the UI banner: outcome kinds present.

    Returns {"outcome": "binary"|"continuous"|"mixed"} so the caller can
    pre-set the NMA spec's outcome/measure sensibly (a pure-binary import
    shouldn't land in a continuous-outcome model).
    """

    kinds = set()
    for study in studies:
        for arm in study.get("arms", []):
            if "events" in arm:
                kinds.add("binary")
            elif "mean" in arm:
                kinds.add("continuous")
    if kinds == {"binary"}:
        return {"outcome": "binary"}
    if kinds == {"continuous"}:
        return {"outcome": "continuous"}
    return {"outcome": "mixed"}


# ---------------------------------------------------------------------------
# RevMan .rm5 (XML)
# ---------------------------------------------------------------------------


def parse_revman_rm5(content: bytes) -> list[dict[str, Any]]:
    """Parse a RevMan 5 ``.rm5`` XML file into arm-level study dicts.

    A .rm5 file stores each analysis's data under DICH_DATA /
    CONT_DATA elements, one ``<..._DATA>`` row per study with per-arm
    fields (EVENTS_1 / N_1 / EVENTS_2 / N_2 for dichotomous, MEAN_1 /
    SD_1 / N_1 ... for continuous). Multi-arm trials appear as multiple
    pairwise rows sharing a STUDY_ID; we regroup by study id and emit one
    arm per distinct treatment label, so a 3-arm trial becomes one study
    with three arms instead of two pseudo-2-arm studies.
    """

    try:
        root = ET.fromstring(content.decode("utf-8-sig", errors="replace"))
    except ET.ParseError as exc:
        raise NMAImportError(f"Not a valid .rm5 XML file: {exc}") from None

    # Collect per-study arm observations across every analysis in the file.
    # study_id -> {treatment_label -> arm-fields}
    by_study: dict[str, dict[str, dict[str, float]]] = {}
    order: list[str] = []  # keep the file's study order for a stable table

    def absorb(study_id: str, treatment: str, fields: dict[str, float]) -> None:
        if study_id not in by_study:
            by_study[study_id] = {}
            order.append(study_id)
        arms = by_study[study_id]
        if treatment not in arms:
            arms[treatment] = {}
        arms[treatment].update(fields)

    # Dichotomous data rows.
    for node in root.iter():
        tag = node.tag.upper()
        if not (tag.endswith("DICH_DATA") or tag.endswith("CONT_DATA")):
            continue
        study_id = (node.get("STUDY_ID") or node.get("NAME") or "").strip()
        if not study_id:
            continue
        dich = tag.endswith("DICH_DATA")
        # Arm 1 / arm 2 labels: RevMan stores the group labels on the
        # comparison; fall back to "Experimental"/"Control" when absent.
        label_1 = (node.get("GROUP_LABEL_1") or node.get("GROUP_1") or "Experimental").strip()
        label_2 = (node.get("GROUP_LABEL_2") or node.get("GROUP_2") or "Control").strip()
        try:
            if dich:
                absorb(study_id, label_1, {
                    "events": float(node.get("EVENTS_1", 0) or 0),
                    "n": float(node.get("N_1", 0) or 0),
                })
                absorb(study_id, label_2, {
                    "events": float(node.get("EVENTS_2", 0) or 0),
                    "n": float(node.get("N_2", 0) or 0),
                })
            else:
                absorb(study_id, label_1, {
                    "mean": float(node.get("MEAN_1", 0) or 0),
                    "sd": float(node.get("SD_1", 0) or 0),
                    "n": float(node.get("N_1", 0) or 0),
                })
                absorb(study_id, label_2, {
                    "mean": float(node.get("MEAN_2", 0) or 0),
                    "sd": float(node.get("SD_2", 0) or 0),
                    "n": float(node.get("N_2", 0) or 0),
                })
        except (TypeError, ValueError) as exc:
            raise NMAImportError(
                f"Study '{study_id}': a numeric field couldn't be parsed ({exc})"
            ) from None

    if not by_study:
        raise NMAImportError("No DICH_DATA / CONT_DATA rows found in this .rm5 file")

    studies: list[dict[str, Any]] = []
    for study_id in order:
        arms = [
            _arm(treatment, **fields)
            for treatment, fields in by_study[study_id].items()
        ]
        if len(arms) >= 2:
            studies.append({"id": study_id, "arms": arms, "source": "revman_rm5"})
    if not studies:
        raise NMAImportError("No study with at least two arms found in this .rm5 file")
    return studies


# ---------------------------------------------------------------------------
# RevMan CSV export
# ---------------------------------------------------------------------------

# RevMan's "Export data and analyses -> CSV" uses these column families.
_REV_BINARY_COLS = {
    "events_1": ("experimental events", "events 1", "events_1"),
    "n_1": ("experimental n", "total 1", "n_1"),
    "events_2": ("control events", "events 2", "events_2"),
    "n_2": ("control n", "total 2", "n_2"),
}
_REV_CONT_COLS = {
    "mean_1": ("experimental mean", "mean 1", "mean_1"),
    "sd_1": ("experimental sd", "sd 1", "sd_1"),
    "n_1": ("experimental n", "total 1", "n_1"),
    "mean_2": ("control mean", "mean 2", "mean_2"),
    "sd_2": ("control sd", "sd 2", "sd_2"),
    "n_2": ("control n", "total 2", "n_2"),
}


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text).strip().lower()).strip()


def _find_column(header: list[str], candidates: tuple[str, ...]) -> int | None:
    normed = [_norm(h) for h in header]
    for candidate in candidates:
        target = _norm(candidate)
        for index, name in enumerate(normed):
            if name == target or target in name:
                return index
    return None


def _to_float(raw: str) -> float | None:
    raw = str(raw).strip().replace(",", "")
    if not raw or raw.lower() in {"na", "n/a", "-", "."}:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def parse_revman_csv(content: bytes) -> list[dict[str, Any]]:
    """Parse RevMan's CSV "data and analyses" export into study dicts.

    One row per study per comparison (a multi-arm trial repeats across
    rows); rows are regrouped by the study column into one study with one
    arm per distinct group label, same as parse_revman_rm5.
    """

    reader = csv.reader(io.StringIO(content.decode("utf-8-sig", errors="replace")))
    rows = [row for row in reader if any(cell.strip() for cell in row)]
    if len(rows) < 2:
        raise NMAImportError("RevMan CSV: no data rows found")
    header = rows[0]

    study_col = _find_column(header, ("study", "study id", "study name"))
    if study_col is None:
        raise NMAImportError("RevMan CSV: no 'Study' column found")

    binary = {key: _find_column(header, cols) for key, cols in _REV_BINARY_COLS.items()}
    continuous = {key: _find_column(header, cols) for key, cols in _REV_CONT_COLS.items()}
    is_binary = all(v is not None for v in binary.values())
    is_continuous = all(v is not None for v in continuous.values())
    if not (is_binary or is_continuous):
        raise NMAImportError(
            "RevMan CSV: couldn't find the dichotomous (events/n) or "
            "continuous (mean/SD/n) column families"
        )

    by_study: dict[str, dict[str, dict[str, float]]] = {}
    order: list[str] = []

    for row in rows[1:]:
        study_id = row[study_col].strip()
        if not study_id:
            continue
        if study_id not in by_study:
            by_study[study_id] = {}
            order.append(study_id)
        arms = by_study[study_id]
        for label, keys in (("Experimental", ("1",)), ("Control", ("2",))):
            fields: dict[str, float] = {}
            suffix = f"_{keys[0]}"
            if is_binary:
                e = _to_float(row[binary[f"events{suffix}"]])
                n = _to_float(row[binary[f"n{suffix}"]])
                if e is not None:
                    fields["events"] = e
                if n is not None:
                    fields["n"] = n
            else:
                m = _to_float(row[continuous[f"mean{suffix}"]])
                s = _to_float(row[continuous[f"sd{suffix}"]])
                n = _to_float(row[continuous[f"n{suffix}"]])
                if m is not None:
                    fields["mean"] = m
                if s is not None:
                    fields["sd"] = s
                if n is not None:
                    fields["n"] = n
            if fields:
                arms.setdefault(label, {}).update(fields)

    studies = []
    for study_id in order:
        arms = [_arm(t, **f) for t, f in by_study[study_id].items() if f]
        if len(arms) >= 2:
            studies.append({"id": study_id, "arms": arms, "source": "revman_csv"})
    if not studies:
        raise NMAImportError("RevMan CSV: no study with two usable arms found")
    return studies


# ---------------------------------------------------------------------------
# Covidence CSV extraction export
# ---------------------------------------------------------------------------

# Covidence extraction CSVs are one row per study x arm, with an outcome
# name column and per-arm numeric columns. Column names vary slightly
# across template versions, so each is matched case/space-insensitively.
_COV_STUDY = ("study", "study id", "title", "reference")
_COV_ARM = ("intervention", "arm", "treatment", "group")
_COV_OUTCOME = ("outcome", "outcome name")
_COV_BINARY = {"events": ("events", "n events", "number of events"), "n": ("n", "total", "sample size", "n randomised")}
_COV_CONT = {"mean": ("mean",), "sd": ("sd", "standard deviation"), "n": ("n", "total", "sample size")}


def parse_covidence_csv(content: bytes) -> list[dict[str, Any]]:
    """Parse a Covidence data-extraction CSV into arm-level study dicts.

    Covidence exports one row per study x intervention arm. Each row is
    already one arm, so no pairwise regrouping is needed -- just group the
    rows by study and outcome. Studies with a single arm are dropped (an
    NMA contrast needs at least two).
    """

    reader = csv.reader(io.StringIO(content.decode("utf-8-sig", errors="replace")))
    rows = [row for row in reader if any(cell.strip() for cell in row)]
    if len(rows) < 2:
        raise NMAImportError("Covidence CSV: no data rows found")
    header = rows[0]

    study_col = _find_column(header, _COV_STUDY)
    arm_col = _find_column(header, _COV_ARM)
    if study_col is None or arm_col is None:
        raise NMAImportError("Covidence CSV: need a Study and an Intervention/Arm column")

    binary = {key: _find_column(header, cols) for key, cols in _COV_BINARY.items()}
    continuous = {key: _find_column(header, cols) for key, cols in _COV_CONT.items()}
    is_binary = all(v is not None for v in binary.values())
    is_continuous = all(v is not None for v in continuous.values())
    if not (is_binary or is_continuous):
        raise NMAImportError(
            "Covidence CSV: couldn't find events/n or mean/SD/n columns"
        )

    by_study: dict[str, dict[str, dict[str, float]]] = {}
    order: list[str] = []
    for row in rows[1:]:
        study_id = row[study_col].strip()
        arm_label = row[arm_col].strip()
        if not study_id or not arm_label:
            continue
        if study_id not in by_study:
            by_study[study_id] = {}
            order.append(study_id)
        fields: dict[str, float] = {}
        if is_binary:
            e = _to_float(row[binary["events"]])
            n = _to_float(row[binary["n"]])
            if e is not None:
                fields["events"] = e
            if n is not None:
                fields["n"] = n
        else:
            for key in ("mean", "sd", "n"):
                value = _to_float(row[continuous[key]])
                if value is not None:
                    fields[key] = value
        if fields:
            by_study[study_id].setdefault(arm_label, {}).update(fields)

    studies = []
    for study_id in order:
        arms = [_arm(t, **f) for t, f in by_study[study_id].items() if f]
        if len(arms) >= 2:
            studies.append({"id": study_id, "arms": arms, "source": "covidence_csv"})
    if not studies:
        raise NMAImportError("Covidence CSV: no study with two usable arms found")
    return studies
