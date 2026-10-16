"""Streamlit-free helpers for the NMA Editor screen (nma_editor.py).

Kept separate from the UI so they can be unit-tested without Streamlit:
document <-> editable tables, the network-plot payload, and result tables.
The NMA document is the same JSON shape heor_tree_engine.nma_parser reads.
"""

from __future__ import annotations

import json
import os
import math
from typing import Any

import pandas as pd

from heor_tree_engine.nma_models import NMAFit, NetworkMetaAnalysis, is_ratio_measure
from heor_tree_engine.nma_network import NetworkSummary

TREATMENT_COLUMNS = ["name", "aliases", "class", "color"]
ARM_COLUMNS = ["study", "treatment", "events", "n", "mean", "sd", "exposure", "source"]
CONTRAST_COLUMNS = ["study", "treatment", "versus", "estimate", "se", "effect", "lower", "upper", "baseline_se", "source"]
NUMERIC_ARM_COLUMNS = ["events", "n", "mean", "sd", "exposure"]
NUMERIC_CONTRAST_COLUMNS = ["estimate", "se", "effect", "lower", "upper", "baseline_se"]
TREATMENT_LABELS = {"name": "Name", "aliases": "Aliases", "class": "Class", "color": "Colour"}
ARM_LABELS = {"study": "Study", "treatment": "Treatment", "events": "Events", "n": "Total n", "mean": "Mean",
              "sd": "SD", "exposure": "Person-time", "source": "Source (DOI, table)"}
CONTRAST_LABELS = {"study": "Study", "treatment": "Treatment", "versus": "Versus", "estimate": "Estimate",
                   "se": "SE", "effect": "Effect", "lower": "Lower", "upper": "Upper",
                   "baseline_se": "Baseline SE", "source": "Source"}


def _spec(keys: list[str], labels: dict[str, str], numeric: list[str], widths: dict[str, int]) -> list[dict]:
    return [{"key": key, "label": labels[key], "type": "number" if key in numeric else "text",
             **({"width": widths[key]} if key in widths else {})} for key in keys]


TREATMENT_COLUMN_SPEC = _spec(["name", "aliases", "class", "color"], TREATMENT_LABELS, [],
                              {"name": 120, "aliases": 110, "class": 100, "color": 90})
ARM_COLUMN_SPEC = _spec(ARM_COLUMNS, ARM_LABELS, NUMERIC_ARM_COLUMNS,
                        {"study": 105, "treatment": 120, "events": 80, "n": 80, "mean": 90, "sd": 80,
                         "exposure": 110, "source": 150})
CONTRAST_COLUMN_SPEC = _spec(CONTRAST_COLUMNS, CONTRAST_LABELS, NUMERIC_CONTRAST_COLUMNS,
                             {"study": 105, "treatment": 120, "versus": 120, "source": 150})

# Arm-level fields each outcome type actually uses (the editor hides the rest).
ARM_FIELDS_BY_OUTCOME = {
    "binary": ["events", "n"],
    "continuous": ["mean", "sd", "n"],
    "rate": ["events", "exposure"],
}

# Colour-blind-safe class palette (blue / orange / teal / purple / ochre / pink),
# each dark enough for a white node outline and legible label beside it.
CLASS_PALETTE = ["#2f6fd6", "#ea580c", "#0f766e", "#7c3aed", "#a16207", "#be185d"]
UNCLASSIFIED_COLOR = "#64748b"
SINGLE_COLOR = "#2f6fd6"
CONTROL_CLASS_NAMES = {"control", "placebo", "reference", "standard of care", "soc", "usual care"}


def default_document() -> dict[str, Any]:
    """Illustrative starter network (made-up numbers) so the screen is not empty."""

    def arms(*rows):
        return [{"treatment": t, "events": r, "n": n} for t, r, n in rows]

    return {
        "id": "nma",
        "name": "Example network (illustrative data)",
        "spec": {"outcome": "binary", "measure": "OR", "reference": "Placebo", "small_values": "good",
                 "zero_correction": 0.5, "ci_level": 0.95, "effects": "random"},
        "treatments": [
            {"name": "Placebo", "aliases": ["PBO"], "class": "Control", "color": "#64748b"},
            {"name": "Drug A", "class": "Class X", "color": "#2f6fd6"},
            {"name": "Drug B", "class": "Class X", "color": "#0891b2"},
            {"name": "Drug C", "class": "Class Y", "color": "#ea580c"},
            {"name": "Drug D", "class": "Class Y", "color": "#a16207"},
        ],
        "studies": [
            {"id": "Study 01", "arms": arms(("Placebo", 18, 120), ("Drug A", 11, 118))},
            {"id": "Study 02", "arms": arms(("Placebo", 25, 200), ("Drug A", 14, 196))},
            {"id": "Study 03", "arms": arms(("Placebo", 15, 100), ("Drug A", 10, 98))},
            {"id": "Study 04", "arms": arms(("Placebo", 30, 150), ("Drug B", 20, 148))},
            {"id": "Study 05", "arms": arms(("Placebo", 22, 130), ("Drug B", 18, 131))},
            {"id": "Study 06", "arms": arms(("Placebo", 16, 90), ("Drug C", 13, 92))},
            {"id": "Study 07", "arms": arms(("Drug A", 12, 90), ("Drug B", 15, 92), ("Drug C", 17, 91))},
            {"id": "Study 08", "arms": arms(("Drug B", 9, 80), ("Drug D", 7, 82))},
            {"id": "Study 09", "arms": arms(("Placebo", 20, 110), ("Drug D", 12, 108))},
        ],
    }


def is_example_data(document: dict[str, Any]) -> bool:
    """True while the study data are still the built-in illustrative example
    (so the 'illustrative data' note disappears once real data are entered)."""

    example = default_document()
    normalised = studies_from_frames(arms_frame(example), contrasts_frame(example))
    return document.get("studies") in (example["studies"], normalised)


# --------------------------------------------------------------------------
# Document <-> tables
# --------------------------------------------------------------------------


def _blank(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    return isinstance(value, str) and value.strip() == ""


def _clean(value: Any) -> Any:
    return None if _blank(value) else value


def _number(value: Any) -> float | None:
    if _blank(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def treatments_frame(document: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for raw in document.get("treatments", []):
        if isinstance(raw, str):
            raw = {"name": raw}
        rows.append({
            "name": raw.get("name", ""),
            "aliases": ", ".join(raw.get("aliases", [])),
            "class": raw.get("class") or "",
            "color": raw.get("color") or "",
        })
    return pd.DataFrame(rows, columns=TREATMENT_COLUMNS).astype(object)


def arms_frame(document: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for study in document.get("studies", []):
        for arm in study.get("arms", []):
            rows.append({"study": study.get("id", ""), **{key: arm.get(key) for key in ARM_COLUMNS if key != "study"}})
    frame = pd.DataFrame(rows, columns=ARM_COLUMNS)
    for column in NUMERIC_ARM_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("float64")
    for column in ("study", "treatment", "source"):
        frame[column] = frame[column].map(lambda value: "" if _blank(value) else value).astype(object)
    return frame


def contrasts_frame(document: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for study in document.get("studies", []):
        for contrast in study.get("contrasts", []):
            row = {"study": study.get("id", ""), **{key: contrast.get(key) for key in CONTRAST_COLUMNS if key != "study"}}
            row["baseline_se"] = study.get("baseline_se")
            rows.append(row)
    frame = pd.DataFrame(rows, columns=CONTRAST_COLUMNS)
    for column in NUMERIC_CONTRAST_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("float64")
    for column in ("study", "treatment", "versus", "source"):
        frame[column] = frame[column].map(lambda value: "" if _blank(value) else value).astype(object)
    return frame


def treatments_from_frame(frame: pd.DataFrame) -> list[dict[str, Any]]:
    treatments = []
    for row in frame.to_dict("records"):
        name = _clean(row.get("name"))
        if name is None:
            continue
        item: dict[str, Any] = {"name": str(name).strip()}
        aliases = _clean(row.get("aliases"))
        if aliases:
            item["aliases"] = [alias.strip() for alias in str(aliases).split(",") if alias.strip()]
        if _clean(row.get("class")):
            item["class"] = str(row["class"]).strip()
        if _clean(row.get("color")):
            item["color"] = str(row["color"]).strip()
        treatments.append(item)
    return treatments


def studies_from_frames(arms: pd.DataFrame, contrasts: pd.DataFrame) -> list[dict[str, Any]]:
    """Rebuild the document's studies (in first-appearance order). Rows with
    no study id or no treatment are ignored (half-typed new rows)."""

    studies: dict[str, dict[str, Any]] = {}
    for row in arms.to_dict("records"):
        study_id, treatment = _clean(row.get("study")), _clean(row.get("treatment"))
        if study_id is None or treatment is None:
            continue
        arm: dict[str, Any] = {"treatment": str(treatment).strip()}
        for key in NUMERIC_ARM_COLUMNS:
            value = _number(row.get(key))
            if value is not None:
                arm[key] = value
        if _clean(row.get("source")):
            arm["source"] = str(row["source"])
        studies.setdefault(str(study_id).strip(), {"id": str(study_id).strip()}).setdefault("arms", []).append(arm)

    for row in contrasts.to_dict("records"):
        study_id, treatment, versus = _clean(row.get("study")), _clean(row.get("treatment")), _clean(row.get("versus"))
        if study_id is None or treatment is None or versus is None:
            continue
        contrast: dict[str, Any] = {"treatment": str(treatment).strip(), "versus": str(versus).strip()}
        for key in ("estimate", "se", "effect", "lower", "upper"):
            value = _number(row.get(key))
            if value is not None:
                contrast[key] = value
        if _clean(row.get("source")):
            contrast["source"] = str(row["source"])
        study = studies.setdefault(str(study_id).strip(), {"id": str(study_id).strip()})
        study.setdefault("contrasts", []).append(contrast)
        baseline_se = _number(row.get("baseline_se"))
        if baseline_se is not None:
            study["baseline_se"] = baseline_se
    return list(studies.values())


def frames_from_csv(frame: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    """Classify an imported CSV as arm-level or contrast-level and return
    ("arms" | "contrasts", frame with the editor's columns)."""

    columns = {str(column).strip().lower(): column for column in frame.columns}
    renamed = frame.rename(columns={original: lower for lower, original in columns.items()})
    if "comparator" in renamed.columns and "versus" not in renamed.columns:
        renamed = renamed.rename(columns={"comparator": "versus"})
    if "study" not in renamed.columns or "treatment" not in renamed.columns:
        raise ValueError("The CSV needs at least 'study' and 'treatment' columns")
    if "versus" in renamed.columns:
        kind, target, numeric = "contrasts", CONTRAST_COLUMNS, NUMERIC_CONTRAST_COLUMNS
    else:
        kind, target, numeric = "arms", ARM_COLUMNS, NUMERIC_ARM_COLUMNS
    out = pd.DataFrame({column: renamed[column] if column in renamed.columns else None for column in target})
    for column in numeric:
        out[column] = pd.to_numeric(out[column], errors="coerce").astype("float64")
    for column in [c for c in target if c not in numeric]:
        out[column] = out[column].map(lambda value: "" if _blank(value) else value).astype(object)
    return kind, out


def document_hash(document: dict[str, Any]) -> str:
    """Fingerprint of everything that changes the fitted model (the displayed
    fixed/random choice is excluded: both fits are computed on every run;
    study characteristics, CINeMA judgements and titles don't enter the fit)."""

    spec = {key: value for key, value in document.get("spec", {}).items()
            if key not in ("effects", "engine", "bayes", "split_method")}
    fitted = {key: value for key, value in document.items()
              if key not in ("study_characteristics", "characteristic_columns", "cinema", "absolute", "prisma", "sof", "title", "name", "analysis_id")}
    return json.dumps({**fitted, "spec": spec}, sort_keys=True, ensure_ascii=False, default=str)


# --------------------------------------------------------------------------
# Network plot
# --------------------------------------------------------------------------


def node_colors(model: NetworkMetaAnalysis, mode: str) -> dict[str, str]:
    classes: list[str] = []
    for treatment in model.treatments:
        if treatment.treatment_class and treatment.treatment_class not in classes:
            classes.append(treatment.treatment_class)
    # A control / placebo / reference class is grey; other classes take the
    # palette in order of appearance.
    class_color: dict[str, str] = {}
    palette_index = 0
    for name in classes:
        if name.strip().casefold() in CONTROL_CLASS_NAMES:
            class_color[name] = UNCLASSIFIED_COLOR
        else:
            class_color[name] = CLASS_PALETTE[palette_index % len(CLASS_PALETTE)]
            palette_index += 1
    colors = {}
    for treatment in model.treatments:
        by_class = class_color.get(treatment.treatment_class or "", UNCLASSIFIED_COLOR)
        if mode == "single":
            colors[treatment.name] = SINGLE_COLOR
        elif mode == "custom":
            colors[treatment.name] = treatment.color or by_class
        else:
            colors[treatment.name] = by_class
    return colors


def class_legend(model: NetworkMetaAnalysis) -> list[tuple[str, str, list[str]]]:
    """(class label, colour, members) in first-appearance order."""

    colors = node_colors(model, "class")
    legend: dict[str, tuple[str, list[str]]] = {}
    for treatment in model.treatments:
        label = treatment.treatment_class or "No class"
        legend.setdefault(label, (colors[treatment.name], []))[1].append(treatment.name)
    return [(label, color, members) for label, (color, members) in legend.items()]


def network_payload(model: NetworkMetaAnalysis, summary: NetworkSummary, color_mode: str,
                    size_by: str, selected_edge: tuple[str, str] | None) -> dict[str, Any]:
    colors = node_colors(model, color_mode)
    if size_by == "patients":
        weights = summary.patients
    elif size_by == "studies":
        weights = {name: float(count) for name, count in summary.n_studies.items()}
    else:
        weights = {name: 1.0 for name in summary.treatments}
    top = max([w for w in weights.values() if w > 0], default=1.0)
    nodes = []
    for name in summary.treatments:
        weight = weights.get(name, 0.0)
        radius = 10.0 + 18.0 * math.sqrt(weight / top) if weight > 0 else 8.0
        nodes.append({
            "id": name,
            "color": colors.get(name, UNCLASSIFIED_COLOR),
            "r": round(radius, 1),
            "patients": summary.patients.get(name, 0.0),
            "studies": summary.n_studies.get(name, 0),
        })
    max_studies = max([edge.n_studies for edge in summary.edges.values()], default=1)
    edges = []
    for key, edge in summary.edges.items():
        edges.append({
            "a": key[0],
            "b": key[1],
            "n": edge.n_studies,
            "w": round(2.0 + 7.0 * (edge.n_studies - 1) / max(max_studies - 1, 1), 1),
            "patients": edge.patients,
            "selected": selected_edge is not None and tuple(sorted(selected_edge)) == key,
        })
    return {"nodes": nodes, "edges": edges}


def indirect_paths(summary: NetworkSummary, a: str, b: str) -> list[str]:
    """Treatments C with both A-C and C-B comparisons (one-step indirect routes)."""

    neighbours_a = set(summary.neighbours(a))
    return [c for c in summary.neighbours(b) if c in neighbours_a and c not in (a, b)]


# --------------------------------------------------------------------------
# Results tables
# --------------------------------------------------------------------------


def format_effect(value: float, lower: float, upper: float, digits: int = 2) -> str:
    return f"{value:.{digits}f} ({lower:.{digits}f} to {upper:.{digits}f})"


def ranking_order(fit: NMAFit) -> list[str]:
    return sorted(fit.p_scores, key=lambda name: -fit.p_scores[name])


def league_frame(fit: NMAFit, order: list[str], digits: int = 2) -> pd.DataFrame:
    """Lower triangle: cell (row i, column j) = effect of column j vs row i."""

    table = pd.DataFrame("", index=order, columns=order, dtype=object)
    for i, row_name in enumerate(order):
        table.loc[row_name, row_name] = row_name
        for column_name in order[:i]:
            effect = fit.league[(column_name, row_name)]
            table.loc[row_name, column_name] = format_effect(effect.natural, effect.natural_lower, effect.natural_upper, digits)
    return table


def effects_frame(fit: NMAFit, measure: str) -> pd.DataFrame:
    rows = [{
        "treatment": effect.treatment,
        "versus": effect.versus,
        "estimate": effect.natural,
        "lower": effect.natural_lower,
        "upper": effect.natural_upper,
        "log_scale_estimate" if is_ratio_measure(measure) else "estimate_raw": effect.estimate,
        "se": effect.se,
        "p_score": fit.p_scores.get(effect.treatment),
    } for effect in fit.effects]
    return pd.DataFrame(rows)


def significant(effect_lower: float, effect_upper: float, measure: str) -> bool:
    null = 1.0 if is_ratio_measure(measure) else 0.0
    return effect_lower > null or effect_upper < null


def node_split_frame(splits, measure: str, digits: int = 2) -> pd.DataFrame:
    """Node-split table; `_flag` marks p < 0.05 (the UI shades those rows)."""

    ratio = is_ratio_measure(measure)
    rows = []
    for split in splits:
        row = {
            "Comparison": f"{split.treatment} vs {split.versus}",
            "Direct studies": split.n_direct_studies,
            f"Direct {measure}": format_effect(split.natural_direct, split.direct_lower, split.direct_upper, digits),
            f"Indirect {measure}": format_effect(split.natural_indirect, split.indirect_lower, split.indirect_upper, digits),
            f"Network {measure}": f"{split.natural_network:.{digits}f}",
        }
        if ratio:
            # indirect / direct on the natural scale (ratio of odds ratios, etc.)
            row["Ratio (indirect / direct)"] = f"{math.exp(-split.difference):.{digits}f}"
        else:
            row["Difference (direct - indirect)"] = f"{split.difference:.{digits}f}"
        row["p"] = "<0.001" if split.p_value < 0.001 else f"{split.p_value:.3f}"
        row["_flag"] = split.p_value < 0.05
        rows.append(row)
    return pd.DataFrame(rows)


TABLE_HEADER_BG = "#1e3a8a"
TABLE_HEADER_FG = "#ffffff"
TABLE_FLAG_BG = "#ffedd5"


def html_table(frame: pd.DataFrame, flags: list[bool] | None = None, index: bool = False,
               max_height: int | None = None) -> str:
    """Read-only table with a dark-blue header and white header text.

    st.dataframe draws its header on a <canvas> in the browser, so its
    colours can't be changed per table - read-only NMA tables are rendered
    as plain HTML instead. `flags` shades those rows orange (e.g. p < 0.05).
    Numbers are right-aligned; everything is HTML-escaped.
    """

    import html as html_lib

    def cell(value: Any) -> str:
        if _blank(value):
            return ""
        if isinstance(value, float):
            return str(int(value)) if value.is_integer() else f"{value:.6g}"
        return html_lib.escape(str(value))

    def is_number(value: Any) -> bool:
        return isinstance(value, (int, float)) and not isinstance(value, bool)

    columns = ([frame.index.name or ""] if index else []) + [str(column) for column in frame.columns]
    head = "".join(
        f"<th style='background:{TABLE_HEADER_BG};color:{TABLE_HEADER_FG};font-weight:600;"
        f"padding:8px 12px;text-align:left;white-space:nowrap;position:sticky;top:0'>{html_lib.escape(name)}</th>"
        for name in columns
    )
    body_rows = []
    for position, (label, row) in enumerate(frame.iterrows()):
        flagged = bool(flags[position]) if flags is not None and position < len(flags) else False
        background = TABLE_FLAG_BG if flagged else ("#f5f8fe" if position % 2 else "#ffffff")
        cells = []
        if index:
            cells.append(f"<td style='padding:7px 12px;font-weight:600;white-space:nowrap'>{cell(label)}</td>")
        for value in row.tolist():
            align = "right" if is_number(value) else "left"
            cells.append(f"<td style='padding:7px 12px;text-align:{align};white-space:nowrap'>{cell(value)}</td>")
        body_rows.append(f"<tr style='background:{background};border-top:1px solid #e5eaf2'>{''.join(cells)}</tr>")
    height = f"max-height:{max_height}px;overflow-y:auto;" if max_height else ""
    return (
        f"<div style='{height}overflow-x:auto;border:1px solid #dbe2ee;border-radius:8px;margin-bottom:8px'>"
        f"<table style='width:100%;border-collapse:collapse;font-size:14px;color:#0f172a'>"
        f"<thead><tr>{head}</tr></thead><tbody>{''.join(body_rows)}</tbody></table></div>"
    )



def frame_to_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """JSON-safe rows for the table component (NaN / None -> None)."""

    rows = []
    for record in frame.to_dict("records"):
        rows.append({str(key): (None if _blank(value) else (value.item() if hasattr(value, "item") else value))
                     for key, value in record.items()})
    return rows


def rows_to_frame(rows: list[dict[str, Any]], columns: list[str], numeric: list[str]) -> pd.DataFrame:
    """Inverse of frame_to_rows, with the editor's column order and dtypes."""

    frame = pd.DataFrame([{column: row.get(column) for column in columns} for row in rows], columns=columns)
    for column in columns:
        if column in numeric:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("float64")
        else:
            frame[column] = frame[column].map(lambda value: "" if _blank(value) else value).astype(object)
    return frame


def bayes_hash(document: dict[str, Any]) -> str:
    """Fingerprint for Bayesian results: the data plus the Bayesian settings."""

    return document_hash(document) + json.dumps(document.get("spec", {}).get("bayes", {}), sort_keys=True)


def rank_probability_frame(rank_probabilities: dict[str, list[float]], order: list[str]) -> pd.DataFrame:
    """Long format for a stacked rankogram: treatment, rank (1 = best), probability."""

    rows = []
    for name in order:
        for rank, probability in enumerate(rank_probabilities[name], start=1):
            rows.append({"treatment": name, "rank": rank, "probability": probability})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Study characteristics (transitivity) and risk of bias
# --------------------------------------------------------------------------

# Study characteristics: fixed columns (study, risk of bias, indirectness,
# notes) around a user-defined set of variables. The defaults below are used
# until the file stores its own list in document["characteristic_columns"] =
# [{"key", "label", "type": "number" | "text"}, ...].
DEFAULT_CHARACTERISTIC_VARIABLES: list[dict[str, str]] = [
    {"key": "year", "label": "Year", "type": "number"},
    {"key": "mean_age", "label": "Mean age", "type": "number"},
    {"key": "female_pct", "label": "Female %", "type": "number"},
    {"key": "follow_up", "label": "Follow-up (months)", "type": "number"},
]
FIXED_CHARACTERISTIC_KEYS = ("study", "rob", "indirectness", "notes")
CHARACTERISTIC_LABELS = {"study": "Study", "rob": "Risk of bias", "indirectness": "Indirectness", "notes": "Notes",
                         **{v["key"]: v["label"] for v in DEFAULT_CHARACTERISTIC_VARIABLES}}
_CHARACTERISTIC_WIDTHS = {"study": 105, "year": 70, "mean_age": 90, "female_pct": 90, "follow_up": 110, "rob": 110,
                          "indirectness": 110}


def characteristic_variables(document: dict[str, Any]) -> list[dict[str, str]]:
    """The file's variables (validated copy), or the defaults."""

    stored = document.get("characteristic_columns")
    if not isinstance(stored, list):
        return [dict(v) for v in DEFAULT_CHARACTERISTIC_VARIABLES]
    out, seen = [], set()
    for item in stored:
        if not isinstance(item, dict):
            continue
        key = str(item.get("key") or "").strip()
        if not key or key in FIXED_CHARACTERISTIC_KEYS or key in seen:
            continue
        seen.add(key)
        out.append({"key": key, "label": str(item.get("label") or key),
                    "type": "number" if item.get("type") == "number" else "text"})
    return out


def variable_key(label: str, existing: list[str]) -> str:
    """Stable column key from a label ('Mean BMI' -> 'mean_bmi'), unique among `existing`."""

    base = slug(label).strip("_") or "var"
    key, n = base, 2
    while key in existing or key in FIXED_CHARACTERISTIC_KEYS:
        key, n = f"{base}_{n}", n + 1
    return key


def characteristic_columns(document: dict[str, Any]) -> list[str]:
    return ["study", *(v["key"] for v in characteristic_variables(document)), "rob", "indirectness", "notes"]


def numeric_characteristic_columns(document: dict[str, Any]) -> list[str]:
    return [v["key"] for v in characteristic_variables(document) if v["type"] == "number"]


def characteristic_labels(document: dict[str, Any]) -> dict[str, str]:
    return {**CHARACTERISTIC_LABELS, **{v["key"]: v["label"] for v in characteristic_variables(document)}}


def characteristic_column_spec(document: dict[str, Any]) -> list[dict]:
    labels = characteristic_labels(document)
    return [
        {**column, **({"placeholder": "Low / Some / High"} if column["key"] in ("rob", "indirectness") else {})}
        for column in _spec(characteristic_columns(document), labels, numeric_characteristic_columns(document),
                            _CHARACTERISTIC_WIDTHS)
    ]


# Backwards-compatible module constants (the defaults).
CHARACTERISTIC_COLUMNS = characteristic_columns({})
NUMERIC_CHARACTERISTIC_COLUMNS = numeric_characteristic_columns({})
CHARACTERISTIC_COLUMN_SPEC = characteristic_column_spec({})
ROB_LEVELS = ("Low", "Some concerns", "High")


def normalize_rob(value: Any) -> str:
    """'high', 'H', 'high risk' -> 'High'; 'some', 'unclear', 'moderate' -> 'Some concerns'."""

    text = "" if _blank(value) else str(value).strip().lower()
    if not text:
        return ""
    if text.startswith("h"):
        return "High"
    if text.startswith("l"):
        return "Low"
    if text.startswith(("s", "u", "m")):
        return "Some concerns"
    return str(value).strip()


def characteristics_frame(document: dict[str, Any]) -> pd.DataFrame:
    """One row per study: stored rows first, then any study without a row."""

    stored = {str(row.get("study")): row for row in document.get("study_characteristics", []) if row.get("study")}
    ids = [str(study.get("id")) for study in document.get("studies", [])]
    rows = [stored.get(study_id, {"study": study_id}) for study_id in ids]
    rows += [row for study_id, row in stored.items() if study_id not in ids]
    return rows_to_frame(rows, characteristic_columns(document), numeric_characteristic_columns(document))


def characteristics_from_frame(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for row in frame_to_rows(frame):
        if not row.get("study"):
            continue
        clean = {key: value for key, value in row.items() if value not in (None, "")}
        for key in ("rob", "indirectness"):
            if key in clean:
                clean[key] = normalize_rob(clean[key])
        rows.append(clean)
    return rows


def pooled_sd(document: dict[str, Any]) -> float | None:
    """SD pooled over all arms with mean/sd/n (weights n - 1), for rescaling
    SMD-scale priors to the MD scale."""

    num = den = 0.0
    for study in document.get("studies", []):
        for arm in study.get("arms", []) or []:
            sd, n = arm.get("sd"), arm.get("n")
            if isinstance(sd, (int, float)) and isinstance(n, (int, float)) and n > 1 and sd > 0:
                num += (n - 1) * sd * sd
                den += n - 1
    return math.sqrt(num / den) if den > 0 else None


def high_rob_studies(rows: list[dict[str, Any]]) -> list[str]:
    return [str(row["study"]) for row in rows if normalize_rob(row.get("rob")) == "High"]


def _fmt_range(values: list[float], digits: int = 0) -> str:
    if not values:
        return ""
    low, high = min(values), max(values)
    mean = sum(values) / len(values)
    if low == high:
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ({low:.{digits}f}–{high:.{digits}f})"


def transitivity_frame(summary: NetworkSummary, characteristic_rows: list[dict[str, Any]],
                       variables: list[dict[str, str]] | None = None) -> pd.DataFrame:
    """Per direct comparison: numeric study characteristics summarised as
    mean (min–max) (years as a range), text ones as their distinct values,
    and risk-of-bias counts - to judge whether the comparisons are similar
    enough to be combined (transitivity)."""

    variables = DEFAULT_CHARACTERISTIC_VARIABLES if variables is None else variables
    by_study = {str(row.get("study")): row for row in characteristic_rows}

    def texts(ids: list[str], key: str) -> str:
        seen: list[str] = []
        for study_id in ids:
            value = by_study.get(study_id, {}).get(key)
            text = "" if value is None else str(value).strip()
            if text and text not in seen:
                seen.append(text)
        return ", ".join(seen)

    def values(ids: list[str], key: str) -> list[float]:
        out = []
        for study_id in ids:
            value = by_study.get(study_id, {}).get(key)
            if isinstance(value, (int, float)) and not (isinstance(value, float) and math.isnan(value)):
                out.append(float(value))
        return out

    rows = []
    for (a, b), edge in summary.edges.items():
        ids = edge.study_ids
        rob = [normalize_rob(by_study.get(study_id, {}).get("rob")) for study_id in ids]
        rob_text = ", ".join(f"{level}: {rob.count(level)}" for level in ROB_LEVELS if rob.count(level))
        record: dict[str, Any] = {"Comparison": f"{a} vs {b}", "Studies": edge.n_studies}
        for variable in variables:
            key, label = variable["key"], variable["label"]
            if variable["type"] != "number":
                record[label] = texts(ids, key)
            elif key == "year":
                years = values(ids, key)
                record[label] = "" if not years else (f"{min(years):.0f}" if min(years) == max(years)
                                                      else f"{min(years):.0f}–{max(years):.0f}")
            else:
                record[label] = _fmt_range(values(ids, key), 1)
        record["Risk of bias"] = rob_text or "not recorded"
        rows.append(record)
    return pd.DataFrame(rows)


def class_effects_frame(bayes: Any, measure: str, interval_label: str = "CrI") -> pd.DataFrame:
    """Class effects of a Bayesian run with treatment classes: members, effect versus the reference
    (median and credible interval on the natural scale) and SUCRA across the classes."""

    rows = []
    for name, effect in getattr(bayes, "class_effects", {}).items():
        sucra = bayes.class_sucra.get(name)
        rows.append({"Class": name, "Treatments": ", ".join(bayes.class_members.get(name, [])),
                     f"{measure} vs {bayes.reference} ({interval_label})":
                         format_effect(effect.natural, effect.natural_lower, effect.natural_upper),
                     "SUCRA": "–" if sucra is None else f"{sucra:.3f}"})
    return pd.DataFrame(rows, columns=["Class", "Treatments", f"{measure} vs {getattr(bayes, 'reference', '')} ({interval_label})",
                                       "SUCRA"]) if rows else pd.DataFrame()


ROB_NOT_RECORDED = "Not recorded"
TRANSITIVITY_POINT_COLUMNS = ["variable", "label", "comparison", "study", "value", "rob"]


def transitivity_points(summary: NetworkSummary, characteristic_rows: list[dict[str, Any]],
                        variables: list[dict[str, str]] | None = None) -> pd.DataFrame:
    """One row per (numeric variable, direct comparison, study) with a recorded
    value - the data behind the transitivity plots (each dot is a study, coloured
    by its risk of bias). A multi-arm study appears in every comparison it
    contributes to. Studies without a value for a variable are left out."""

    variables = DEFAULT_CHARACTERISTIC_VARIABLES if variables is None else variables
    by_study = {str(row.get("study")): row for row in characteristic_rows}
    rows: list[dict[str, Any]] = []
    for (a, b), edge in summary.edges.items():
        for study_id in edge.study_ids:
            record = by_study.get(study_id, {})
            rob = normalize_rob(record.get("rob"))
            rob = rob if rob in ROB_LEVELS else ROB_NOT_RECORDED
            for variable in variables:
                if variable["type"] != "number":
                    continue
                value = record.get(variable["key"])
                if isinstance(value, bool) or not isinstance(value, (int, float)) or math.isnan(float(value)):
                    continue
                rows.append({"variable": variable["key"], "label": variable["label"], "comparison": f"{a} vs {b}",
                             "study": study_id, "value": float(value), "rob": rob})
    return pd.DataFrame(rows, columns=TRANSITIVITY_POINT_COLUMNS)


def transitivity_categories(summary: NetworkSummary, characteristic_rows: list[dict[str, Any]],
                            variables: list[dict[str, str]] | None = None) -> pd.DataFrame:
    """Text variables: number of studies per category in each direct comparison
    (columns: variable, comparison, category, studies)."""

    variables = DEFAULT_CHARACTERISTIC_VARIABLES if variables is None else variables
    by_study = {str(row.get("study")): row for row in characteristic_rows}
    rows: list[dict[str, Any]] = []
    for variable in variables:
        if variable["type"] == "number":
            continue
        for (a, b), edge in summary.edges.items():
            counts: dict[str, int] = {}
            for study_id in edge.study_ids:
                value = by_study.get(study_id, {}).get(variable["key"])
                text = "" if value is None else str(value).strip()
                if text:
                    counts[text] = counts.get(text, 0) + 1
            for category, count in counts.items():
                rows.append({"variable": variable["label"], "comparison": f"{a} vs {b}",
                             "category": category, "studies": count})
    return pd.DataFrame(rows, columns=["variable", "comparison", "category", "studies"])


def figure_legend(model: NetworkMetaAnalysis, color_mode: str) -> list[dict[str, str]]:
    """Legend rows drawn under the exported network PNG."""

    if color_mode == "class":
        return [{"label": label, "color": color, "members": ", ".join(members)}
                for label, color, members in class_legend(model)]
    if color_mode == "custom":
        colors = node_colors(model, "custom")
        return [{"label": name, "color": colors[name], "members": ""} for name in model.treatment_names]
    return []


def figure_filename(title: str, view: str) -> str:
    import re

    slug = re.sub(r"[^0-9a-zA-Z]+", "-", title or "").strip("-").lower()[:60]
    return f"{slug or 'nma'}-network-{view.lower()}.png"


# --------------------------------------------------------------------------
# Several analyses (outcomes) of the same network in one file
# --------------------------------------------------------------------------

FILE_FORMAT = "decide-nma"
SHARED_KEYS = ("name", "treatments", "study_characteristics", "characteristic_columns", "prisma")  # common to every analysis of the file
ANALYSIS_KEYS = ("analysis_id", "title", "spec", "studies", "cinema", "absolute", "sof")


def _new_id() -> str:
    import uuid

    return uuid.uuid4().hex[:10]


def analysis_title(document: dict[str, Any], index: int) -> str:
    return str(document.get("title") or "").strip() or f"Analysis {index + 1}"


def ensure_analysis_id(document: dict[str, Any]) -> dict[str, Any]:
    if not document.get("analysis_id"):
        document["analysis_id"] = _new_id()
    return document


def new_analysis(template: dict[str, Any], title: str, copy_studies: bool = False) -> dict[str, Any]:
    """A new outcome on the same network: shares treatments, study
    characteristics and the network name; starts empty unless copy_studies."""

    import copy

    document = copy.deepcopy(template)
    document["analysis_id"] = _new_id()
    document["title"] = title
    document.pop("cinema", None)
    if not copy_studies:
        document["studies"] = []
    return document


def share_common(source: dict[str, Any], target: dict[str, Any]) -> dict[str, Any]:
    """Copy the shared parts (treatments, characteristics, network name) of the
    analysis being left into the one being opened."""

    import copy

    for key in SHARED_KEYS:
        if key in source:
            target[key] = copy.deepcopy(source[key])
    return target


def bundle_documents(documents: list[dict[str, Any]], active: int = 0) -> dict[str, Any]:
    """File content: the plain single-analysis document when there is one
    analysis (same as older files), otherwise
    {"format", "version", "name", "treatments", "study_characteristics",
     "active", "analyses": [{analysis_id, title, spec, studies, cinema}, ...]}."""

    if len(documents) == 1:
        return documents[0]
    active = min(max(active, 0), len(documents) - 1)
    shared = documents[active]
    return {
        "format": FILE_FORMAT,
        "version": 2,
        "name": shared.get("name", ""),
        "treatments": shared.get("treatments", []),
        "study_characteristics": shared.get("study_characteristics", []),
        "active": active,
        "analyses": [{key: document[key] for key in ANALYSIS_KEYS if key in document} for document in documents],
    }


def documents_from_file(data: dict[str, Any]) -> tuple[list[dict[str, Any]], int]:
    """Inverse of bundle_documents; also accepts a plain single-analysis file."""

    if not isinstance(data, dict):
        raise ValueError("The file does not contain an NMA document")
    if "analyses" not in data:
        return [ensure_analysis_id(dict(data))], 0
    analyses = data.get("analyses") or []
    if not isinstance(analyses, list) or not analyses:
        raise ValueError("The file has no analyses")
    shared = {key: data[key] for key in SHARED_KEYS if key in data}
    documents = []
    for analysis in analyses:
        if not isinstance(analysis, dict):
            raise ValueError("Each analysis must be an object")
        documents.append(ensure_analysis_id({**shared, **analysis}))
    active = data.get("active", 0)
    active = active if isinstance(active, int) and 0 <= active < len(documents) else 0
    return documents, active


def slug(text: str) -> str:
    import re

    return re.sub(r"[^0-9a-zA-Z]+", "_", text or "").strip("_").lower()[:30]


# --------------------------------------------------------------------------
# Contribution matrix and CINeMA tables
# --------------------------------------------------------------------------

CONFIDENCE_COLORS = {"High": "#1a9641", "Moderate": "#8cc63f", "Low": "#f59e0b", "Very low": "#d7191c"}
CONCERN_COLORS = {"No concerns": "#dcfce7", "Undetected": "#dcfce7", "Some concerns": "#fef3c7",
                  "Suspected": "#fef3c7", "Major concerns": "#fee2e2"}


def contribution_frame(contribution, percent: bool = True) -> pd.DataFrame:
    """Rows = network estimates, columns = direct comparisons (in %)."""

    rows = []
    for record in contribution.rows:
        row = {"Network estimate": record.label}
        for edge in contribution.edges:
            value = record.edges.get(edge, 0.0)
            row[contribution.edge_label(edge)] = round(value * 100, 1) if percent else value
        rows.append(row)
    return pd.DataFrame(rows)


def contribution_html_table(contribution, labels: dict[str, str] | None = None) -> str:
    """Heat-map table (darker blue = larger share); header row dark blue."""

    import html as html_lib

    labels = labels or {}
    head = [labels.get("estimate", "Network estimate"), *[contribution.edge_label(e) for e in contribution.edges]]
    th = "".join(
        f"<th style='background:{TABLE_HEADER_BG};color:{TABLE_HEADER_FG};font-weight:600;padding:8px 10px;"
        f"text-align:{'left' if i == 0 else 'center'};white-space:nowrap;position:sticky;top:0'>{html_lib.escape(name)}</th>"
        for i, name in enumerate(head))
    body = []
    for record in contribution.rows:
        cells = [f"<td style='padding:6px 10px;font-weight:600;white-space:nowrap'>{html_lib.escape(record.label)}</td>"]
        for edge in contribution.edges:
            value = record.edges.get(edge, 0.0)
            alpha = min(1.0, value * 1.4)
            color = "#ffffff" if alpha > 0.55 else "#0f172a"
            text = "" if value < 0.0005 else f"{value * 100:.1f}"
            border = "outline:2px solid #ea580c;outline-offset:-2px;" if edge == (record.treatment, record.versus) else ""
            cells.append(f"<td style='padding:6px 10px;text-align:center;background:rgba(30,58,138,{alpha:.3f});"
                         f"color:{color};{border}'>{text}</td>")
        body.append(f"<tr style='border-top:1px solid #e5eaf2'>{''.join(cells)}</tr>")
    return (
        "<div style='overflow-x:auto;border:1px solid #dbe2ee;border-radius:8px;margin-bottom:8px'>"
        "<table style='border-collapse:collapse;font-size:13.5px;color:#0f172a;width:100%'>"
        f"<thead><tr>{th}</tr></thead><tbody>{''.join(body)}</tbody></table></div>"
    )


def cinema_html_table(cinema, domain_labels: dict[str, str], level_labels: dict[str, str],
                      headers: dict[str, str], digits: int = 2) -> str:
    """Summary table: estimate (CI), PI, direct share, six domains as coloured
    chips, overall confidence as a filled badge. Overridden judgements get *."""

    import html as html_lib

    from heor_tree_engine.nma_cinema import DOMAINS

    def fmt(value, lower, upper) -> str:
        return format_effect(value, lower, upper, digits)

    columns = [headers.get("comparison", "Comparison"), headers.get("estimate", "Estimate (CI)"),
               headers.get("pi", "Prediction interval"), headers.get("direct", "Direct %"),
               *[domain_labels.get(d, d) for d in DOMAINS], headers.get("confidence", "Confidence")]
    th = "".join(
        f"<th style='background:{TABLE_HEADER_BG};color:{TABLE_HEADER_FG};font-weight:600;padding:8px 10px;"
        f"text-align:left;position:sticky;top:0;vertical-align:bottom'>{html_lib.escape(name)}</th>"
        for name in columns)
    body = []
    for position, row in enumerate(cinema.rows):
        pi = ("–" if row.natural_pred_lower is None
              else f"{row.natural_pred_lower:.{digits}f} to {row.natural_pred_upper:.{digits}f}")
        cells = [
            f"<td style='padding:6px 10px;font-weight:600;white-space:nowrap'>{html_lib.escape(row.label)}</td>",
            f"<td style='padding:6px 10px;white-space:nowrap'>{html_lib.escape(fmt(row.natural, row.natural_lower, row.natural_upper))}</td>",
            f"<td style='padding:6px 10px;white-space:nowrap'>{html_lib.escape(pi)}</td>",
            f"<td style='padding:6px 10px;text-align:right'>{row.direct_share * 100:.0f}</td>",
        ]
        for domain in DOMAINS:
            judgement = row.domains[domain]
            label = level_labels.get(judgement.level, judgement.level) + (" *" if judgement.overridden else "")
            cells.append(
                f"<td style='padding:5px 8px' title='{html_lib.escape(judgement.reason, quote=True)}'>"
                f"<span style='display:inline-block;padding:2px 8px;border-radius:10px;white-space:nowrap;"
                f"background:{CONCERN_COLORS.get(judgement.level, '#f1f5f9')};font-size:12.5px'>"
                f"{html_lib.escape(label)}</span></td>")
        confidence = level_labels.get(row.confidence, row.confidence) + (" *" if row.confidence_overridden else "")
        cells.append(
            f"<td style='padding:5px 8px'><span style='display:inline-block;padding:3px 10px;border-radius:10px;"
            f"white-space:nowrap;font-weight:700;color:#ffffff;background:{CONFIDENCE_COLORS.get(row.confidence, '#64748b')}'>"
            f"{html_lib.escape(confidence)}</span></td>")
        background = "#f5f8fe" if position % 2 else "#ffffff"
        body.append(f"<tr style='background:{background};border-top:1px solid #e5eaf2'>{''.join(cells)}</tr>")
    return (
        "<div style='overflow-x:auto;border:1px solid #dbe2ee;border-radius:8px;margin-bottom:8px'>"
        "<table style='border-collapse:collapse;font-size:13.5px;color:#0f172a;width:100%'>"
        f"<thead><tr>{th}</tr></thead><tbody>{''.join(body)}</tbody></table></div>"
    )


# --------------------------------------------------------------------------
# Ranking across outcomes (rank-heat plot; Veroniki et al. 2016)
# --------------------------------------------------------------------------


def rank_heat_frame(rankings: list[tuple[str, dict[str, float]]], treatment_order: list[str]) -> pd.DataFrame:
    """Rows = treatments, columns = outcomes, values = P-score / SUCRA (0-1)
    or NaN when the treatment is not in that outcome's network."""

    names = [name for name in treatment_order if any(name in scores for _, scores in rankings)]
    names += sorted({name for _, scores in rankings for name in scores} - set(names))
    return pd.DataFrame({title: [scores.get(name, float("nan")) for name in names] for title, scores in rankings},
                        index=pd.Index(names, name="Treatment"))


def _heat_color(value: float) -> str:
    """0 = red, 0.5 = amber, 1 = green (text stays dark)."""

    stops = [(0.0, (239, 68, 68)), (0.5, (250, 204, 21)), (1.0, (34, 197, 94))]
    for (x0, c0), (x1, c1) in zip(stops, stops[1:]):
        if value <= x1:
            t = (value - x0) / (x1 - x0)
            r, g, b = (round(a + (b_ - a) * t) for a, b_ in zip(c0, c1))
            return f"rgba({r},{g},{b},0.75)"
    return "rgba(34,197,94,0.75)"


def rank_heat_html(frame: pd.DataFrame, treatment_label: str = "Treatment") -> str:
    import html as html_lib

    th = "".join(
        f"<th style='background:{TABLE_HEADER_BG};color:{TABLE_HEADER_FG};font-weight:600;padding:8px 12px;"
        f"text-align:{'left' if i == 0 else 'center'};white-space:nowrap'>{html_lib.escape(str(name))}</th>"
        for i, name in enumerate([treatment_label, *frame.columns]))
    body = []
    for name, row in frame.iterrows():
        cells = [f"<td style='padding:7px 12px;font-weight:600;white-space:nowrap'>{html_lib.escape(str(name))}</td>"]
        for value in row.tolist():
            if value is None or (isinstance(value, float) and math.isnan(value)):
                cells.append("<td style='padding:7px 12px;text-align:center;color:#94a3b8'>–</td>")
            else:
                cells.append(f"<td style='padding:7px 12px;text-align:center;background:{_heat_color(float(value))}'>"
                             f"{value * 100:.0f}</td>")
        body.append(f"<tr style='border-top:1px solid #e5eaf2'>{''.join(cells)}</tr>")
    return ("<div style='overflow-x:auto;border:1px solid #dbe2ee;border-radius:8px;margin-bottom:8px'>"
            "<table style='border-collapse:collapse;font-size:14px;color:#0f172a;width:100%'>"
            f"<thead><tr>{th}</tr></thead><tbody>{''.join(body)}</tbody></table></div>")


# --------------------------------------------------------------------------
# Autosave (a file in the project's autosave/ folder)
# --------------------------------------------------------------------------

AUTOSAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "autosave")
AUTOSAVE_FILE = os.path.join(AUTOSAVE_DIR, "nma-autosave.json")
AUTOSAVE_KEEP = 5  # numbered backups nma-autosave-1.json (newest) ... -5


def autosave_fingerprint(data: dict[str, Any]) -> str:
    import hashlib

    return hashlib.md5(json.dumps(data, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")).hexdigest()


def autosave_write(data: dict[str, Any]) -> str:
    """Write the file content (as Save would) to autosave/, rotating the
    previous copies. Returns the path; never raises (best effort)."""

    import datetime
    import shutil

    try:
        os.makedirs(AUTOSAVE_DIR, exist_ok=True)
        if os.path.exists(AUTOSAVE_FILE):
            for index in range(AUTOSAVE_KEEP - 1, 0, -1):
                older = os.path.join(AUTOSAVE_DIR, f"nma-autosave-{index}.json")
                if os.path.exists(older):
                    shutil.move(older, os.path.join(AUTOSAVE_DIR, f"nma-autosave-{index + 1}.json"))
            shutil.move(AUTOSAVE_FILE, os.path.join(AUTOSAVE_DIR, "nma-autosave-1.json"))
        payload = {**data, "autosaved_at": datetime.datetime.now().isoformat(timespec="seconds")}
        tmp = AUTOSAVE_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=1)
        os.replace(tmp, AUTOSAVE_FILE)
    except OSError:
        pass
    return AUTOSAVE_FILE


def autosave_read() -> dict[str, Any] | None:
    try:
        with open(AUTOSAVE_FILE, encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None


def autosave_discard() -> None:
    try:
        if os.path.exists(AUTOSAVE_FILE):
            os.replace(AUTOSAVE_FILE, os.path.join(AUTOSAVE_DIR, "nma-autosave-discarded.json"))
    except OSError:
        pass


def threshold_frame(result, scope: str = "studies", labels: dict[str, str] | None = None) -> pd.DataFrame:
    """Table of invariant intervals from `nma_threshold.ThresholdResult`. Values on the natural scale (ratio
    measures exponentiated). scope = "studies" (one row per study contrast) or "network" (per NMA estimate)."""

    L = {"study": "Study", "comparison": "Comparison", "estimate": "Estimate", "lower": "Lower limit",
         "lower_best": "Best treatment below", "upper": "Upper limit", "upper_best": "Best treatment above",
         "fragile": "Within 95% interval of the data", "none": "never", "yes": "yes", "no": "no"}
    L.update(labels or {})
    rows = result.rows if scope == "studies" else result.network_rows
    data = []
    for row in rows:
        data.append({
            L["study"]: row.study, L["comparison"]: row.comparison, L["estimate"]: round(row.natural, 4),
            L["lower"]: L["none"] if row.natural_lower is None else round(row.natural_lower, 4),
            L["lower_best"]: row.lower_new_best or "",
            L["upper"]: L["none"] if row.natural_upper is None else round(row.natural_upper, 4),
            L["upper_best"]: row.upper_new_best or "",
            L["fragile"]: L["yes"] if row.fragile else L["no"],
        })
    return pd.DataFrame(data)


def component_effects_frame(result, interval_label: str = "CI", labels: dict[str, str] | None = None) -> pd.DataFrame:
    """Effect of adding each component (additive component NMA), natural scale."""

    L = {"component": "Component", "effect": "Effect", "se": "SE"}
    L.update(labels or {})
    return pd.DataFrame([{
        L["component"]: e.treatment, L["effect"]: round(e.natural, 4),
        f"{interval_label}": f"{e.natural_lower:.3f} to {e.natural_upper:.3f}", L["se"]: round(e.se, 4)}
        for e in result.component_effects])


def component_treatments_frame(result, labels: dict[str, str] | None = None) -> pd.DataFrame:
    """Additive-model effect of each treatment against the reference, beside the standard NMA estimate."""

    L = {"treatment": "Treatment", "components": "Components", "additive": "Additive model", "standard": "Standard NMA"}
    L.update(labels or {})
    standard = {}
    if result.standard is not None:
        fit = result.standard.random if result.effects == "random" else result.standard.fixed
        standard = {e.treatment: e for e in fit.effects}
    rows = []
    for e in result.fit.effects:
        s = standard.get(e.treatment)
        rows.append({
            L["treatment"]: e.treatment, L["components"]: " + ".join(result.membership.get(e.treatment, [])),
            L["additive"]: f"{e.natural:.3f} ({e.natural_lower:.3f} to {e.natural_upper:.3f})",
            L["standard"]: "" if s is None else f"{s.natural:.3f} ({s.natural_lower:.3f} to {s.natural_upper:.3f})"})
    return pd.DataFrame(rows)


def design_frame(result, labels: dict[str, str] | None = None) -> pd.DataFrame:
    """Treatment effects vs the reference under each non-randomised-evidence scenario (natural scale)."""

    L = {"treatment": "Treatment", "tau2": "tau²", "studies": "Studies"}
    L.update(labels or {})
    treatments = []
    for scenario in result.scenarios:
        for name in scenario.effects:
            if name not in treatments:
                treatments.append(name)
    rows = [{L["treatment"]: name, **{
        s.name: ("" if name not in s.effects else "{:.3f} ({:.3f} to {:.3f})".format(*s.effects[name]))
        for s in result.scenarios}} for name in treatments]
    rows.append({L["treatment"]: L["tau2"], **{s.name: "" if s.error else f"{s.tau2:.3f}" for s in result.scenarios}})
    rows.append({L["treatment"]: L["studies"], **{s.name: str(s.n_studies) for s in result.scenarios}})
    return pd.DataFrame(rows)


def piecewise_frame(result, interval_unit: str = "", labels: dict[str, str] | None = None) -> pd.DataFrame:
    """Piecewise HR NMA: treatment x interval, HR (CI)."""

    L = {"treatment": "Treatment", "tau2": "tau²", "studies": "Studies"}
    L.update(labels or {})
    unit = f" {interval_unit}" if interval_unit else ""
    names = []
    for e in result.estimates:
        if e.treatment not in names:
            names.append(e.treatment)
    rows = []
    for name in names:
        row = {L["treatment"]: name}
        for t0, t1 in result.intervals:
            cell = next((e for e in result.estimates if e.treatment == name and (e.t_from, e.t_to) == (t0, t1)), None)
            row[f"{t0:g}-{t1:g}{unit}"] = "" if cell is None else f"{cell.hr:.3f} ({cell.lower:.3f} to {cell.upper:.3f})"
        rows.append(row)
    for key, fmt in (("tau2", "{:.3f}"), ("studies", "{}")):
        row = {L["treatment"]: L[key]}
        for t0, t1 in result.intervals:
            cell = next((e for e in result.estimates if (e.t_from, e.t_to) == (t0, t1)), None)
            value = None if cell is None else (cell.tau2 if key == "tau2" else cell.n_studies)
            row[f"{t0:g}-{t1:g}{unit}"] = "" if value is None else fmt.format(value)
        rows.append(row)
    return pd.DataFrame(rows)


def ph_frame(result, labels: dict[str, str] | None = None) -> pd.DataFrame:
    L = {"scope": "Level", "item": "Contrast / treatment", "intervals": "Intervals", "q": "Q", "df": "df", "p": "p",
         "flag": "HR changes over time?", "study": "study", "network": "network", "yes": "possibly", "no": "no evidence"}
    L.update(labels or {})
    return pd.DataFrame([{
        L["scope"]: L[r.scope], L["item"]: r.label, L["intervals"]: r.n_intervals, L["q"]: round(r.q, 2), L["df"]: r.df,
        L["p"]: "" if r.p is None else f"{r.p:.3f}", L["flag"]: L["yes"] if r.flagged else L["no"]} for r in result.ph])


def survival_models_frame(fits: dict, labels: dict[str, str] | None = None) -> pd.DataFrame:
    """DIC comparison of fitted survival NMA models: {label: SurvivalResult}."""

    L = {"model": "Model", "effects": "Effects", "pd": "pD", "dbar": "Mean deviance", "dic": "DIC", "delta": "ΔDIC",
         "rhat": "Max R-hat", "fixed": "fixed", "random": "random"}
    L.update(labels or {})
    best = min((r.dic for r in fits.values() if r.dic == r.dic), default=float("nan"))
    rows = []
    for label, r in fits.items():
        rhat = max((d.rhat for d in r.diagnostics if d.rhat == d.rhat), default=float("nan"))
        rows.append({L["model"]: label, L["effects"]: L[r.effects], L["pd"]: round(r.p_d, 1), L["dbar"]: round(r.dbar, 1),
                     L["dic"]: round(r.dic, 1), L["delta"]: round(r.dic - best, 1), L["rhat"]: round(rhat, 3)})
    return pd.DataFrame(rows)


def survival_effects_frame(result, labels: dict[str, str] | None = None) -> pd.DataFrame:
    L = {"parameter": "Parameter", "median": "Median", "interval": "95% CrI", "rhat": "R-hat", "ess": "ESS"}
    L.update(labels or {})
    return pd.DataFrame([{L["parameter"]: d.name, L["median"]: round(d.median, 3),
                          L["interval"]: f"{d.lower:.3f} to {d.upper:.3f}", L["rhat"]: round(d.rhat, 3),
                          L["ess"]: int(round(d.ess))} for d in result.diagnostics])


def maic_frames(effect, labels: dict[str, str] | None = None) -> dict[str, pd.DataFrame]:
    """Tables of a MAIC result (nma_maic.MaicEffect): balance and effects."""

    L = {"covariate": "Covariate", "unweighted": "IPD, unweighted", "weighted": "IPD, weighted", "target": "Target",
         "analysis": "Analysis", "estimate": "Estimate", "interval": "95% CI", "ess": "ESS", "unadjusted": "Unadjusted",
         "adjusted": "Adjusted (MAIC)", "method": "SE method"}
    L.update(labels or {})
    balance = pd.DataFrame([{L["covariate"]: b["covariate"], L["unweighted"]: round(b["unweighted"], 3),
                             L["weighted"]: round(b["weighted"], 3), L["target"]: round(b["target"], 3)}
                            for b in (effect.weights.balance if effect.weights else [])])
    transform = math.exp if effect.ratio else (lambda v: v)

    def ci(est, se):
        return f"{transform(est - 1.959963984540054 * se):.3f} to {transform(est + 1.959963984540054 * se):.3f}"

    rows = [{L["analysis"]: L["unadjusted"], L["estimate"]: round(transform(effect.unadjusted), 3),
             L["interval"]: ci(effect.unadjusted, effect.unadjusted_se), L["ess"]: effect.n, L["method"]: "model-based"},
            {L["analysis"]: L["adjusted"], L["estimate"]: round(effect.natural, 3), L["interval"]: ci(effect.estimate, effect.se),
             L["ess"]: round(effect.ess, 1), L["method"]: effect.se_method}]
    return {"balance": balance, "effects": pd.DataFrame(rows)}


def mlnmr_frames(result, profile, labels: dict[str, str] | None = None) -> dict[str, pd.DataFrame]:
    """Tables of an ML-NMR fit: coefficients and the conditional odds ratios at a covariate profile."""

    L = {"parameter": "Parameter", "estimate": "Estimate", "se": "SE", "treatment": "Treatment",
         "or": "Odds ratio vs reference", "interval": "95% CI", "profile": "Covariate profile"}
    L.update(labels or {})
    coefficients = pd.DataFrame([{L["parameter"]: n, L["estimate"]: round(float(v), 4), L["se"]: round(result.se(n), 4)}
                                 for n, v in zip(result.names, result.theta) if not n.startswith("mu[")])
    rows = []
    for t in result.treatments[1:]:
        est, se = result.effect(t, profile)
        rows.append({L["treatment"]: t, L["or"]: round(math.exp(est), 3),
                     L["interval"]: f"{math.exp(est - 1.959963984540054 * se):.3f} to {math.exp(est + 1.959963984540054 * se):.3f}",
                     L["profile"]: ", ".join(f"{c} = {v:g}" for c, v in zip(result.covariates, profile))})
    return {"coefficients": coefficients, "effects": pd.DataFrame(rows)}


def mbnma_models_frame(fits: dict, labels: dict[str, str] | None = None) -> pd.DataFrame:
    L = {"function": "Function", "aic": "AIC", "delta": "ΔAIC", "dev": "Residual deviance", "df": "df", "p": "Fit p",
         "disp": "Dispersion"}
    L.update(labels or {})
    best = min((f.aic for f in fits.values() if f.aic == f.aic), default=float("nan"))
    return pd.DataFrame([{L["function"]: k, L["aic"]: round(f.aic, 1), L["delta"]: round(f.aic - best, 1),
                          L["dev"]: round(f.deviance, 1), L["df"]: f.df, L["p"]: "" if f.gof_p is None else f"{f.gof_p:.3f}",
                          L["disp"]: round(f.dispersion, 2)} for k, f in fits.items()])


def mbnma_params_frame(fit, labels: dict[str, str] | None = None) -> pd.DataFrame:
    L = {"parameter": "Parameter", "estimate": "Estimate", "se": "SE", "interval": "95% CI"}
    L.update(labels or {})
    return pd.DataFrame([{L["parameter"]: n, L["estimate"]: round(float(v), 4), L["se"]: round(fit.se(n), 4),
                          L["interval"]: f"{v - 1.959963984540054 * fit.se(n):.3f} to {v + 1.959963984540054 * fit.se(n):.3f}"}
                         for n, v in zip(fit.parameter_names, fit.theta)])
