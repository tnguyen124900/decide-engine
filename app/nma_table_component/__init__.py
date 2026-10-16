"""Editable table for the NMA Editor with a dark-blue header.

Why this exists instead of st.data_editor: data_editor draws its header on
a <canvas>, and Streamlit only lets the theme change that header's
background, never its (grey) label colour - so a dark-blue header with
white labels is impossible there. This is a plain HTML table of <input>s,
served from a single static frontend/index.html with the same hand-rolled
Streamlit Components v1 protocol as tree_canvas_component (no npm build,
no extra package).

Data flow: Python passes the baseline `rows` (list of dicts, all fields)
plus a `version`. The frontend keeps its own edited copy and only resets
to `rows` when `version` changes, so a rerun triggered by an edit never
wipes the edit or steals focus. Every committed edit (cell change on blur
or Enter, add row, delete row, undo, redo) returns {"version", "rows",
"nonce"}. Undo/redo: Cmd/Ctrl+Z and Shift+Cmd/Ctrl+Z (or Ctrl+Y) while the
table has focus, or the Undo / Redo buttons under it.

Paste from Excel / Sheets: multi-cell text pasted into a cell fills from
that cell rightwards and downwards (rows are added as needed, undoable);
the "Paste table" panel takes a whole block, matches a header row to the
columns (labels, keys or common aliases) and replaces or appends.
`paste_labels` = {"label", "hint", "replace", "append", "close", "empty"}.
"""

from __future__ import annotations

import os
from typing import Any

import streamlit.components.v1 as components

_FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "frontend")

_component_func = components.declare_component("nma_table", path=_FRONTEND_DIR)


def nma_table(
    columns: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    version: int,
    max_height: int = 420,
    add_label: str = "+ Add row",
    delete_label: str = "Delete row",
    undo_label: str = "\u21b6 Undo (\u2318Z)",
    redo_label: str = "\u21b7 Redo (\u21e7\u2318Z)",
    header_bg: str = "#1e3a8a",
    header_fg: str = "#ffffff",
    paste_labels: dict[str, str] | None = None,
    key: str | None = None,
) -> list[dict[str, Any]] | None:
    """Render the table; return the edited rows for this `version`, or None
    if nothing has been edited since the baseline was set.

    columns: [{"key": "events", "label": "Events", "type": "number"|"text",
               "width": 90 (px, optional), "placeholder": "..." (optional)}]
    """

    value = _component_func(
        columns=columns,
        rows=rows,
        version=version,
        max_height=max_height,
        add_label=add_label,
        delete_label=delete_label,
        undo_label=undo_label,
        redo_label=redo_label,
        header_bg=header_bg,
        header_fg=header_fg,
        **{f"paste_{name}": text for name, text in (paste_labels or {}).items()},
        key=key,
        default=None,
    )
    if isinstance(value, dict) and value.get("version") == version:
        return list(value.get("rows") or [])
    return None
