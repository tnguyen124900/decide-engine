"""Helpers for stable editor widget identity."""

from __future__ import annotations

import uuid
from typing import Any


def editor_row_key(prefix: str, row: dict[str, Any]) -> str:
    """Build a widget key that follows a row when its list index changes --
    deleting/reordering any OTHER row must never shift this one's key, or
    every text_input/selectbox below the edited position silently shows and
    writes back a stale cached value belonging to whichever row used to sit
    one slot further down (a report that deleting a variable also removed the last row:
    deleting any row looked like it always deleted the last one, because
    every row after the deleted position kept inheriting the row-below's
    old widget state). Index-based keys (f"{prefix}_{field}_{index}") can't
    do this; Python's own id(row) looked tempting but is just a CPython
    memory address -- it changes on every json.load() (Open saved model,
    autosave reload), leaking an ever-growing pile of orphaned
    st.session_state keys, and CPython is explicitly allowed to reuse a
    freed object's address for a later, unrelated dict, which would make
    two completely different rows collide onto the same widget key (a far
    harder bug to reproduce than the index-shift one it replaces, since it
    depends on GC/allocator timing rather than a fixed, repeatable pattern).

    So identity lives in the data itself: a "_id" field, generated once and
    carried in the saved JSON like any other field, lazily backfilled here
    the first time an older row (saved before this existed) is rendered --
    no migration step needed for existing models."""

    row_id = row.get("_id")
    if not row_id:
        row_id = uuid.uuid4().hex
        row["_id"] = row_id
    return f"{prefix}_{row_id}"


def editor_state_canvas_label(
    name: Any,
    fallback: str,
    node_kind: str,
    is_health_state: bool,
) -> str:
    """Mirror Markov canvas labels for direct and nested state nodes."""

    has_custom_name = bool(name) and name != fallback
    show_label = node_kind != "chance" or is_health_state or has_custom_name
    return str(name or fallback) if show_label else ""
