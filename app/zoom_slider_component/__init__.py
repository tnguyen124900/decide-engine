"""Zoom slider for the blue toolbar (desktop app): a white line with a white
round handle you can drag, and the % above it. Same hand-rolled Streamlit
Components v1 protocol as nma_table_component (no npm build).

Python passes the current `pct`; the frontend returns {"pct", "nonce"} once a
drag ends (or the line is clicked). The caller only applies a result whose
nonce it has not seen yet, so +/- buttons that change the zoom elsewhere are
never overwritten by an old component value."""

from __future__ import annotations

import os
from typing import Any

import streamlit.components.v1 as components

_FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "frontend")
_component_func = components.declare_component("zoom_slider", path=_FRONTEND_DIR)


def zoom_slider(pct: int, vmin: int, vmax: int, step: int = 1, key: str | None = None) -> dict[str, Any] | None:
    value = _component_func(pct=int(pct), vmin=int(vmin), vmax=int(vmax), step=int(step), key=key, default=None)
    return value if isinstance(value, dict) else None
