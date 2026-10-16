"""A small, dependency-free custom Streamlit component for the Decision
Tree canvas.

Why this exists (instead of the `streamlit-flow-component` package the
Decision Tree canvas used before): that third-party component wraps React
Flow but does not expose enough of its API for a fully fixed camera -- it
has no way to turn mouse-wheel scroll into a page scroll instead of a zoom
(`panOnScroll` isn't exposed by its Python wrapper), and every recomputed
layout (Auto-arrange) -- and in fact nearly every structural edit (add,
delete, insert, cut, paste) -- has to remount the whole component, which
resets whatever pan/zoom the user had back to a fresh-mount default (often
visible as the canvas snapping to a different zoom than the toolbar's +/-
control shows, e.g. jumping to 100% while the toolbar still reads 75%).

This is the exact same problem the Markov canvas had, and it was fixed
there by replacing streamlit-flow-component with a small hand-rolled SVG
component (see markov_canvas_component/__init__.py for the original
writeup). This component is that same architecture, ported for the
Decision Tree's node/branch data model: a plain SVG with no camera at all
-- no pan, no zoom-on-scroll, ever. The SVG is always drawn at 1:1 (times
the explicit +/- zoom level from the toolbar) and sized to fit its whole
content plus a comfortable buffer, so the grid always extends past the
last node (feels "infinite") and a large tree just makes the component --
and with it the surrounding scroll area -- taller/wider, never a
canvas-internal camera to lose track of.

Because it's plain SVG re-rendered from fresh props on every Streamlit
rerun (not a library that only reads node positions once at mount),
recomputing the layout (Auto-arrange) never needs to reset a viewport in
the first place -- there isn't one to reset.

No JS build step/npm dependency: the frontend is a single static
`frontend/index.html` that reimplements the small part of the Streamlit
Components v1 postMessage protocol it needs directly (componentReady /
render / setComponentValue / setFrameHeight), and pulls in nothing else
(no React, no CDN scripts). `components.v1.declare_component(path=...)`
serves that folder as-is.
"""

from __future__ import annotations

import os
from typing import Any

import streamlit.components.v1 as components

_FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "frontend")

_component_func = components.declare_component(
    "tree_canvas",
    path=_FRONTEND_DIR,
)


def tree_canvas(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    zoom: float = 1.0,
    selected_id: str | None = None,
    viewport_height: int = 720,
    reset_scroll_nonce: int = 0,
    export_nonce: int = 0,
    text_scale: float = 1.0,
    key: str | None = None,
) -> dict[str, Any] | None:
    """Render the Decision Tree canvas and return the latest user
    interaction.

    Returns a dict shaped like one of:
      {"type": "select", "id": <node id>}
      {"type": "move", "id": <node id>, "x": <float>, "y": <float>}
      {"type": "add_node", "parent_id": <node id>, "child_kind": "chance"|"terminal"}
      {"type": "change_node_kind", "id": <node id>, "canvas_kind": "chance"|"terminal"}
      {"type": "insert_node_before", "id": <node id>}
      {"type": "insert_node_after", "id": <node id>}
      {"type": "copy_node", "id": <node id>}
      {"type": "cut_node", "id": <node id>}
      {"type": "paste_node", "parent_id": <node id>}
      {"type": "delete_node", "id": <node id>}
      {"type": "edit_node", "id": <node id>, "name": <str>}
      {"type": "edit_branch", "editable": {...}, "value": <str>}
      {"type": "quick_branch", "id": <node id>}
      {"type": "undo"}
    or None if the user hasn't interacted with the canvas yet this session.

    Every event carries a "nonce" so the same interaction (e.g. clicking
    the same node twice) is still seen as a fresh event by Streamlit.
    """

    return _component_func(
        nodes=nodes,
        edges=edges,
        zoom=zoom,
        selected_id=selected_id,
        viewport_height=viewport_height,
        reset_scroll_nonce=reset_scroll_nonce,
        export_nonce=export_nonce,
        text_scale=text_scale,
        key=key,
        default=None,
    )
