"""A small, dependency-free custom Streamlit component for the Partitioned
Survival Model (PSM) canvas.

Forked from tree_canvas_component/__init__.py (see that file's own
docstring for why this hand-rolled-SVG-no-camera architecture exists at
all instead of a third-party React Flow wrapper). This is a fork rather
than a shared/parameterized component, matching this codebase's existing
convention: tree_canvas_component and markov_canvas_component are
themselves two independently-maintained near-copies, not one
parameterized file. Forking here also guarantees an ordinary Decision
Tree model never sees a "PSM node" menu option -- Tree keeps using
tree_canvas_component unmodified.

The only content difference from tree_canvas_component/frontend/index.html
is a 4th node kind, "psm" (light blue circle, dark blue border, black bold
"P" -- see SHAPE.psm in frontend/index.html), available alongside
chance/terminal (never "decision", never "markov") in the add-node and
change-type menus.
"""

from __future__ import annotations

import os
from typing import Any

import streamlit.components.v1 as components

_FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "frontend")

_component_func = components.declare_component(
    "psm_canvas",
    path=_FRONTEND_DIR,
)


def psm_canvas(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    zoom: float = 1.0,
    selected_id: str | None = None,
    viewport_height: int = 720,
    reset_scroll_nonce: int = 0,
    export_nonce: int = 0,
    key: str | None = None,
) -> dict[str, Any] | None:
    """Render the PSM canvas and return the latest user interaction.

    Returns a dict shaped like one of:
      {"type": "select", "id": <node id>}
      {"type": "move", "id": <node id>, "x": <float>, "y": <float>}
      {"type": "add_node", "parent_id": <node id>, "child_kind": "chance"|"psm"|"terminal"}
      {"type": "change_node_kind", "id": <node id>, "canvas_kind": "chance"|"psm"|"terminal"}
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
        key=key,
        default=None,
    )
