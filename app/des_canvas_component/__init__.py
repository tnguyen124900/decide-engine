"""A small, dependency-free custom Streamlit component for the DES
(Discrete Event Simulation) canvas.

Forked from psm_canvas_component (itself a fork of tree_canvas_component),
matching this codebase's fork-not-parameterize convention for canvases.
Node vocabulary:
Decision (strategies) -> [Chance ->] DES -> Time -> [event] -> Chance ->
Terminal. Only two node kinds are new -- "des" (purple circle, white bold
"DES") and "time" (maroon circle, white hourglass glyph, no clock hands);
Decision/Chance/Terminal keep exactly the Markov/Tree look (Terminal =
orange triangle). Which child kinds a node may add / change to depends on
where it sits (inside or outside a DES model), so the Python side sends
those lists per node (add_choices / change_choices) instead of the JS
hard-coding them.
"""

from __future__ import annotations

import os
from typing import Any

import streamlit.components.v1 as components

_FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "frontend")

_component_func = components.declare_component(
    "des_canvas",
    path=_FRONTEND_DIR,
)


def des_canvas(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    zoom: float = 1.0,
    selected_id: str | None = None,
    viewport_height: int = 720,
    reset_scroll_nonce: int = 0,
    export_nonce: int = 0,
    key: str | None = None,
) -> dict[str, Any] | None:
    """Render the DES canvas and return the latest user interaction.

    Returns a dict shaped like one of:
      {"type": "select", "id": <node id>}
      {"type": "move", "id": <node id>, "x": <float>, "y": <float>}
      {"type": "add_node", "parent_id": <node id>, "child_kind": "chance"|"des"|"time"|"terminal"}
      {"type": "change_node_kind", "id": <node id>, "canvas_kind": "chance"|"des"|"time"|"terminal"}
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
