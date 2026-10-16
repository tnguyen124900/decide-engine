"""A small, dependency-free custom Streamlit component for the Markov
canvas.

Why this exists (instead of the `streamlit-flow-component` package used by
the Decision Tree canvas): that third-party component wraps React Flow but
does not expose enough of its API for a completely fixed camera -- it has
no way to turn mouse-wheel scroll into a page scroll instead of a zoom
(`panOnScroll` isn't exposed by its Python wrapper), and every recomputed
layout (Auto-arrange) has to remount the whole component, which resets
whatever pan/zoom the user had. Both are exactly the behaviors the user
asked us not to have: "a diagram that keeps moving/isn't stationary when
you scroll the mouse is a basic thing that's unacceptable -- it has to be
at least as good as Amua and TreeAge here."

This component renders the Markov tree as a plain SVG with no camera at
all: there is no pan and no zoom-on-scroll, ever. The SVG is always drawn
at 1:1 (times the explicit +/- zoom level from the toolbar) and sized to
fit its whole content, so a large model simply makes the component -- and
with it the surrounding page -- taller; the browser's ordinary page
scrollbar is what lets you move through a tall model, exactly like
scrolling any other tall webpage, never a canvas-internal camera. Because
it's plain SVG re-rendered from fresh props on every Streamlit rerun (not
a library that only reads node positions once at mount), recomputing the
layout (Auto-arrange) never needs to reset a viewport in the first place --
there isn't one to reset.

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
    "markov_canvas",
    path=_FRONTEND_DIR,
)


def markov_canvas(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    zoom: float = 1.0,
    selected_id: str | None = None,
    viewport_height: int = 720,
    reset_scroll_nonce: int = 0,
    export_nonce: int = 0,
    selected_composite_id: str | None = None,
    key: str | None = None,
) -> dict[str, Any] | None:
    """Render the Markov canvas and return the latest user interaction.

    Returns a dict shaped like one of:
      {"type": "select", "id": <node id>}
      {"type": "move", "id": <node id>, "x": <float>, "y": <float>}
      {"type": "add_node", "x": <float>, "y": <float>}
      {"type": "delete_node", "id": <node id>}
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
        selected_composite_id=selected_composite_id,
        key=key,
        default=None,
    )
