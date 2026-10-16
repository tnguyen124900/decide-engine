"""Build a MarkovModel from its flat JSON representation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .markov_models import MarkovModel, MarkovState, MarkovTransition


def _numeric_ref(value: Any) -> float | str:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError("Numeric references must be numbers or parameter names")
    return float(value) if isinstance(value, (int, float)) else value


def _parse_state(raw: Mapping[str, Any]) -> MarkovState:
    if "id" not in raw:
        raise ValueError("A Markov state requires an 'id'")
    recurring_cost = raw.get("recurring_cost")
    extra_outcomes_raw = raw.get("extra_outcomes") or {}
    return MarkovState(
        id=str(raw["id"]),
        name=str(raw.get("name", raw["id"])),
        cost=_numeric_ref(raw.get("cost", 0.0)),
        tenure_applied=bool(raw.get("tenure_applied", False)),
        tenure_count=int(raw.get("tenure_count", 1)),
        initial_cost=_numeric_ref(raw.get("initial_cost", 0.0)),
        recurring_cost=_numeric_ref(recurring_cost) if recurring_cost is not None else None,
        terminal_cost=_numeric_ref(raw.get("terminal_cost", 0.0)),
        effect=_numeric_ref(raw.get("effect", 0.0)),
        initial_prob=_numeric_ref(raw.get("initial_prob", 0.0)),
        is_absorbing=bool(raw.get("is_absorbing", False)),
        extra_outcomes={str(k): _numeric_ref(v) for k, v in extra_outcomes_raw.items()},
        # "canvas_parent" is the DECIDE app Markov canvas's own marker for
        # a state created as a double-click child of another state (see
        # MarkovState.is_real_state's docstring) -- absent/None/"" means a
        # direct child of the strategy root, i.e. a real health state.
        is_real_state=not raw.get("canvas_parent"),
        # "canvas_alias_of" is set by the DECIDE app's "relabel as
        # terminal" canvas action (see MarkovState.alias_of's docstring).
        alias_of=(str(raw["canvas_alias_of"]) if raw.get("canvas_alias_of") else None),
    )


def _extra_outcomes(document: Mapping[str, Any]) -> list[dict[str, str]]:
    """Cost/Effect rows of the Outcomes table beyond the primary pair --
    same convention as abm_parser._extra_outcomes()/dtm_parser._extra_outcomes().
    ICER/NMB/CER/NHB rows are display settings only, not engine inputs."""

    extras = []
    for row in document.get("outcomes") or []:
        if row.get("type") in ("cost", "effect") and row.get("id") not in ("cost", "effect"):
            extras.append({"id": str(row["id"]), "type": str(row["type"]), "label": str(row.get("label", row["id"]))})
    return extras


def _parse_transition(raw: Mapping[str, Any]) -> MarkovTransition:
    if "from_state" not in raw or "to_state" not in raw:
        raise ValueError("A transition requires 'from_state' and 'to_state'")
    extra_increment_raw = raw.get("extra_increment") or {}
    return MarkovTransition(
        from_state=str(raw["from_state"]),
        to_state=str(raw["to_state"]),
        probability=_numeric_ref(raw.get("probability", 0.0)),
        cost_increment=_numeric_ref(raw.get("cost_increment", 0.0)),
        effect_increment=_numeric_ref(raw.get("effect_increment", 0.0)),
        extra_increment={str(k): _numeric_ref(v) for k, v in extra_increment_raw.items()},
    )


def resolve_markov(document: Mapping[str, Any]) -> MarkovModel:
    """Parse a flat Markov JSON document into a MarkovModel."""

    if "states" not in document:
        raise ValueError("Markov document requires 'states'")
    states = [_parse_state(raw) for raw in document["states"]]
    ids = [state.id for state in states]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate state id in Markov document")
    transitions = [_parse_transition(raw) for raw in document.get("transitions", [])]
    return MarkovModel(
        id=str(document.get("id", "markov")),
        name=str(document.get("name", "Markov model")),
        extra_outcomes=_extra_outcomes(document),
        states=states,
        transitions=transitions,
        cycles=int(document.get("cycles", 10)),
        half_cycle_correction=bool(document.get("half_cycle_correction", True)),
    )
