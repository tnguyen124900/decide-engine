"""Table <-> dict helpers for the Markov node inside a decision tree
(the editing panel in app.py shows states and transitions as two tables)."""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

STATE_COLUMNS = ("id", "name", "cost", "effect", "initial_prob", "is_absorbing")
TRANSITION_COLUMNS = ("from_state", "to_state", "probability", "cost_increment", "effect_increment")


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value != value:  # NaN from an empty table cell
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(value)
    return str(value).strip()


def _number_or_expression(value: Any, default: float = 0.0) -> float | str:
    text = _text(value)
    if text == "":
        return default
    try:
        return float(text)
    except ValueError:
        return text


def states_to_rows(states: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "id": str(s.get("id", "")),
            "name": str(s.get("name", s.get("id", ""))),
            "cost": _text(s.get("cost", 0.0)) or "0",
            "effect": _text(s.get("effect", 0.0)) or "0",
            "initial_prob": _text(s.get("initial_prob", 0.0)) or "0",
            "is_absorbing": bool(s.get("is_absorbing", False)),
        }
        for s in states
    ]


def rows_to_states(rows: Sequence[Mapping[str, Any]], existing: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Table rows -> state dicts. Fields the table does not show (tenure
    settings, canvas info...) are kept from the existing state with the same id."""

    by_id = {str(s.get("id")): dict(s) for s in existing}
    result: list[dict[str, Any]] = []
    for row in rows:
        state_id = _text(row.get("id"))
        if not state_id:
            continue
        state = by_id.get(state_id, {})
        state.update({
            "id": state_id,
            "name": _text(row.get("name")) or state_id,
            "cost": _number_or_expression(row.get("cost")),
            "effect": _number_or_expression(row.get("effect")),
            "initial_prob": _number_or_expression(row.get("initial_prob")),
            "is_absorbing": bool(row.get("is_absorbing")),
        })
        result.append(state)
    return result


def transitions_to_rows(transitions: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "from_state": str(t.get("from_state", "")),
            "to_state": str(t.get("to_state", "")),
            "probability": _text(t.get("probability", 0.0)) or "0",
            "cost_increment": _text(t.get("cost_increment", 0.0)) or "0",
            "effect_increment": _text(t.get("effect_increment", 0.0)) or "0",
        }
        for t in transitions
    ]


def rows_to_transitions(rows: Sequence[Mapping[str, Any]], existing: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Table rows -> transition dicts. Fields the table does not show (label,
    extra_increment...) are kept from an existing transition with the same
    from/to pair (taken in order, so two parallel transitions stay distinct)."""

    pool: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for t in existing:
        pool.setdefault((str(t.get("from_state")), str(t.get("to_state"))), []).append(dict(t))
    result: list[dict[str, Any]] = []
    for row in rows:
        source, target = _text(row.get("from_state")), _text(row.get("to_state"))
        if not source or not target:
            continue
        candidates = pool.get((source, target)) or []
        transition: dict[str, Any] = candidates.pop(0) if candidates else {}
        transition.update({
            "from_state": source,
            "to_state": target,
            "probability": _number_or_expression(row.get("probability")),
        })
        for key in ("cost_increment", "effect_increment"):
            value = _number_or_expression(row.get(key))
            if value not in (0.0, ""):
                transition[key] = value
            else:
                transition.pop(key, None)
        result.append(transition)
    return result


_ID_OK = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def table_problems(states: Sequence[Mapping[str, Any]], transitions: Sequence[Mapping[str, Any]]) -> list[str]:
    """Plain-language problems to show under the tables (the engine still
    validates fully when the model is run)."""

    problems: list[str] = []
    ids = [str(s["id"]) for s in states]
    if not ids:
        problems.append("Add at least one state.")
    for state_id in ids:
        if not _ID_OK.match(state_id):
            problems.append(f"State id '{state_id}' must start with a letter or underscore and use only letters, digits and underscores.")
    if len(ids) != len(set(ids)):
        problems.append("State ids must be unique.")
    for t in transitions:
        for end in ("from_state", "to_state"):
            if str(t[end]) not in ids:
                problems.append(f"Transition {t['from_state']} -> {t['to_state']}: unknown state '{t[end]}'.")
    return problems


def strategies_of(markov_document: Mapping[str, Any]) -> list[tuple[str, str]]:
    """[(strategy id, display name)] of a Markov model file."""

    return [(str(st.get("id")), str(st.get("name") or st.get("id"))) for st in markov_document.get("strategies", [])]


def import_markov_strategy(
    node: dict[str, Any],
    tree_document: dict[str, Any],
    markov_document: Mapping[str, Any],
    strategy_id: str,
    merge_inputs: bool = True,
) -> tuple[str | None, dict[str, list[str]]]:
    """Copy one strategy of a Markov model file into a tree Markov node:
    its states and transitions plus the model's cycles, cycles per year and
    half-cycle setting. With ``merge_inputs`` the file's parameters,
    variables, distributions and tables that the tree does not have yet
    (by name) are added to the tree document, because the copied formulas
    refer to them. Returns ``(error, report)``; the report lists what was
    ``added`` and which names already existed with a different value
    (``conflicts`` -- the tree's value is kept)."""

    import copy

    strategy = next((st for st in markov_document.get("strategies", []) if str(st.get("id")) == strategy_id), None)
    if strategy is None:
        return "That strategy is not in the Markov file.", {}
    if not strategy.get("states"):
        return "That strategy has no states.", {}
    node["states"] = copy.deepcopy(strategy["states"])
    node["transitions"] = copy.deepcopy(strategy.get("transitions", []))
    node["cycles"] = int(markov_document.get("cycles", node.get("cycles", 10)))
    node["cycles_per_year"] = float(markov_document.get("cycles_per_year", node.get("cycles_per_year", 1.0)) or 1.0)
    node["half_cycle_correction"] = bool(markov_document.get("half_cycle_correction", True))

    report: dict[str, list[str]] = {"added": [], "conflicts": []}
    if not merge_inputs:
        return None, report
    for key in ("parameters", "variables", "distributions", "tables"):
        existing = {str(r.get("name")): r for r in tree_document.setdefault(key, [])}
        for row in markov_document.get(key, []) or []:
            name = str(row.get("name", ""))
            if not name:
                continue
            if name in existing:
                if existing[name] != row:
                    report["conflicts"].append(f"{key[:-1]} {name}")
                continue
            if key in ("parameters", "variables") and name in {
                str(r.get("name")) for other in ("parameters", "variables") for r in tree_document.get(other, [])
            }:
                report["conflicts"].append(f"{key[:-1]} {name} (name already used by a parameter/variable)")
                continue
            tree_document[key].append(copy.deepcopy(row))
            existing[name] = row
            report["added"].append(f"{key[:-1]} {name}")
    return None, report
