"""Build DES models from the flat JSON document (same ``nodes`` list shape
as a Decision-Tree document, with two new node types: "des" and "time").

Example (one strategy)::

    {"root_id": "root", "nodes": [
      {"id": "root", "type": "decision",
       "branches": [{"label": "Standard care", "child_id": "des_a"}]},
      {"id": "des_a", "type": "des", "time_horizon": 40, "n_patients": 1000,
       "random_seed": 1, "trackers": [{"name": "startAge", "initial_value": 60}],
       "branches": [{"label": "Mild", "child_id": "mild", "probability": 1}]},
      {"id": "mild", "type": "time", "name": "Mild", "cost_rate": 1000,
       "effect_rate": 0.8,
       "branches": [
         {"label": "Progression", "child_id": "to_severe",
          "time_to_event": {"distribution": "weibull",
                            "params": {"shape": 1.3, "scale": 6}}},
         {"label": "Background death", "child_id": "dead",
          "time_to_event": {"distribution": "table_to_time",
                            "params": {"table": "life", "index_column": "age",
                                       "value_column": "qx"},
                            "clock": "forward", "time_offset": "startAge",
                            "sample_once": true}}]},
      {"id": "to_severe", "type": "terminal", "jump_to": "severe"},
      {"id": "dead", "type": "terminal"}
    ]}
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .des_clone import expand_des_clones
from .des_models import (
    DESBranch,
    DESChance,
    DESLogic,
    DESModel,
    DESNode,
    DESTerminal,
    TimeNode,
    TimeToEvent,
)
from .expressions import build_tables_mapping, resolve_parameter_definitions
from .microsim_parser import _parse_tracker
from .models import Param
from .parser import _numeric_ref, _parse_param


def _optional_ref(raw: Mapping[str, Any], key: str) -> Any:
    value = raw.get(key)
    return _numeric_ref(value) if value not in (None, "") else None


def _extra_map(raw: Any) -> dict[str, Any]:
    return {str(key): _numeric_ref(value) for key, value in (raw or {}).items() if value not in (None, "")}


def _parse_tte(raw: Mapping[str, Any]) -> TimeToEvent:
    if "distribution" not in raw:
        raise ValueError("A time-to-event requires a 'distribution'")
    params: dict[str, Any] = {}
    for key, value in (raw.get("params") or {}).items():
        # table_to_time's table/column names are plain strings, not expressions.
        params[str(key)] = value if isinstance(value, str) else _numeric_ref(value)
    sample_once = bool(raw.get("sample_once", False))
    distribution = str(raw["distribution"])
    if distribution == "table_prob":
        distribution = "table_to_time"  # name used before the TableToTime rename
    return TimeToEvent(
        distribution=distribution,
        params=params,
        clock="forward" if sample_once else str(raw.get("clock", "reset")),
        time_offset=_numeric_ref(raw.get("time_offset", 0.0)),
        sample_once=sample_once,
        prob_type=str(raw.get("prob_type", "conditional")),
        interpolate=bool(raw.get("interpolate", True)),
    )


def _parse_des_branch(raw: Mapping[str, Any]) -> DESBranch:
    if "label" not in raw or "child_id" not in raw:
        raise ValueError("A branch requires 'label' and 'child_id'")
    return DESBranch(
        label=str(raw["label"]),
        child_id=str(raw["child_id"]),
        probability=_optional_ref(raw, "probability"),
        time_to_event=_parse_tte(raw["time_to_event"]) if raw.get("time_to_event") else None,
        cost_increment=_optional_ref(raw, "cost_increment"),
        effect_increment=_optional_ref(raw, "effect_increment"),
        tracker_updates={
            str(name): _numeric_ref(value) for name, value in (raw.get("tracker_updates") or {}).items()
        },
        extra_outcomes=_extra_map(raw.get("extra_outcomes")),
        condition=_optional_ref(raw, "condition"),
    )


def _tracker_map(raw: Any) -> dict[str, Any]:
    return {str(name): _numeric_ref(value) for name, value in (raw or {}).items() if value not in (None, "")}


def _parse_des_node(raw: Mapping[str, Any], defaults: Mapping[str, Any]):
    node_id = str(raw["id"])
    node_type = raw.get("type")
    branches = [_parse_des_branch(branch) for branch in raw.get("branches", [])]
    if node_type == "des":
        def setting(key: str, fallback: Any) -> Any:
            return raw.get(key, defaults.get(key, fallback))

        return DESNode(
            id=node_id,
            name=str(raw.get("name", "DES")),
            branches=branches,
            time_horizon=_numeric_ref(setting("time_horizon", 40.0)),
            n_patients=int(setting("n_patients", 1000)),
            random_seed=int(setting("random_seed", 1)),
            trackers=[_parse_tracker(item) for item in (raw.get("trackers") or defaults.get("trackers") or [])],
            event_log_sample_size=int(setting("event_log_sample_size", 0)),
            trace_step=float(setting("trace_step", 1.0)),
            max_events_per_patient=int(setting("max_events_per_patient", 10_000)),
            termination_condition=_termination_condition(setting("termination_condition", None)),
        )
    if node_type == "time":
        return TimeNode(
            id=node_id,
            name=str(raw.get("name") or raw.get("label") or node_id),
            branches=branches,
            cost_rate=_numeric_ref(raw.get("cost_rate", 0.0)),
            effect_rate=_numeric_ref(raw.get("effect_rate", 0.0)),
            startup_cost=_numeric_ref(raw.get("startup_cost", 0.0)),
            startup_effect=_numeric_ref(raw.get("startup_effect", 0.0)),
            extra_rates=_extra_map(raw.get("extra_rates")),
            extra_startup=_extra_map(raw.get("extra_startup")),
            tracker_updates=_tracker_map(raw.get("tracker_updates")),
        )
    if node_type == "chance":
        return DESChance(id=node_id, branches=branches)
    if node_type == "logic":
        return DESLogic(id=node_id, branches=branches)
    if node_type == "terminal":
        jump_to = raw.get("jump_to")
        return DESTerminal(
            id=node_id,
            name=str(raw.get("name") or raw.get("label") or node_id),
            cost=_numeric_ref(raw.get("cost", 0.0)),
            effect=_numeric_ref(raw.get("effect", 0.0)),
            jump_to=str(jump_to) if jump_to not in (None, "") else None,
            extra_outcomes=_extra_map(raw.get("extra_outcomes")),
            tracker_updates=_tracker_map(raw.get("tracker_updates")),
        )
    raise ValueError(f"Node type '{node_type}' is not allowed inside a DES model (node '{node_id}')")


def _termination_condition(value: Any) -> Any:
    """Blank / None -> no termination condition; otherwise the formula."""

    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    return _numeric_ref(value)


def resolve_des(document: Mapping[str, Any], des_node_id: str) -> DESModel:
    """Parse the DES subtree rooted at ``des_node_id``. Only nodes reachable
    from it through ``child_id`` links are parsed (``jump_to`` targets must
    be among them -- checked by validate_des)."""

    document = expand_des_clones(document)
    raw_nodes = {str(raw["id"]): raw for raw in document.get("nodes", [])}
    if des_node_id not in raw_nodes or raw_nodes[des_node_id].get("type") != "des":
        raise ValueError(f"'{des_node_id}' is not a DES node")
    defaults = document.get("des_settings") or {}
    nodes: dict[str, Any] = {}
    stack = [des_node_id]
    while stack:
        node_id = stack.pop()
        if node_id in nodes:
            continue
        if node_id not in raw_nodes:
            raise ValueError(f"Branch references missing node '{node_id}'")
        node = _parse_des_node(raw_nodes[node_id], defaults)
        if isinstance(node, DESNode) and node_id != des_node_id:
            raise ValueError(f"A DES node cannot be nested inside another DES model ('{node_id}')")
        nodes[node_id] = node
        stack.extend(branch.child_id for branch in getattr(node, "branches", []))
        if isinstance(node, DESTerminal) and node.jump_to:
            # A state reached only by "jump to" (e.g. Severe) is still part
            # of this model even if no child_id link points at it.
            stack.append(node.jump_to)
    outcome_symbols = {
        str(row.get("id")): str(row.get("symbol") or "")
        for row in (document.get("outcomes") or [])
        if isinstance(row, Mapping) and row.get("type") in ("cost", "effect") and row.get("id")
    }
    return DESModel(des=nodes[des_node_id], nodes=nodes, outcome_symbols=outcome_symbols)


def resolve_des_strategies(document: Mapping[str, Any]) -> dict[str, DESModel]:
    """Decision root -> one DES model per strategy branch, keyed by the
    branch label (the node chain Decision -> DES -> Time -> Chance ->
    Terminal). A root that is itself a DES node yields one strategy."""

    document = expand_des_clones(document)
    raw_nodes = {str(raw["id"]): raw for raw in document.get("nodes", [])}
    root_id = str(document.get("root_id", ""))
    root = raw_nodes.get(root_id)
    if root is None:
        raise ValueError(f"Root node '{root_id}' does not exist")
    if root.get("type") == "des":
        return {str(root.get("name", root_id)): resolve_des(document, root_id)}
    if root.get("type") != "decision":
        raise ValueError("A DES document's root must be a Decision node or a DES node")
    strategies: dict[str, DESModel] = {}
    for branch in root.get("branches", []):
        label = str(branch["label"])
        if label in strategies:
            raise ValueError(f"Duplicate strategy label '{label}'")
        strategies[label] = resolve_des(document, str(branch["child_id"]))
    return strategies


def resolve_des_namespace(
    document: Mapping[str, Any],
) -> tuple[dict[str, Param], dict[str, Any], dict[str, Any], dict[str, dict[str, Any]]]:
    """(parameters, variables, distributions, tables) for a DES document.
    Parameter definitions are resolved like parser.resolve() does for a Tree."""

    parameters: dict[str, Param] = {}
    definitions: dict[str, Any] = {}
    for raw in document.get("parameters", []):
        parameter = _parse_param(raw)
        if parameter.name in parameters:
            raise ValueError(f"Duplicate parameter name '{parameter.name}'")
        parameters[parameter.name] = parameter
        definition = raw.get("definition")
        if definition is None and isinstance(raw.get("base"), str):
            try:
                float(raw["base"])
            except ValueError:
                definition = raw["base"]
        if definition is not None:
            definitions[parameter.name] = definition
    distributions = {str(raw["name"]): raw for raw in document.get("distributions", []) if raw.get("name")}
    tables = build_tables_mapping(document.get("tables"))
    parameters = resolve_parameter_definitions(parameters, definitions, distributions, tables)
    variables: dict[str, Any] = {}
    for raw in document.get("variables", []):
        name = str(raw["name"])
        for key in ("definition", "base", "value"):
            if key in raw:
                variables[name] = raw[key]
                break
    # Variables are NOT resolved eagerly (unlike a Tree): in a DES they may
    # depend on per-patient values (trackers, _time, ...), e.g.
    # "age = startAge + _time". The engine passes them to every expression
    # alongside the patient context and the evaluator resolves them lazily.
    return parameters, variables, distributions, tables
