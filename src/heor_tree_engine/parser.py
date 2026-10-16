"""Build linked decision trees from the flat JSON representation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .expressions import (
    build_tables_mapping,
    resolve_parameter_definitions,
    resolve_variables,
    resolve_variables_partial,
)
from .models import (
    Branch,
    ChanceNode,
    DecisionNode,
    DESTreeNode,
    MarkovTreeNode,
    Node,
    Param,
    PSMNode,
    SurvivalCurve,
    TerminalNode,
)


def _numeric_ref(value: Any) -> float | str:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError("Numeric references must be numbers or parameter names")
    return float(value) if isinstance(value, (int, float)) else value


def _parse_extra_outcomes(raw: Mapping[str, Any]) -> dict[str, float | str]:
    """Parse the optional "extra_outcomes" dict on a node or branch --
    additional Cost/Effect outcomes beyond the primary pair, keyed by
    outcome id (see document["outcomes"] / Branch.extra_outcomes /
    TerminalNode.extra_outcomes)."""

    raw_extra = raw.get("extra_outcomes")
    if not raw_extra:
        return {}
    return {str(key): _numeric_ref(value) for key, value in raw_extra.items()}


def _parse_param(raw: Mapping[str, Any]) -> Param:
    if "name" not in raw or "base" not in raw:
        raise ValueError("A parameter requires 'name' and 'base'")
    base = raw["base"]
    if isinstance(base, bool):
        raise ValueError(f"Parameter '{raw['name']}' base must be numeric or an expression")
    try:
        base_value = float(base)
    except (TypeError, ValueError):
        base_value = 0.0
    return Param(
        name=str(raw["name"]),
        base=base_value,
        low=float(raw["low"]) if raw.get("low") is not None else None,
        high=float(raw["high"]) if raw.get("high") is not None else None,
        dist=str(raw["dist"]) if raw.get("dist") is not None else None,
        dist_args=(
            {str(key): float(value) for key, value in raw["dist_args"].items()}
            if raw.get("dist_args") is not None
            else None
        ),
        source=str(raw["source"]) if raw.get("source") is not None else None,
    )


def _parse_branch(raw: Mapping[str, Any]) -> Branch:
    if "label" not in raw or "child_id" not in raw:
        raise ValueError("A branch requires 'label' and 'child_id'")
    return Branch(
        label=str(raw["label"]),
        child_id=str(raw["child_id"]),
        probability=(
            _numeric_ref(raw["probability"])
            if raw.get("probability") is not None
            else None
        ),
        cost_increment=(
            _numeric_ref(raw["cost_increment"])
            if raw.get("cost_increment") is not None
            else None
        ),
        effect_increment=(
            _numeric_ref(raw["effect_increment"])
            if raw.get("effect_increment") is not None
            else None
        ),
        extra_outcomes=_parse_extra_outcomes(raw),
        role=str(raw["role"]) if raw.get("role") is not None else None,
    )


def _parse_curve(raw: Mapping[str, Any]) -> SurvivalCurve:
    if "distribution" not in raw:
        raise ValueError("A survival curve requires a 'distribution'")
    return SurvivalCurve(
        distribution=str(raw["distribution"]),
        params={str(key): _numeric_ref(value) for key, value in raw.get("params", {}).items()},
        # "year" if absent -- every PSM model saved before this field
        # existed keeps its original (years-only) behaviour unchanged.
        time_unit=str(raw["time_unit"]) if raw.get("time_unit") else "year",
    )


def _parse_node(raw: Mapping[str, Any]) -> Node:
    if "id" not in raw or "type" not in raw:
        raise ValueError("A node requires 'id' and 'type'")
    node_id = str(raw["id"])
    node_type = raw["type"]
    if node_type == "terminal":
        return TerminalNode(
            id=node_id,
            cost=_numeric_ref(raw.get("cost", 0.0)),
            effect=_numeric_ref(raw.get("effect", 0.0)),
            year=_numeric_ref(raw.get("year", 0.0)),
            extra_outcomes=_parse_extra_outcomes(raw),
        )
    if node_type == "chance":
        return ChanceNode(
            id=node_id,
            branches=[_parse_branch(branch) for branch in raw.get("branches", [])],
        )
    if node_type == "decision":
        return DecisionNode(
            id=node_id,
            branches=[_parse_branch(branch) for branch in raw.get("branches", [])],
        )
    if node_type == "psm":
        if "pfs_curve" not in raw or "os_curve" not in raw:
            raise ValueError(f"PSM node '{node_id}' requires 'pfs_curve' and 'os_curve'")
        return PSMNode(
            id=node_id,
            pfs_curve=_parse_curve(raw["pfs_curve"]),
            os_curve=_parse_curve(raw["os_curve"]),
            cycles=int(raw.get("cycles", 10)),
            cycles_per_year=float(raw.get("cycles_per_year", 1.0)),
            half_cycle_correction=bool(raw.get("half_cycle_correction", True)),
            branches=[_parse_branch(branch) for branch in raw.get("branches", [])],
        )
    if node_type == "markov":
        from .markov_parser import resolve_markov

        if not raw.get("states"):
            raise ValueError(f"Markov node '{node_id}' requires 'states'")
        return MarkovTreeNode(
            id=node_id,
            model=resolve_markov({**raw, "id": node_id, "name": raw.get("label", node_id)}),
            cycles_per_year=float(raw.get("cycles_per_year", 1.0) or 1.0),
        )
    raise ValueError(f"Unknown node type '{node_type}'")


def _children(node: Node) -> list[Node]:
    if not isinstance(node, (DecisionNode, ChanceNode, PSMNode)):
        return []
    return [branch.child for branch in node.branches if branch.child is not None]


def _check_cycles_and_orphans(root: Node, nodes: dict[str, Node]) -> None:
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: Node) -> None:
        if node.id in visiting:
            raise ValueError(f"Cycle detected at node '{node.id}'")
        if node.id in visited:
            return
        visiting.add(node.id)
        for child in _children(node):
            visit(child)
        visiting.remove(node.id)
        visited.add(node.id)

    visit(root)
    orphaned = set(nodes) - visited
    if orphaned:
        orphan_list = ", ".join(sorted(orphaned))
        raise ValueError(f"Orphan node(s): {orphan_list}")


def resolve(document: Mapping[str, Any]) -> tuple[Node, dict[str, Param]]:
    """Resolve a flat JSON document into a linked root tree and parameters."""

    if "root_id" not in document or "nodes" not in document:
        raise ValueError("Document requires 'root_id' and 'nodes'")
    # DES clones (TreeAge clone copies) -> copies of their master's subtree.
    from .des_clone import expand_des_clones
    from .tree_clone import expand_tree_clones

    document = expand_tree_clones(document)
    document = expand_des_clones(document)
    # DES nodes: each one owns a subtree of
    # Time/Chance/Terminal nodes parsed by des_parser, so those ids are
    # skipped here instead of becoming ordinary Tree nodes.
    des_models: dict[str, Any] = {}
    des_members: set[str] = set()
    raw_des_ids = [str(raw["id"]) for raw in document["nodes"] if raw.get("type") == "des"]
    if raw_des_ids:
        from .des_parser import resolve_des

        for des_id in raw_des_ids:
            des_model = resolve_des(document, des_id)
            des_models[des_id] = des_model
            des_members.update(node_id for node_id in des_model.nodes if node_id != des_id)

    nodes: dict[str, Node] = {}
    for raw_node in document["nodes"]:
        raw_id = str(raw_node.get("id"))
        if raw_id in des_members:
            continue
        if raw_id in des_models:
            if raw_id in nodes:
                raise ValueError(f"Duplicate node id '{raw_id}'")
            nodes[raw_id] = DESTreeNode(id=raw_id, model=des_models[raw_id])
            continue
        node = _parse_node(raw_node)
        if node.id in nodes:
            raise ValueError(f"Duplicate node id '{node.id}'")
        nodes[node.id] = node

    for node in nodes.values():
        if not isinstance(node, (DecisionNode, ChanceNode, PSMNode)):
            continue
        for branch in node.branches:
            child = nodes.get(branch.child_id)
            if child is None:
                raise ValueError(
                    f"Branch '{branch.label}' references missing node '{branch.child_id}'"
                )
            branch.child = child

    root = nodes.get(str(document["root_id"]))
    if root is None:
        raise ValueError(f"Root node '{document['root_id']}' does not exist")
    _check_cycles_and_orphans(root, nodes)

    parameters: dict[str, Param] = {}
    for raw_param in document.get("parameters", []):
        parameter = _parse_param(raw_param)
        if parameter.name in parameters:
            raise ValueError(f"Duplicate parameter name '{parameter.name}'")
        parameters[parameter.name] = parameter
    parameter_definitions: dict[str, Any] = {}
    for raw_param in document.get("parameters", []):
        parameter_name = str(raw_param["name"])
        definition = raw_param.get("definition")
        if definition is None and isinstance(raw_param.get("base"), str):
            try:
                float(raw_param["base"])
            except ValueError:
                definition = raw_param["base"]
        if definition is not None:
            parameter_definitions[parameter_name] = definition

    distributions = {
        str(raw["name"]): raw
        for raw in document.get("distributions", [])
        if raw.get("name")
    }
    tables = build_tables_mapping(document.get("tables"))

    try:
        parameters = resolve_parameter_definitions(parameters, parameter_definitions, distributions, tables)
    except ValueError as exc:
        raise ValueError(f"Invalid parameter definition: {exc}") from exc

    variables: dict[str, Any] = {}
    for raw_variable in document.get("variables", []):
        if "name" not in raw_variable:
            raise ValueError("A variable requires 'name'")
        variable_name = str(raw_variable["name"])
        if "definition" in raw_variable:
            variables[variable_name] = raw_variable["definition"]
        elif "base" in raw_variable:
            variables[variable_name] = raw_variable["base"]
        elif "value" in raw_variable:
            variables[variable_name] = raw_variable["value"]
        else:
            raise ValueError(f"Variable '{variable_name}' requires 'definition'")

    markov_nodes = [node for node in nodes.values() if isinstance(node, MarkovTreeNode)]
    if des_models or markov_nodes:
        # A variable may depend on per-patient DES values (e.g. "age =
        # startAge + _time"), which only exist inside the simulation: keep
        # the ones that resolve now as usual, and hand every raw formula to
        # the DES nodes so they are evaluated per patient.
        parameters, _unresolved = resolve_variables_partial(parameters, variables, distributions, tables)
        for node in nodes.values():
            if isinstance(node, (DESTreeNode, MarkovTreeNode)):
                node.variables = dict(variables)
        return root, parameters
    try:
        parameters = resolve_variables(parameters, variables, distributions, tables)
    except ValueError as exc:
        raise ValueError(f"Invalid variable definition: {exc}") from exc
    return root, parameters