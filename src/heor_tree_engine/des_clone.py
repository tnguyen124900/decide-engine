"""Clones of a DES model (TreeAge "clone master / clone copy").

A DES node may be a *clone* of another DES node (the master) instead of
owning its own Time/Chance/Logic/Terminal subtree:

    {"id": "new_des", "type": "des", "clone_of": "soc_des",
     "clone_substitutions": {"prog_soc": "prog_new", "c_drug": "c_drug_new"}}

Before anything is parsed, `expand_des_clones()` replaces every clone by a
fresh copy of the master's subtree (node ids suffixed ``__<clone id>`` so
they stay unique), so every edit to the master reaches all its clones on
the next run. `clone_substitutions` renames names inside the copy -- in
every formula and in distribution / table names (e.g. use the
Distribution ``prog_new`` where the master uses ``prog_soc``); labels and
node names are left untouched. The clone keeps only its own name; every
DES setting (horizon, N patients, seed, trackers, termination condition...)
comes from the master, so one change updates all clones.
"""

from __future__ import annotations

import copy
import re
from typing import Any, Mapping

# Keys that hold identifiers/labels, never formulas -- not substituted
# (except inside a time-to-event "params" dict, where "name" is the
# Distribution name).
_STRUCTURAL_KEYS = {"id", "type", "label", "name", "child_id", "jump_to", "clone_of", "clone_substitutions"}
# What a clone keeps of its own; everything else comes from the master.
CLONE_OWN_SETTINGS = ("name", "label")


def _substitute_text(text: str, substitutions: Mapping[str, str]) -> str:
    for old, new in substitutions.items():
        text = re.sub(rf"(?<![A-Za-z0-9_]){re.escape(old)}(?![A-Za-z0-9_])", new, text)
    return text


def _substitute(value: Any, substitutions: Mapping[str, str], key: str | None = None,
                parent_key: str | None = None) -> Any:
    if not substitutions:
        return value
    if isinstance(value, dict):
        return {k: _substitute(v, substitutions, k, key) for k, v in value.items()}
    if isinstance(value, list):
        return [_substitute(v, substitutions, None, key) for v in value]
    if isinstance(value, str):
        if key in _STRUCTURAL_KEYS and parent_key != "params":
            return value
        return _substitute_text(value, substitutions)
    return value


def master_subtree_ids(nodes: Mapping[str, Mapping[str, Any]], master_id: str) -> list[str]:
    """Every node reachable from the master DES node (child_id + jump_to),
    master excluded, in a stable order."""

    seen: list[str] = []
    stack = [str(b.get("child_id")) for b in reversed(nodes[master_id].get("branches", []) or [])]
    while stack:
        node_id = stack.pop()
        if node_id in seen or node_id not in nodes:
            continue
        seen.append(node_id)
        node = nodes[node_id]
        if node.get("jump_to"):
            stack.append(str(node["jump_to"]))
        stack.extend(str(b.get("child_id")) for b in reversed(node.get("branches", []) or []))
    return seen


def clone_errors(document: Mapping[str, Any]) -> list[tuple[str, str]]:
    """[(clone node id, message)] for clones pointing at something that is
    not a non-clone DES node."""

    nodes = {str(n.get("id")): n for n in document.get("nodes", [])}
    errors = []
    for node in document.get("nodes", []):
        master_id = node.get("clone_of")
        if not master_id or node.get("type") != "des":
            continue
        master = nodes.get(str(master_id))
        if master is None or master.get("type") != "des":
            errors.append((str(node.get("id")), f"Clone of '{master_id}': that DES node does not exist"))
        elif master.get("clone_of"):
            errors.append((str(node.get("id")), f"Clone of '{master_id}', which is itself a clone -- "
                                                "point it at the original (master) DES node"))
        elif str(master_id) == str(node.get("id")):
            errors.append((str(node.get("id")), "A DES node cannot be a clone of itself"))
    return errors


def clone_copy_nodes(document: Mapping[str, Any], clone: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """(expanded clone DES node, copied subtree nodes) for one clone."""

    nodes = {str(n.get("id")): n for n in document.get("nodes", [])}
    master_id = str(clone["clone_of"])
    master = nodes[master_id]
    clone_id = str(clone["id"])
    substitutions = {str(k).strip(): str(v).strip() for k, v in (clone.get("clone_substitutions") or {}).items()
                     if str(k).strip() and str(v).strip()}
    member_ids = master_subtree_ids(nodes, master_id)
    new_id = {old: f"{old}__{clone_id}" for old in member_ids}

    def relink(node: dict[str, Any]) -> dict[str, Any]:
        for branch in node.get("branches", []) or []:
            if branch.get("child_id") in new_id:
                branch["child_id"] = new_id[branch["child_id"]]
        if node.get("jump_to") in new_id:
            node["jump_to"] = new_id[node["jump_to"]]
        return node

    copies = []
    for old in member_ids:
        node = _substitute(copy.deepcopy(nodes[old]), substitutions)
        node["id"] = new_id[old]
        copies.append(relink(node))
    expanded = _substitute(copy.deepcopy(master), substitutions)
    for key in CLONE_OWN_SETTINGS:
        if key in clone and clone[key] not in (None, ""):
            expanded[key] = copy.deepcopy(clone[key])
        elif key in ("name", "label"):
            expanded.pop(key, None)
    expanded["id"] = clone_id
    expanded["cloned_from"] = master_id
    return relink(expanded), copies


def expand_des_clones(document: Mapping[str, Any]) -> Mapping[str, Any]:
    """The document with every DES clone replaced by a copy of its master's
    subtree (see module docstring). Returned unchanged when it has no
    clones; an invalid clone (see clone_errors) raises ValueError."""

    raw_nodes = list(document.get("nodes", []) or [])
    if not any(node.get("clone_of") and node.get("type") == "des" for node in raw_nodes):
        return document
    problems = clone_errors(document)
    if problems:
        raise ValueError("; ".join(f"{node_id}: {message}" for node_id, message in problems))
    bad: set[str] = set()
    expanded_nodes: list[dict[str, Any]] = []
    extra: list[dict[str, Any]] = []
    for node in raw_nodes:
        if node.get("clone_of") and node.get("type") == "des" and str(node.get("id")) not in bad:
            clone_node, copies = clone_copy_nodes(document, node)
            expanded_nodes.append(clone_node)
            extra.extend(copies)
        else:
            expanded_nodes.append(node)
    return {**document, "nodes": expanded_nodes + extra}
