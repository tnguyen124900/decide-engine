"""Reusable subtrees for the decision tree (TreeAge "subtree / clone").

A decision or chance node may be a *clone* of another decision/chance node
(the master) instead of owning its own branches:

    {"id": "after_test_b", "type": "chance", "label": "...",
     "clone_of": "after_test_a",
     "clone_substitutions": {"p_pos_a": "p_pos_b", "c_drug_a": "c_drug_b"}}

Before the document is parsed, ``expand_tree_clones()`` replaces every clone
by a fresh copy of the master's branches and descendants (copied node ids
get the suffix ``__<clone id>``). Editing the master therefore updates all
its clones on the next run. ``clone_substitutions`` renames names inside the
copy -- in every formula (probabilities, cost/effect increments, terminal
payoffs); ids and labels are left untouched. The probability, cost and
effect on the branch that *leads to* the clone belong to its parent, so they
stay independent of the master's.
"""

from __future__ import annotations

import copy
from typing import Any, Mapping

from .des_clone import _substitute, master_subtree_ids

TREE_CLONE_TYPES = ("decision", "chance")


def _is_tree_clone(node: Mapping[str, Any]) -> bool:
    return bool(node.get("clone_of")) and node.get("type") in TREE_CLONE_TYPES


def sync_clone_masters(document: Mapping[str, Any]) -> int:
    """Identify each clone's master by its automatic node number, not by its name.

    A clone stores ``clone_of`` (the master's current id, kept for readability and older files) and
    ``clone_of_num`` (the master's stable node number). When the master was renamed by any route, the number still
    finds it and ``clone_of`` is repointed; a clone without ``clone_of_num`` (older file) gets it from its id.
    Returns the number of clones whose reference changed."""

    from .node_numbers import ensure_node_numbers

    nodes = [n for n in document.get("nodes", []) if isinstance(n, dict)]
    if not any(_is_tree_clone(n) for n in nodes):
        return 0
    ensure_node_numbers(document)  # type: ignore[arg-type]
    by_num = {n.get("num"): n for n in nodes if n.get("num") is not None}
    by_id = {str(n.get("id")): n for n in nodes}
    changed = 0
    for node in nodes:
        if not _is_tree_clone(node):
            continue
        master = by_num.get(node.get("clone_of_num"))
        if master is not None and master is not node and master.get("type") in TREE_CLONE_TYPES:
            if str(master.get("id")) != str(node.get("clone_of")):
                node["clone_of"] = master["id"]
                changed += 1
            continue
        master = by_id.get(str(node.get("clone_of")))
        if master is not None and master.get("num") is not None and node.get("clone_of_num") != master["num"]:
            node["clone_of_num"] = master["num"]
            changed += 1
    return changed


def tree_clone_errors(document: Mapping[str, Any]) -> list[tuple[str, str]]:
    """[(clone node id, message)] for clones pointing at something other
    than a non-clone decision/chance node that does not contain the clone."""

    nodes = {str(n.get("id")): n for n in document.get("nodes", [])}
    errors: list[tuple[str, str]] = []
    for node in document.get("nodes", []):
        if not _is_tree_clone(node):
            continue
        clone_id, master_id = str(node.get("id")), str(node["clone_of"])
        master = nodes.get(master_id)
        if master_id == clone_id:
            errors.append((clone_id, "A node cannot be a clone of itself"))
        elif master is None or master.get("type") not in TREE_CLONE_TYPES:
            errors.append((clone_id, f"Clone of '{master_id}': that decision/chance node does not exist"))
        elif master.get("clone_of"):
            errors.append((clone_id, f"Clone of '{master_id}', which is itself a clone -- "
                                     "point it at the original (master) node"))
        elif clone_id in master_subtree_ids(nodes, master_id):
            errors.append((clone_id, f"'{clone_id}' sits inside the subtree of its own master '{master_id}'"))
    return errors


def _clone_copy(document: Mapping[str, Any], clone: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    nodes = {str(n.get("id")): n for n in document.get("nodes", [])}
    master_id, clone_id = str(clone["clone_of"]), str(clone["id"])
    substitutions = {str(k).strip(): str(v).strip() for k, v in (clone.get("clone_substitutions") or {}).items()
                     if str(k).strip() and str(v).strip()}
    members = master_subtree_ids(nodes, master_id)
    new_id = {old: f"{old}__{clone_id}" for old in members}

    def relink(node: dict[str, Any]) -> dict[str, Any]:
        for branch in node.get("branches", []) or []:
            if branch.get("child_id") in new_id:
                branch["child_id"] = new_id[branch["child_id"]]
        return node

    copies = []
    for old in members:
        node = _substitute(copy.deepcopy(nodes[old]), substitutions)
        node["id"] = new_id[old]
        copies.append(relink(node))
    expanded = _substitute(copy.deepcopy(nodes[master_id]), substitutions)
    expanded["id"] = clone_id
    expanded["type"] = nodes[master_id]["type"]
    if clone.get("label"):
        expanded["label"] = clone["label"]
    else:
        expanded.pop("label", None)
    expanded["cloned_from"] = master_id
    expanded.pop("clone_of", None)
    expanded.pop("clone_substitutions", None)
    return relink(expanded), copies


def expand_tree_clones(document: Mapping[str, Any]) -> Mapping[str, Any]:
    """Expand clones, including clones nested inside a master's subtree."""

    sync_clone_masters(document)
    for _ in range(8):
        expanded = _expand_once(document)
        if expanded is document:
            return document
        document = expanded
    raise ValueError("Tree clones are nested too deeply (or form a loop)")


def _expand_once(document: Mapping[str, Any]) -> Mapping[str, Any]:
    """The document with every decision/chance clone replaced by a copy of
    its master's subtree. Unchanged when it has no clones; an invalid clone
    raises ValueError."""

    raw_nodes = list(document.get("nodes", []) or [])
    if not any(_is_tree_clone(node) for node in raw_nodes):
        return document
    problems = tree_clone_errors(document)
    if problems:
        raise ValueError("; ".join(f"{node_id}: {message}" for node_id, message in problems))
    expanded_nodes: list[dict[str, Any]] = []
    extra: list[dict[str, Any]] = []
    for node in raw_nodes:
        if _is_tree_clone(node):
            clone_node, copies = _clone_copy(document, node)
            expanded_nodes.append(clone_node)
            extra.extend(copies)
        else:
            expanded_nodes.append(node)
    return {**document, "nodes": expanded_nodes + extra}


def parse_substitutions(text: str) -> dict[str, str]:
    """'old -> new' per line (also 'old=new' or 'old:new') -> dict."""

    result: dict[str, str] = {}
    for line in str(text or "").splitlines():
        for sep in ("->", "=>", "=", ":"):
            if sep in line:
                old, new = line.split(sep, 1)
                if old.strip() and new.strip():
                    result[old.strip()] = new.strip()
                break
    return result


def add_tree_clone(
    document: dict[str, Any], parent_id: str, master_id: str, clone_id: str,
    substitutions: Mapping[str, str] | None = None,
) -> str | None:
    """Add a new clone node of ``master_id`` as a new branch under
    ``parent_id`` (probability 0.0 when the parent is a chance node, so the
    incomplete-p highlight shows). Returns an error message or None."""

    clone_id = clone_id.strip()
    nodes = {str(n.get("id")): n for n in document.get("nodes", [])}
    parent, master = nodes.get(parent_id), nodes.get(master_id)
    if not clone_id:
        return "Name for the clone is required."
    if clone_id in nodes:
        return f"Node id '{clone_id}' already exists."
    if parent is None or parent.get("type") not in TREE_CLONE_TYPES:
        return "Attach the clone under a decision or chance node."
    if master is None or master.get("type") not in TREE_CLONE_TYPES or master.get("clone_of"):
        return "Master must be an existing decision/chance node that is not itself a clone."
    if parent_id == master_id or parent_id in master_subtree_ids(nodes, master_id):
        return "Cannot attach a clone inside its own master's subtree."
    document["nodes"].append({
        "id": clone_id, "type": master["type"], "label": f"clone of {master_id}",
        "clone_of": master_id, "clone_substitutions": dict(substitutions or {}), "branches": [],
    })
    sync_clone_masters(document)
    branch: dict[str, Any] = {"label": clone_id, "child_id": clone_id}
    if parent.get("type") == "chance":
        branch["probability"] = "0.0"
    parent.setdefault("branches", []).append(branch)
    return None


def make_independent_copy(document: dict[str, Any], clone_id: str) -> str | None:
    """Replace one clone by a real copy of its master's current subtree."""

    nodes = [n for n in document.get("nodes", [])]
    index = next((i for i, n in enumerate(nodes) if str(n.get("id")) == clone_id), None)
    if index is None or not _is_tree_clone(nodes[index]):
        return "That node is not a clone."
    problems = [m for i, m in tree_clone_errors(document) if i == clone_id]
    if problems:
        return problems[0]
    expanded, copies = _clone_copy(document, nodes[index])
    expanded.pop("cloned_from", None)
    nodes[index] = expanded
    document["nodes"] = nodes + copies
    return None


def rename_master_references(document: dict[str, Any], old_id: str, new_id: str) -> int:
    """After a node id changes, repoint every clone that mirrored it.
    Returns the number of clones updated."""

    if not old_id or old_id == new_id:
        return 0
    count = 0
    for node in document.get("nodes", []):
        if _is_tree_clone(node) and str(node.get("clone_of")) == old_id:
            node["clone_of"] = new_id
            count += 1
    return count


def detach_clones_of(document: dict[str, Any], doomed_ids: set[str]) -> list[str]:
    """Before nodes in ``doomed_ids`` are deleted: turn every surviving clone
    whose master is among them into an independent copy of the master's
    current subtree, so deleting a master never leaves a dangling clone.
    Clones that are themselves being deleted are skipped. Repeats because a
    detached copy can contain further clones of a doomed master. Returns the
    ids of the clones that were detached."""

    detached: list[str] = []
    for _ in range(8):
        pending = [
            str(n["id"]) for n in document.get("nodes", [])
            if _is_tree_clone(n) and str(n["id"]) not in doomed_ids and str(n.get("clone_of")) in doomed_ids
        ]
        if not pending:
            break
        for clone_id in pending:
            if make_independent_copy(document, clone_id) is None:
                detached.append(clone_id)
            else:
                # Master already invalid (missing / itself a clone): the clone cannot be
                # expanded, so drop the link and leave an empty node the user can fix.
                for n in document["nodes"]:
                    if str(n.get("id")) == clone_id:
                        n.pop("clone_of", None)
                        n.pop("clone_substitutions", None)
                detached.append(clone_id)
    return detached
