"""Stable automatic node numbers for decision-tree documents.

Node names/labels and the internal `id` change when a node is renamed (or are repeated, e.g. "Test" under every
strategy), so they cannot identify a node to a person.  Every node therefore also carries a plain number
`node["num"]` = 1, 2, 3, ... that is assigned automatically, never changes when the node is renamed or moved, and
is never reused after the node is deleted (`document["next_node_num"]` only grows).  The canvas can show the number
above each node (hidden by default).

`ensure_node_numbers(document)` is idempotent and is called wherever the document is displayed or saved:
  * a document without numbers (an older file) is numbered in canvas order: depth-first from the root, branches in order;
  * a node added later (by the editor, Bayes revision, Flip, paste, clone...) simply has no number yet and gets the
    next free one;
  * a copied node that still carries its source's number (a duplicate) keeps the number only if it comes first in
    `document["nodes"]`; the later one is renumbered.
"""

from __future__ import annotations

from typing import Any


def _walk_order(document: dict[str, Any]) -> list[dict[str, Any]]:
    """Nodes in canvas order (depth-first from the root, branches in order), then any unreachable nodes in list order."""

    nodes = document.get("nodes", [])
    by_id = {n.get("id"): n for n in nodes if isinstance(n, dict)}
    seen: set[Any] = set()
    ordered: list[dict[str, Any]] = []
    stack = [document.get("root_id")]
    while stack:
        node_id = stack.pop()
        node = by_id.get(node_id)
        if node is None or node_id in seen:
            continue
        seen.add(node_id)
        ordered.append(node)
        children = [b.get("child_id") for b in node.get("branches", []) or [] if isinstance(b, dict)]
        stack.extend(reversed(children))
    ordered.extend(n for n in nodes if isinstance(n, dict) and n.get("id") not in seen)
    return ordered


def _valid(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


def ensure_node_numbers(document: dict[str, Any]) -> int:
    """Give every node a unique positive integer `num`; returns how many nodes were (re)numbered."""

    nodes = [n for n in document.get("nodes", []) if isinstance(n, dict)]
    if not nodes:
        return 0
    taken: set[int] = set()
    keep: set[int] = set()  # id() of nodes that keep their current number
    for node in nodes:  # list order decides who keeps a duplicated number
        num = node.get("num")
        if _valid(num) and num not in taken:
            taken.add(num)
            keep.add(id(node))
    counter = max([int(document.get("next_node_num") or 1), max(taken, default=0) + 1])
    assigned = 0
    for node in _walk_order(document):
        if id(node) in keep:
            continue
        node["num"] = counter
        counter += 1
        assigned += 1
    if document.get("next_node_num") != counter:
        document["next_node_num"] = counter
    return assigned


def node_number_text(node: dict[str, Any]) -> str:
    """"#12" for a numbered node, "" otherwise."""

    num = node.get("num")
    return f"#{num}" if _valid(num) else ""


def node_by_number(document: dict[str, Any], number: int) -> dict[str, Any] | None:
    for node in document.get("nodes", []):
        if isinstance(node, dict) and node.get("num") == number:
            return node
    return None


def ensure_state_numbers(strategy: dict[str, Any]) -> int:
    """Same idea as ensure_node_numbers for the states of a Markov / Microsimulation / ABM strategy:
    every state gets a stable automatic `num` (1, 2, 3 ... in list order the first time), kept when the state is
    renamed or moved and never reused after deletion (`strategy["next_state_num"]` only grows)."""

    states = [s for s in strategy.get("states", []) if isinstance(s, dict)]
    if not states:
        return 0
    taken: set[int] = set()
    keep: set[int] = set()
    for state in states:
        num = state.get("num")
        if _valid(num) and num not in taken:
            taken.add(num)
            keep.add(id(state))
    counter = max([int(strategy.get("next_state_num") or 1), max(taken, default=0) + 1])
    changed = 0
    for state in states:
        if id(state) not in keep:
            state["num"] = counter
            counter += 1
            changed += 1
    strategy["next_state_num"] = counter
    return changed


def state_number_label(strategy: dict[str, Any], state: dict[str, Any]) -> str:
    """'#7' for the Edit-state headers (the name is not shown: names repeat and change)."""

    ensure_state_numbers(strategy)
    return f"#{state.get('num', '?')}"
