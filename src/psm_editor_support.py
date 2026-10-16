"""Pure helpers for editing PSM canvas documents."""

from __future__ import annotations

from typing import Any


def rename_psm_node(document: dict[str, Any], old_id: str, name: str) -> str | None:
    """Set a node's display name while keeping its document ID unique."""

    new_name = name.strip()
    nodes = {node["id"]: node for node in document.get("nodes", [])}
    node = nodes.get(old_id)
    if node is None or not new_name:
        return None

    if new_name == (node.get("label") or old_id):
        return old_id

    new_id = new_name
    if new_id != old_id and new_id in nodes:
        suffix = 2
        while f"{new_name}_{suffix}" in nodes:
            suffix += 1
        new_id = f"{new_name}_{suffix}"

    node["id"] = new_id
    node["label"] = new_name
    if document.get("root_id") == old_id:
        document["root_id"] = new_id
    for other in document.get("nodes", []):
        for branch in other.get("branches", []):
            if branch.get("child_id") == old_id:
                branch["child_id"] = new_id
    return new_id