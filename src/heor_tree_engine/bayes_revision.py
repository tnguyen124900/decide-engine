"""Bayes revision helper for the decision tree (binary disease / binary test).

Given a prior prevalence P(D), sensitivity Se = P(T+|D) and specificity
Sp = P(T-|~D), the tree needs the *revised* probabilities in the order
test result -> disease status:

    P(T+)      = Se*P + (1-Sp)*(1-P)
    P(D|T+)    = Se*P / P(T+)               (positive predictive value)
    P(D|T-)    = (1-Se)*P / (1-P(T+))       (1 - negative predictive value)

``add_binary_bayes_structure`` writes those as *derived variables*
(definition expressions) plus a two-stage chance subtree, so Run, SA, PSA
and VOI keep working: Se/Sp/prior can carry distributions and the posteriors
are recomputed from every draw.
"""
from __future__ import annotations

from typing import Any


def posterior_binary(prior: float, sensitivity: float, specificity: float) -> dict[str, float]:
    for name, value in (("prior", prior), ("sensitivity", sensitivity), ("specificity", specificity)):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1 (got {value}).")
    p_pos = sensitivity * prior + (1.0 - specificity) * (1.0 - prior)
    if p_pos <= 0.0 or p_pos >= 1.0:
        raise ValueError("P(test positive) is 0 or 1, so the posterior is undefined.")
    return {
        "p_test_pos": p_pos,
        "p_disease_given_pos": sensitivity * prior / p_pos,
        "p_disease_given_neg": (1.0 - sensitivity) * prior / (1.0 - p_pos),
    }


def _numeric_or_none(text: str) -> float | None:
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def _param_row(name: str, text: str) -> dict[str, Any]:
    number = _numeric_or_none(text)
    if number is not None:
        return {"name": name, "base": number}
    return {"name": name, "base": 0.0, "definition": text.strip()}


def add_binary_bayes_structure(
    document: dict[str, Any],
    parent_id: str,
    prefix: str,
    prior: str,
    sensitivity: str,
    specificity: str,
    disease_label: str = "Disease",
    test_label: str = "Test",
) -> tuple[str | None, str | None]:
    """Append parameters, derived variables and a Test -> Disease chance subtree under
    ``parent_id``. Returns ``(error, new_root_id)``."""

    prefix = prefix.strip()
    if not prefix or not prefix.replace("_", "").isalnum() or prefix[0].isdigit():
        return "Prefix must be letters, digits or underscores and not start with a digit.", None
    for label, text in (("Prevalence", prior), ("Sensitivity", sensitivity), ("Specificity", specificity)):
        if not str(text).strip():
            return f"{label} is required.", None
    nodes = {node["id"]: node for node in document.get("nodes", [])}
    parent = nodes.get(parent_id)
    if parent is None or parent.get("type") == "terminal":
        return "Select a decision or chance node to attach the test to.", None

    names = {
        "prev": f"{prefix}_prev", "se": f"{prefix}_se", "sp": f"{prefix}_sp",
        "pos": f"{prefix}_p_pos", "dpos": f"{prefix}_p_d_pos", "dneg": f"{prefix}_p_d_neg",
    }
    parameters = document.setdefault("parameters", [])
    variables = document.setdefault("variables", [])
    existing = {row.get("name") for row in parameters} | {row.get("name") for row in variables}
    clash = [n for n in names.values() if n in existing]
    if clash:
        return f"Parameter name already exists: {', '.join(clash)}.", None

    for number_text, label in ((prior, "Prevalence"), (sensitivity, "Sensitivity"), (specificity, "Specificity")):
        number = _numeric_or_none(number_text)
        if number is not None and not 0.0 <= number <= 1.0:
            return f"{label} must be between 0 and 1.", None

    prev, se, sp = names["prev"], names["se"], names["sp"]
    parameters.extend([
        _param_row(prev, str(prior)),
        _param_row(se, str(sensitivity)),
        _param_row(sp, str(specificity)),
    ])
    # Derived values are *variables* (evaluated from the current parameter
    # values), so PSA / SA / VOI recompute the posteriors for every draw.
    variables.extend([
        {"name": names["pos"], "definition": f"({se})*({prev})+(1-({sp}))*(1-({prev}))"},
        {"name": names["dpos"],
         "definition": f"({se})*({prev})/(({se})*({prev})+(1-({sp}))*(1-({prev})))"},
        {"name": names["dneg"],
         "definition": f"(1-({se}))*({prev})/(1-(({se})*({prev})+(1-({sp}))*(1-({prev}))))"},
    ])

    def unique(base: str) -> str:
        taken = {node["id"] for node in document["nodes"]}
        candidate, i = base, 1
        while candidate in taken:
            i += 1
            candidate = f"{base}_{i}"
        return candidate

    test_id = unique(f"{prefix}_test")
    pos_id, neg_id = unique(f"{prefix}_pos"), unique(f"{prefix}_neg")
    new_nodes: list[dict[str, Any]] = [
        {"id": test_id, "type": "chance", "label": test_label, "branches": []},
        {"id": pos_id, "type": "chance", "label": f"{test_label}+", "branches": []},
        {"id": neg_id, "type": "chance", "label": f"{test_label}-", "branches": []},
    ]
    document["nodes"].extend(new_nodes)
    taken = {n["id"] for n in document["nodes"]}

    def terminal(base: str, label: str) -> str:
        node_id, i = base, 1
        while node_id in taken:
            i += 1
            node_id = f"{base}_{i}"
        taken.add(node_id)
        document["nodes"].append({"id": node_id, "type": "terminal", "label": label,
                                  "cost": 0.0, "effect": 0.0, "year": 0.0, "branches": []})
        return node_id

    by_id = {n["id"]: n for n in document["nodes"]}
    by_id[test_id]["branches"] = [
        {"label": f"{test_label}+", "child_id": pos_id, "probability": names["pos"]},
        {"label": f"{test_label}-", "child_id": neg_id, "probability": "#"},
    ]
    for outcome_id, key, tag in ((pos_id, "dpos", "T+"), (neg_id, "dneg", "T-")):
        by_id[outcome_id]["branches"] = [
            {"label": disease_label, "child_id": terminal(f"{outcome_id}_d", f"{disease_label} | {tag}"),
             "probability": names[key]},
            {"label": f"No {disease_label.lower()}",
             "child_id": terminal(f"{outcome_id}_nd", f"No {disease_label.lower()} | {tag}"),
             "probability": "#"},
        ]
    parent.setdefault("branches", []).append({
        "label": test_label, "child_id": test_id,
        **({"probability": "0.0"} if parent.get("type") == "chance" else {}),
    })
    return None, test_id


# --------------------------------------------------------------------------
# v2: n disease states x m test results, and "Flip" of an existing tree
# --------------------------------------------------------------------------

def posterior_matrix(priors: list[float], likelihood: list[list[float]]) -> dict[str, Any]:
    """``likelihood[i][j] = P(result j | state i)``.  Returns P(result j) and
    ``posterior[i][j] = P(state i | result j) = prior_i * L_ij / P(result j)``."""
    n = len(priors)
    if n < 2 or len(likelihood) != n:
        raise ValueError("Need at least two states and one likelihood row per state.")
    m = len(likelihood[0])
    if m < 2 or any(len(row) != m for row in likelihood):
        raise ValueError("Every likelihood row needs the same number (>= 2) of test results.")
    if abs(sum(priors) - 1.0) > 1e-9 or any(p < 0 for p in priors):
        raise ValueError("Prior probabilities must be >= 0 and sum to 1.")
    for row in likelihood:
        if abs(sum(row) - 1.0) > 1e-9 or any(v < 0 for v in row):
            raise ValueError("Each likelihood row must be >= 0 and sum to 1.")
    p_result = [sum(priors[i] * likelihood[i][j] for i in range(n)) for j in range(m)]
    if any(p <= 0 for p in p_result):
        raise ValueError("A test result has probability 0, so its posterior is undefined.")
    posterior = [[priors[i] * likelihood[i][j] / p_result[j] for j in range(m)] for i in range(n)]
    return {"p_result": p_result, "posterior": posterior}


def _valid_prefix(prefix: str) -> bool:
    return bool(prefix) and prefix.replace("_", "").isalnum() and not prefix[0].isdigit()


def _expr_of(text: Any) -> str:
    return str(text).strip()


def add_matrix_bayes_structure(
    document: dict[str, Any],
    parent_id: str,
    prefix: str,
    state_labels: list[str],
    priors: list[str],
    result_labels: list[str],
    likelihood: list[list[str]],
    test_label: str = "Test",
) -> tuple[str | None, str | None]:
    """Test (m results) -> state (n states) subtree from priors and a likelihood
    matrix.  A cell/prior equal to ``#`` is the complement of the rest of its
    row (prior list).  Inputs become parameters (numbers) or variables
    (expressions / complements); posteriors are variables so PSA recomputes."""

    prefix = prefix.strip()
    if not _valid_prefix(prefix):
        return "Prefix must be letters, digits or underscores and not start with a digit.", None
    n, m = len(state_labels), len(result_labels)
    if n < 2 or m < 2:
        return "Need at least two states and two test results.", None
    if len(priors) != n or len(likelihood) != n or any(len(r) != m for r in likelihood):
        return "Prior list and likelihood matrix do not match the numbers of states and results.", None
    if len(set(state_labels)) != n or len(set(result_labels)) != m or any(not s.strip() for s in [*state_labels, *result_labels]):
        return "State labels and result labels must be non-empty and unique.", None
    nodes = {node["id"]: node for node in document.get("nodes", [])}
    parent = nodes.get(parent_id)
    if parent is None or parent.get("type") == "terminal":
        return "Select a decision or chance node to attach the test to.", None

    def complement_count(cells: list[str]) -> int:
        return sum(1 for c in cells if _expr_of(c) == "#")

    if complement_count(priors) > 1 or any(complement_count(r) > 1 for r in likelihood):
        return "Use '#' (the remainder) at most once per row / for the prior list.", None
    for text in [*priors, *[c for r in likelihood for c in r]]:
        if not _expr_of(text):
            return "Every prior and likelihood cell is required.", None
        number = _numeric_or_none(text)
        if number is not None and not 0.0 <= number <= 1.0:
            return "Probabilities must be between 0 and 1.", None
    # fully numeric inputs can be validated now
    flat = [*priors, *[c for r in likelihood for c in r]]
    if all(_numeric_or_none(c) is not None or _expr_of(c) == "#" for c in flat):
        try:
            nums_p = _fill_complement([_numeric_or_none(c) if _expr_of(c) != "#" else None for c in priors])
            nums_l = [_fill_complement([_numeric_or_none(c) if _expr_of(c) != "#" else None for c in r]) for r in likelihood]
            posterior_matrix(nums_p, nums_l)
        except ValueError as exc:
            return str(exc), None

    parameters = document.setdefault("parameters", [])
    variables = document.setdefault("variables", [])
    existing = {r.get("name") for r in parameters} | {r.get("name") for r in variables}
    prev_n = [f"{prefix}_prev{i + 1}" for i in range(n)]
    lik_n = [[f"{prefix}_L{i + 1}_{j + 1}" for j in range(m)] for i in range(n)]
    pres_n = [f"{prefix}_pT{j + 1}" for j in range(m)]
    post_n = [[f"{prefix}_pD{i + 1}_T{j + 1}" for j in range(m)] for i in range(n)]
    all_names = [*prev_n, *[x for r in lik_n for x in r], *pres_n, *[x for r in post_n for x in r]]
    clash = [x for x in all_names if x in existing]
    if clash:
        return f"Parameter name already exists: {', '.join(clash[:4])}.", None

    def add_input(name: str, text: str, siblings: list[str], index: int) -> None:
        text = _expr_of(text)
        if text == "#":
            others = " - ".join(f"({s})" for k, s in enumerate(siblings) if k != index)
            variables.append({"name": name, "definition": f"1 - {others}"})
        else:
            row = _param_row(name, text)
            if "definition" in row:
                variables.append({"name": name, "definition": row["definition"]})
            else:
                parameters.append(row)

    for i in range(n):
        add_input(prev_n[i], priors[i], prev_n, i)
    for i in range(n):
        for j in range(m):
            add_input(lik_n[i][j], likelihood[i][j], lik_n[i], j)
    for j in range(m):
        variables.append({"name": pres_n[j], "definition": " + ".join(f"({prev_n[i]})*({lik_n[i][j]})" for i in range(n))})
    for i in range(n):
        for j in range(m):
            variables.append({"name": post_n[i][j], "definition": f"({prev_n[i]})*({lik_n[i][j]})/({pres_n[j]})"})

    taken = {node["id"] for node in document["nodes"]}

    def unique(base: str) -> str:
        candidate, k = base, 1
        while candidate in taken:
            k += 1
            candidate = f"{base}_{k}"
        taken.add(candidate)
        return candidate

    test_id = unique(f"{prefix}_test")
    result_ids = [unique(f"{prefix}_r{j + 1}") for j in range(m)]
    document["nodes"].append({"id": test_id, "type": "chance", "label": test_label, "branches": []})
    test_node = document["nodes"][-1]
    for j in range(m):
        document["nodes"].append({"id": result_ids[j], "type": "chance", "label": result_labels[j], "branches": []})
        res_node = document["nodes"][-1]
        test_node["branches"].append({
            "label": result_labels[j], "child_id": result_ids[j],
            "probability": "#" if j == m - 1 else pres_n[j],
        })
        for i in range(n):
            leaf = unique(f"{result_ids[j]}_s{i + 1}")
            document["nodes"].append({"id": leaf, "type": "terminal", "label": f"{state_labels[i]} | {result_labels[j]}",
                                      "cost": 0.0, "effect": 0.0, "year": 0.0, "branches": []})
            res_node["branches"].append({
                "label": state_labels[i], "child_id": leaf,
                "probability": "#" if i == n - 1 else post_n[i][j],
            })
    parent.setdefault("branches", []).append({
        "label": test_label, "child_id": test_id,
        **({"probability": "0.0"} if parent.get("type") == "chance" else {}),
    })
    return None, test_id


def _fill_complement(values: list[float | None]) -> list[float]:
    missing = [k for k, v in enumerate(values) if v is None]
    if len(missing) > 1 or any(v is None and False for v in values):
        raise ValueError("At most one '#' per row.")
    if missing:
        rest = sum(v for v in values if v is not None)
        values = list(values)
        values[missing[0]] = 1.0 - rest
        if values[missing[0]] < -1e-12:
            raise ValueError("Probabilities in a row exceed 1.")
    if any(v is None for v in values):
        raise ValueError("Missing probability.")
    return [float(v) for v in values]


def _branch_prob_expr(branches: list[dict[str, Any]], index: int) -> str:
    """Expression for branch ``index``; '#' becomes one minus the other branches."""
    text = _expr_of(branches[index].get("probability", ""))
    if text and text != "#":
        return f"({text})"
    if text == "#":
        others = [_expr_of(b.get("probability", "")) for k, b in enumerate(branches) if k != index]
        if any((not o) or o == "#" for o in others):
            raise ValueError("A '#' branch needs all other branches to have explicit probabilities.")
        return "(1 - " + " - ".join(f"({o})" for o in others) + ")"
    raise ValueError("Every branch needs a probability to flip the tree.")


def flip_chance_tree(document: dict[str, Any], node_id: str, prefix: str) -> tuple[str | None, str | None]:
    """Swap the order of two consecutive chance levels (Disease -> Test  becomes
    Test -> Disease), as TreeAge's Flip does.  ``node_id`` is the first chance
    node; each of its children must be a chance node with the same set of branch
    labels.  The grandchild subtrees are re-used, not copied.  The flipped
    probabilities are variables (Bayes' rule) so SA / PSA recompute them.
    Returns ``(error, node_id)``."""

    prefix = prefix.strip()
    if not _valid_prefix(prefix):
        return "Prefix must be letters, digits or underscores and not start with a digit.", None
    by_id = {n["id"]: n for n in document.get("nodes", [])}
    top = by_id.get(node_id)
    if top is None or top.get("type") != "chance" or len(top.get("branches", [])) < 2:
        return "Select a chance node with at least two branches to flip.", None
    middle_ids = []
    for branch in top["branches"]:
        child = by_id.get(branch.get("child_id"))
        if child is None or child.get("type") != "chance":
            return "Every branch of the node must lead to a chance node (the second level).", None
        middle_ids.append(child["id"])
    if len(set(middle_ids)) != len(middle_ids):
        return "The second-level nodes must be distinct (not shared or cloned).", None
    if any(by_id[m].get("clone_of") or by_id[node_id].get("clone_of") for m in middle_ids):
        return "Clone nodes cannot be flipped; detach them first.", None
    first_labels = [b.get("label") for b in by_id[middle_ids[0]]["branches"]]
    if len(first_labels) < 2 or len(set(first_labels)) != len(first_labels):
        return "Second-level branch labels must be unique and at least two.", None
    for mid in middle_ids:
        labels = [b.get("label") for b in by_id[mid]["branches"]]
        if sorted(map(str, labels)) != sorted(map(str, first_labels)):
            return "All second-level nodes need the same set of branch labels.", None
    for owner in (top, *[by_id[m] for m in middle_ids]):
        for branch in owner["branches"]:
            extra = [k for k, v in branch.items()
                     if k not in ("label", "child_id", "probability") and v not in (None, "", 0, 0.0, False)]
            if extra:
                return f"Branch '{branch.get('label')}' carries {', '.join(extra)}; move payoffs to the terminals before flipping.", None

    n, m = len(middle_ids), len(first_labels)
    state_labels = [b.get("label") for b in top["branches"]]
    try:
        prior_expr = [_branch_prob_expr(top["branches"], i) for i in range(n)]
        lik_expr: list[list[str]] = []
        grand: list[list[str]] = []
        for i, mid in enumerate(middle_ids):
            branches = by_id[mid]["branches"]
            index_by_label = {b.get("label"): k for k, b in enumerate(branches)}
            lik_expr.append([_branch_prob_expr(branches, index_by_label[lab]) for lab in first_labels])
            grand.append([branches[index_by_label[lab]]["child_id"] for lab in first_labels])
    except ValueError as exc:
        return str(exc), None

    variables = document.setdefault("variables", [])
    existing = {r.get("name") for r in document.get("parameters", [])} | {r.get("name") for r in variables}
    pres_n = [f"{prefix}_pT{j + 1}" for j in range(m)]
    post_n = [[f"{prefix}_pD{i + 1}_T{j + 1}" for j in range(m)] for i in range(n)]
    clash = [x for x in [*pres_n, *[y for r in post_n for y in r]] if x in existing]
    if clash:
        return f"Name already exists: {', '.join(clash[:4])}.", None

    for j in range(m):
        variables.append({"name": pres_n[j], "definition": " + ".join(f"{prior_expr[i]}*{lik_expr[i][j]}" for i in range(n))})
    for i in range(n):
        for j in range(m):
            variables.append({"name": post_n[i][j], "definition": f"{prior_expr[i]}*{lik_expr[i][j]}/({pres_n[j]})"})

    taken = {x["id"] for x in document["nodes"]}

    def unique(base: str) -> str:
        candidate, k = base, 1
        while candidate in taken:
            k += 1
            candidate = f"{base}_{k}"
        taken.add(candidate)
        return candidate

    # discard the old second level, build the new one
    old_middle = set(middle_ids)
    document["nodes"] = [x for x in document["nodes"] if x["id"] not in old_middle]
    new_top_branches = []
    for j in range(m):
        new_id = unique(f"{prefix}_f{j + 1}")
        document["nodes"].append({
            "id": new_id, "type": "chance", "label": first_labels[j],
            "branches": [
                {"label": state_labels[i], "child_id": grand[i][j],
                 "probability": "#" if i == n - 1 else post_n[i][j]}
                for i in range(n)
            ],
        })
        new_top_branches.append({
            "label": first_labels[j], "child_id": new_id,
            "probability": "#" if j == m - 1 else pres_n[j],
        })
    top["branches"] = new_top_branches
    return None, node_id
