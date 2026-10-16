"""Whole-model formula check.

``check_document_formulas`` walks every expression of a model document
(parameters, variables, branch probabilities and payoffs, terminal payoffs,
distribution arguments) and lists *all* problems at once instead of stopping at
the first evaluation error: syntax errors, unknown names / functions /
distributions / tables (each with a "Did you mean" hint), duplicate names and
circular variable definitions.  Nothing is evaluated, so it is cheap and safe
to call on every edit.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any

from .expressions import ExpressionError, _FUNCTION_NAMES, _did_you_mean, _parse_expression

# keys whose string value is a formula
_EXPR_KEYS = {
    "probability", "cost", "effect", "definition", "base", "low", "high",
    "cost_increment", "effect_increment", "initial_cost", "recurring_cost", "terminal_cost",
    "expr", "initial",  # tracker updates / initial values
}
_RESERVED = {"e", "pi", "True", "False"}
_DIST_TABLE_FUNCS = {"DistMean", "DistSample", "DistCDF", "DistPPF", "DistSampleCond", "DistToProb",
                     "DistToTime", "DistTrim", "DistSurv", "DistHazard", "TableToProb", "TableToTime"}


@dataclass
class FormulaIssue:
    severity: str  # "error" | "warning"
    where: str
    formula: str
    message: str

    def as_row(self) -> dict[str, str]:
        return {"Severity": self.severity, "Where": self.where, "Formula": self.formula, "Problem": self.message}


def _names(rows: Any, key: str = "name") -> list[str]:
    return [str(r.get(key)) for r in (rows or []) if isinstance(r, dict) and r.get(key)]


def _walk_expressions(obj: Any, where: str, extra_keys: set[str]):
    """Yield (where, text) for every formula-like string in a nested structure."""
    if isinstance(obj, dict):
        label = obj.get("name") or obj.get("label") or obj.get("id")
        here = f"{where} [{label}]" if label and where else where
        for key, value in obj.items():
            if isinstance(value, str) and (key in _EXPR_KEYS or key in extra_keys):
                yield (f"{here} → {key}" if here else key), value
            elif key == "dist_args" and isinstance(value, dict):
                for arg, arg_value in value.items():
                    if isinstance(arg_value, str):
                        yield f"{here} → {arg}", arg_value
            elif isinstance(value, (dict, list)) and key not in ("analysis", "distribution", "tables"):
                if key in ("nodes", "branches"):
                    child = here
                else:
                    child = f"{here} → {key}" if here else key
                yield from _walk_expressions(value, child, extra_keys)
    elif isinstance(obj, list):
        for item in obj:
            yield from _walk_expressions(item, where, extra_keys)


def _referenced(expression: str) -> tuple[set[str], set[str], set[str]]:
    """(names, called functions, names given as the first argument of a Dist*/Table* call)."""
    tree = _parse_expression(expression)
    names: set[str] = set()
    funcs: set[str] = set()
    first_args: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                funcs.add(node.func.id)
                if node.func.id in _DIST_TABLE_FUNCS and node.args:
                    first = node.args[0]
                    if isinstance(first, ast.Name):
                        first_args.add(first.id)
                    elif isinstance(first, ast.Constant) and isinstance(first.value, str):
                        first_args.add(first.value)
        elif isinstance(node, ast.Name):
            names.add(node.id)
    # names that are only function targets are not variables
    names -= funcs & {n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    return names, funcs, first_args


def check_document_formulas(document: dict[str, Any]) -> list[FormulaIssue]:
    issues: list[FormulaIssue] = []
    params = _names(document.get("parameters"))
    variables = _names(document.get("variables"))
    dists = _names(document.get("distributions"))
    tables = _names(document.get("tables"))

    for kind, names in (("parameter", params), ("variable", variables)):
        seen: set[str] = set()
        for name in names:
            if name in seen:
                issues.append(FormulaIssue("error", f"{kind} [{name}]", "", f"Duplicate {kind} name '{name}'."))
            seen.add(name)
    for name in set(params) & set(variables):
        issues.append(FormulaIssue("error", f"[{name}]", "", f"'{name}' is both a parameter and a variable."))

    trackers = _names(document.get("trackers"))
    known_values = set(params) | set(variables) | set(trackers) | _RESERVED
    known_refs = known_values | set(dists) | set(tables)
    functions = {n.rstrip("_") for n in _FUNCTION_NAMES} | _FUNCTION_NAMES | set(tables)
    outcome_ids = {str(o.get("id")) for o in document.get("outcomes", []) if isinstance(o, dict) and o.get("id")}

    var_deps: dict[str, set[str]] = {}
    for where, text in _walk_expressions(document, "", outcome_ids):
        text = text.strip()
        if not text or text == "#":
            continue
        try:
            float(text)
            continue
        except ValueError:
            pass
        try:
            names, funcs, first_args = _referenced(text)
        except (ExpressionError, SyntaxError, ValueError) as exc:
            issues.append(FormulaIssue("error", where, text, f"Cannot parse: {exc}"))
            continue
        for name in sorted(names):
            if name in known_refs or name.startswith("_") or name in first_args:
                continue
            issues.append(FormulaIssue("error", where, text, f"Unknown name '{name}'" + _did_you_mean(name, sorted(known_values))))
        for func in sorted(funcs):
            if func in functions or func == "if":
                continue
            issues.append(FormulaIssue("error", where, text, f"Unknown function '{func}'" + _did_you_mean(func, sorted(functions))))
        for ref in sorted(first_args):
            if ref not in dists and ref not in tables:
                issues.append(FormulaIssue("error", where, text, f"Unknown distribution or table '{ref}'" + _did_you_mean(ref, [*dists, *tables])))
        if where.startswith("variables [") and where.endswith("→ definition"):
            owner = where[len("variables ["):where.index("]")]
            var_deps[owner] = names & set(variables)

    # circular variable references
    state: dict[str, int] = {}
    reported: set[frozenset[str]] = set()

    def visit(name: str, stack: list[str]) -> None:
        if state.get(name) == 2:
            return
        if state.get(name) == 1:
            cycle = stack[stack.index(name):]
            key = frozenset(cycle)
            if key not in reported:
                reported.add(key)
                issues.append(FormulaIssue("error", f"variable [{name}]", "", "Circular definition: " + " → ".join([*cycle, name])))
            return
        state[name] = 1
        for dep in sorted(var_deps.get(name, ())):
            visit(dep, [*stack, name])
        state[name] = 2

    for name in list(var_deps):
        visit(name, [])
    return issues
