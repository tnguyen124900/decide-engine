"""Translate a DECIDE expression into an Excel formula, for the "live" cells of the transparency workbooks.

Deliberately conservative: only a small, unambiguous subset is translated (arithmetic, comparisons, and/or/not,
if(...), a handful of math functions, parameter names, `_cycle`).  Anything else -- distribution/table functions,
unknown names, round(), floor division, subscripts... -- returns None and the caller falls back to a resolved
(amber, commented) value, exactly as for any expression with no Excel equivalent.  Every sub-expression is wrapped
in parentheses so Excel's operator precedence (e.g. unary minus vs ^) can never change the meaning.
"""

from __future__ import annotations

import ast
import math
import re
from collections.abc import Mapping
from typing import Callable

_ONE_ARG = {
    "abs": "ABS", "exp": "EXP", "ln": "LN", "log10": "LOG10", "sqrt": "SQRT", "sign": "SIGN",
    "sin": "SIN", "cos": "COS", "tan": "TAN", "sinh": "SINH", "cosh": "COSH", "tanh": "TANH",
    "arcsin": "ASIN", "arccos": "ACOS", "arctan": "ATAN", "floor": "INT",
}
_VARIADIC = {"min": "MIN", "max": "MAX"}
_COMPARE = {ast.Eq: "=", ast.NotEq: "<>", ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">="}


class _Unsupported(Exception):
    pass


def expression_to_excel(
    expression: str,
    name_to_ref: Mapping[str, str],
    cycle_ref: str | None = None,
) -> str | None:
    """`name_to_ref` maps a parameter name to its Excel reference (defined name); `cycle_ref` is the cell holding
    the current `_cycle` (None -> any expression using `_cycle` is not translatable).  Returns the formula body
    WITHOUT a leading '=', or None when the expression is outside the supported subset."""

    normalized = re.sub(r"\bif\s*\(", "if_(", str(expression))
    try:
        tree = ast.parse(normalized, mode="eval")
        return _emit(tree.body, name_to_ref, cycle_ref)
    except (_Unsupported, SyntaxError, RecursionError):
        return None


def _num(value: float) -> str:
    return repr(float(value)) if not float(value).is_integer() else str(int(value))


def _emit(node: ast.AST, names: Mapping[str, str], cycle_ref: str | None) -> str:
    def rec(n: ast.AST) -> str:
        return _emit(n, names, cycle_ref)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool):
            return "TRUE" if node.value else "FALSE"
        if isinstance(node.value, (int, float)):
            return _num(node.value)
        raise _Unsupported
    if isinstance(node, ast.Name):
        if node.id == "_cycle":
            if cycle_ref is None:
                raise _Unsupported
            return cycle_ref
        if node.id == "_tenure":
            return "1"
        if node.id == "e":
            return _num(math.e)
        if node.id == "pi":
            return "PI()"
        if node.id in names:
            return names[node.id]
        raise _Unsupported
    if isinstance(node, ast.UnaryOp):
        if isinstance(node.op, ast.USub):
            return f"(-({rec(node.operand)}))"
        if isinstance(node.op, ast.UAdd):
            return f"({rec(node.operand)})"
        if isinstance(node.op, ast.Not):
            return f"(NOT({rec(node.operand)}))"
        raise _Unsupported
    if isinstance(node, ast.BinOp):
        left, right = rec(node.left), rec(node.right)
        if isinstance(node.op, ast.Add):
            return f"(({left})+({right}))"
        if isinstance(node.op, ast.Sub):
            return f"(({left})-({right}))"
        if isinstance(node.op, ast.Mult):
            return f"(({left})*({right}))"
        if isinstance(node.op, ast.Div):
            return f"(({left})/({right}))"
        if isinstance(node.op, ast.Pow):
            return f"(({left})^({right}))"
        if isinstance(node.op, ast.Mod):
            return f"MOD({left},{right})"
        raise _Unsupported
    if isinstance(node, ast.BoolOp):
        parts = ",".join(rec(v) for v in node.values)
        if isinstance(node.op, ast.And):
            return f"AND({parts})"
        if isinstance(node.op, ast.Or):
            return f"OR({parts})"
        raise _Unsupported
    if isinstance(node, ast.Compare):
        if len(node.ops) != 1:
            raise _Unsupported
        symbol = next((s for k, s in _COMPARE.items() if isinstance(node.ops[0], k)), None)
        if symbol is None:
            raise _Unsupported
        return f"(({rec(node.left)}){symbol}({rec(node.comparators[0])}))"
    if isinstance(node, ast.IfExp):
        return f"IF({rec(node.test)},{rec(node.body)},{rec(node.orelse)})"
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.keywords:
            raise _Unsupported
        fn, args = node.func.id, node.args
        if fn == "if_" and len(args) == 3:
            return f"IF({rec(args[0])},{rec(args[1])},{rec(args[2])})"
        if fn in _ONE_ARG and len(args) == 1:
            return f"{_ONE_ARG[fn]}({rec(args[0])})"
        if fn == "log" and len(args) == 1:
            return f"LN({rec(args[0])})"
        if fn == "log" and len(args) == 2:
            return f"LOG({rec(args[0])},{rec(args[1])})"
        if fn in _VARIADIC and len(args) >= 1 and not any(isinstance(a, (ast.Tuple, ast.List, ast.Set)) for a in args):
            return f"{_VARIADIC[fn]}({','.join(rec(a) for a in args)})"
        if fn == "ceil" and len(args) == 1:
            return f"(-INT(-({rec(args[0])})))"
        raise _Unsupported
    raise _Unsupported
