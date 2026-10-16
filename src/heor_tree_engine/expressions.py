"""Safe evaluation of DECIDE numeric expressions."""

from __future__ import annotations

import ast
import functools
import math
import operator
import re
import statistics
from collections.abc import Mapping, Sequence
from typing import Any, Callable

from .models import Param
from .distributions import (
    distribution_cdf,
    distribution_hazard,
    distribution_mean,
    distribution_ppf,
    distribution_sample,
    distribution_sample_cond,
    distribution_survival,
    distribution_to_prob,
    distribution_to_time,
    distribution_trim,
)


class ExpressionError(ValueError):
    """Raised when a DECIDE expression cannot be evaluated safely."""


@functools.lru_cache(maxsize=8192)
def _parse_expression(expression: str) -> ast.Expression:
    """Parse one DECIDE expression string into its AST, memoized by the
    expression's own text.

    evaluate() used to re-run this (a regex substitution + ast.parse,
    which itself calls the C-level compile()) from scratch on EVERY
    call, including every one of the millions of calls a long Markov
    run makes re-evaluating the same handful of transition-probability/
    cost/effect expression strings once per (state, cycle, tenure)
    combination -- profiling a real 600-cycle, tenure-tracked model
    (tenure_count=1000) showed ~11s of a ~26s run spent inside
    ast.parse/compile alone, purely re-parsing text that had already
    been parsed thousands of times before. The parsed tree depends only
    on the expression's text, never on the parameters/variables it will
    be evaluated against (those are looked up fresh by _eval_node every
    call against THIS call's own namespace), so caching the parse here
    is safe and cycle/tenure-agnostic.
    """

    normalized = re.sub(r"\bif\s*\(", "if_(", expression)
    try:
        return ast.parse(normalized, mode="eval")
    except SyntaxError as exc:
        raise ExpressionError(f"Invalid expression '{expression}': {exc.msg}") from exc


def _format_number(value: float) -> str:
    return f"{value:g}"


_FUNCTION_NAMES = {
    "abs",
    "arccos",
    "arcsin",
    "arctan",
    "bound",
    "cbrt",
    "ceil",
    "comb",
    "cos",
    "cosh",
    "erf",
    "exp",
    "factorial",
    "floor",
    "gamma",
    "hypot",
    "if_",
    "invErf",
    "lgamma",
    "ln",
    "log",
    "log10",
    "logit",
    "logis",
    "max",
    "min",
    "ProbToProb",
    "ProbToRate",
    "RateToProb",
    "round",
    "sin",
    "sinh",
    "sqrt",
    "tan",
    "tanh",
    "sign",
    "mean",
    "prod",
    "quantile",
    "sd",
    "sum",
    "var",
    "DistMean",
    "DistSample",
    "DistCDF",
    "DistPPF",
    "DistSampleCond",
    "DistToProb",
    "DistToTime",
    "TableToProb",
    "TableToTime",
    "DistTrim",
    "DistSurv",
    "DistHazard",
}


class _Evaluator:
    def __init__(
        self,
        parameters: Mapping[str, Param],
        variables: Mapping[str, Any] | None = None,
        distributions: Mapping[str, Mapping[str, Any]] | None = None,
        tables: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> None:
        self.parameters = parameters
        self.variables = variables or {}
        self.distributions = distributions or {}
        self.tables = tables or {}
        self._variable_cache: dict[str, float] = {}
        self._resolving: set[str] = set()

    def evaluate(self, value: Any) -> float:
        if isinstance(value, bool):
            raise ExpressionError("Boolean values are not numeric")
        if isinstance(value, (int, float)):
            return float(value)
        if not isinstance(value, str):
            raise ExpressionError(f"Unsupported expression value: {value!r}")

        expression = value.strip()
        if not expression:
            raise ExpressionError("Expression cannot be empty")
        if expression in self.parameters or expression in self.variables:
            return self._resolve_name(expression)

        tree = _parse_expression(expression)
        return self._eval_node(tree.body)

    def _resolve_name(self, name: str) -> float:
        parameter = self.parameters.get(name)
        if parameter is not None:
            return float(parameter.base)
        if name in self._variable_cache:
            return self._variable_cache[name]
        if name not in self.variables:
            raise ExpressionError(
                f"Unknown name '{name}'"
                + _did_you_mean(name, [*self.parameters, *self.variables])
            )
        if name in self._resolving:
            raise ExpressionError(f"Circular variable reference involving '{name}'")
        self._resolving.add(name)
        try:
            result = self.evaluate(self.variables[name])
        finally:
            self._resolving.remove(name)
        self._variable_cache[name] = result
        return result

    def _eval_node(self, node: ast.AST) -> float | bool | list[float] | tuple[float, ...]:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool):
                return node.value
            if isinstance(node.value, (int, float)):
                return float(node.value)
            raise ExpressionError("Only numeric constants are allowed")

        if isinstance(node, ast.Name):
            if node.id == "e":
                return math.e
            if node.id == "pi":
                return math.pi
            return self._resolve_name(node.id)

        if isinstance(node, ast.UnaryOp):
            operand = self._numeric(self._eval_node(node.operand))
            if isinstance(node.op, ast.UAdd):
                return +operand
            if isinstance(node.op, ast.USub):
                return -operand
            if isinstance(node.op, ast.Not):
                return not bool(operand)
            raise ExpressionError("Unsupported unary operator")

        if isinstance(node, ast.BinOp):
            left = self._numeric(self._eval_node(node.left))
            right = self._numeric(self._eval_node(node.right))
            operations: dict[type[ast.operator], Callable[[float, float], float]] = {
                ast.Add: operator.add,
                ast.Sub: operator.sub,
                ast.Mult: operator.mul,
                ast.Div: operator.truediv,
                ast.Pow: operator.pow,
                ast.Mod: operator.mod,
                ast.FloorDiv: operator.floordiv,
            }
            for operation_type, operation in operations.items():
                if isinstance(node.op, operation_type):
                    return operation(left, right)
            raise ExpressionError("Unsupported binary operator")

        if isinstance(node, ast.BoolOp):
            values = [bool(self._eval_node(value)) for value in node.values]
            if isinstance(node.op, ast.And):
                return all(values)
            if isinstance(node.op, ast.Or):
                return any(values)
            raise ExpressionError("Unsupported logical operator")

        if isinstance(node, ast.Compare):
            left = self._numeric_or_bool(self._eval_node(node.left))
            comparisons = []
            for operation, comparator in zip(node.ops, node.comparators):
                right = self._numeric_or_bool(self._eval_node(comparator))
                comparison_map: dict[type[ast.cmpop], Callable[[Any, Any], bool]] = {
                    ast.Eq: operator.eq,
                    ast.NotEq: operator.ne,
                    ast.Lt: operator.lt,
                    ast.LtE: operator.le,
                    ast.Gt: operator.gt,
                    ast.GtE: operator.ge,
                }
                comparison = next(
                    (function for kind, function in comparison_map.items() if isinstance(operation, kind)),
                    None,
                )
                if comparison is None:
                    raise ExpressionError("Unsupported comparison operator")
                comparisons.append(comparison(left, right))
                left = right
            return all(comparisons)

        if isinstance(node, ast.IfExp):
            branch = node.body if bool(self._eval_node(node.test)) else node.orelse
            return self._eval_node(branch)

        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            values = tuple(self._numeric(self._eval_node(element)) for element in node.elts)
            return list(values) if isinstance(node, (ast.List, ast.Set)) else values

        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCTION_NAMES:
                raise ExpressionError("Only approved DECIDE functions may be called")
            function_name = node.func.id
            if function_name == "if_":
                if len(node.args) != 3:
                    raise ExpressionError("if(condition, a, b) requires three arguments")
                branch = node.args[1] if bool(self._eval_node(node.args[0])) else node.args[2]
                return self._eval_node(branch)
            if function_name == "bound":
                return self._bound(node)
            if function_name in {"TableToProb", "TableToTime"}:
                return self._call_table_probability(function_name, node)
            if function_name in {
                "DistMean", "DistSample", "DistCDF", "DistPPF", "DistSampleCond",
                "DistToProb", "DistToTime", "DistTrim", "DistSurv", "DistHazard",
            }:
                return self._call_distribution(function_name, node)
            args = [self._eval_node(argument) for argument in node.args]
            return self._call_function(function_name, args)

        if isinstance(node, ast.Subscript):
            return self._eval_table_lookup(node)

        raise ExpressionError(f"Unsupported expression syntax: {ast.dump(node)}")

    def _bound(self, node: ast.Call) -> float:
        if len(node.args) != 2 or not isinstance(node.args[1], (ast.Tuple, ast.List)):
            raise ExpressionError("bound(x, (a,b)) or bound(x, [a,b]) is required")
        value = self._numeric(self._eval_node(node.args[0]))
        interval = node.args[1]
        endpoints = [self._numeric(self._eval_node(element)) for element in interval.elts]
        if len(endpoints) != 2:
            raise ExpressionError("bound intervals require exactly two endpoints")
        lower, upper = endpoints
        if lower > upper:
            raise ExpressionError("bound interval lower endpoint exceeds upper endpoint")
        lower_ok = value > lower if isinstance(interval, ast.Tuple) else value >= lower
        upper_ok = value < upper if isinstance(interval, ast.Tuple) else value <= upper
        if lower_ok and upper_ok:
            return value
        return min(max(value, lower), upper)

    def _distribution_reference(self, node: ast.AST) -> Mapping[str, Any]:
        if isinstance(node, ast.Name):
            name = node.id
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            name = node.value
        else:
            raise ExpressionError("The first argument must be a distribution name")
        definition = self.distributions.get(name)
        if definition is None:
            raise ExpressionError(
                f"Unknown distribution '{name}'" + _did_you_mean(name, list(self.distributions))
            )
        return definition

    def _resolve_distribution_parameter(self, value: Any) -> float:
        return self.evaluate(value)

    def _call_distribution(self, name: str, node: ast.Call) -> float:
        if not node.args:
            raise ExpressionError(f"{name} requires a distribution argument")
        definition = self._distribution_reference(node.args[0])
        resolve = self._resolve_distribution_parameter
        if name == "DistMean":
            return distribution_mean(definition, resolve)
        if name == "DistSample":
            # Inside a DES the patient's own stream ("__rng__") makes the
            # draw reproducible; elsewhere it is scipy's global state as before.
            return distribution_sample(definition, resolve, rng=self.variables.get("__rng__"))
        if name == "DistCDF":
            if len(node.args) != 2:
                raise ExpressionError("DistCDF(distribution, s) requires two arguments")
            return distribution_cdf(definition, self._numeric(self._eval_node(node.args[1])), resolve)
        if name == "DistPPF":
            if len(node.args) != 2:
                raise ExpressionError("DistPPF(distribution, p) requires two arguments")
            return distribution_ppf(definition, self._numeric(self._eval_node(node.args[1])), resolve)
        if name in {"DistSampleCond", "DistTrim"}:
            if len(node.args) != 2 or not isinstance(node.args[1], (ast.Tuple, ast.List)):
                raise ExpressionError(f"{name}(distribution, [lower, upper]) requires an interval")
            bounds = [self._numeric(self._eval_node(item)) for item in node.args[1].elts]
            if len(bounds) != 2:
                raise ExpressionError("Distribution intervals require two bounds")
            function = distribution_sample_cond if name == "DistSampleCond" else distribution_trim
            return function(definition, bounds[0], bounds[1], resolve)
        if name == "DistToProb":
            if len(node.args) != 3:
                raise ExpressionError("DistToProb(distribution, t1, t2) requires three arguments")
            return distribution_to_prob(
                definition,
                self._numeric(self._eval_node(node.args[1])),
                self._numeric(self._eval_node(node.args[2])),
                resolve,
            )
        if name == "DistToTime":
            # DistToTime(distribution) or DistToTime(distribution, t1): a random
            # remaining time until the event for someone event-free at t1 --
            # the time-returning counterpart of DistToProb (same conditional
            # survival formula; see distributions.conditional_event_time).
            if len(node.args) not in (1, 2):
                raise ExpressionError("DistToTime(distribution, t1) requires one or two arguments")
            start = self._numeric(self._eval_node(node.args[1])) if len(node.args) == 2 else 0.0
            return distribution_to_time(definition, start, resolve, rng=self.variables.get("__rng__"))
        if name in {"DistSurv", "DistHazard"}:
            if len(node.args) != 2:
                raise ExpressionError(f"{name}(distribution, s) requires two arguments")
            value = self._numeric(self._eval_node(node.args[1]))
            function = distribution_survival if name == "DistSurv" else distribution_hazard
            return function(definition, value, resolve)
        raise ExpressionError(f"Unknown distribution function '{name}'")

    _TABLE_METHODS = {"exact", "truncate", "linear", "spline"}

    def _call_table_probability(self, name: str, node: ast.Call) -> float:
        """TableToProb(table, t1, t2[, value_column[, type]]) and
        TableToTime(table, t1[, value_column[, type]]) -- the table
        counterparts of DistToProb/DistToTime, sharing their conditional-
        survival formula (des_sampling.table_to_prob/table_to_time). The
        table's first column is the index (e.g. age); the value column
        defaults to the table's default value column, else its second
        column; type is "conditional" (q(x), default), "probability" or
        "cumulative"."""

        from .des_sampling import build_table_cdf, table_to_prob, table_to_time

        n_numeric = 2 if name == "TableToProb" else 1
        n_args = len(node.args)
        if n_args < 1 + n_numeric or n_args > 3 + n_numeric:
            signature = "table, t1, t2" if name == "TableToProb" else "table, t1"
            raise ExpressionError(f"{name}({signature}[, value_column[, type]]) has the wrong number of arguments")
        table_name, table = self._table_reference(node.args[0])
        numbers = [self._numeric(self._eval_node(arg)) for arg in node.args[1:1 + n_numeric]]
        extras: list[str] = []
        for arg in node.args[1 + n_numeric:]:
            if not (isinstance(arg, ast.Constant) and isinstance(arg.value, str)):
                raise ExpressionError(f"{name}: value_column and type must be quoted text, e.g. \"qx\"")
            extras.append(arg.value)
        columns = [str(column) for column in (table.get("columns") or [])]
        if len(columns) < 2:
            raise ExpressionError(f"{name}: table '{table_name}' needs an index column and a value column")
        value_column = extras[0] if extras else (table.get("default_value_column") or columns[1])
        prob_type = extras[1] if len(extras) > 1 else "conditional"
        try:
            cdf = build_table_cdf(table.get("rows") or [], columns[0], str(value_column), prob_type, True)
        except ValueError as exc:
            raise ExpressionError(f"{name}({table_name}): {exc}") from exc
        if name == "TableToProb":
            return table_to_prob(cdf, numbers[0], numbers[1])
        rng = self.variables.get("__rng__")
        if rng is None:
            import numpy as _np

            rng = _np.random.default_rng()
        return table_to_time(cdf, numbers[0], 1.0 - float(rng.random()))

    def _table_reference(self, node: ast.AST) -> tuple[str, Mapping[str, Any]]:
        if not isinstance(node, ast.Name):
            raise ExpressionError(
                "A table lookup must start with a table name, e.g. table_name[...]"
            )
        name = node.id
        table = self.tables.get(name)
        if table is None:
            raise ExpressionError(f"Unknown table '{name}'")
        return name, table

    def _table_column_selector(
        self, node: ast.AST, columns: Sequence[str], table_name: str
    ) -> str:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value not in columns:
                available = ", ".join(columns)
                raise ExpressionError(
                    f"Table '{table_name}' has no column '{node.value}' (available: {available})"
                )
            return node.value
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, (int, float))
            and not isinstance(node.value, bool)
        ):
            index = int(node.value)
            if index != node.value or not (0 <= index < len(columns)):
                raise ExpressionError(
                    f"Table '{table_name}' column index must be an integer between 0 and {len(columns) - 1}"
                )
            return columns[index]
        raise ExpressionError(
            "A table column must be given as a quoted column name or a 0-based integer index"
        )

    def _table_method_selector(self, node: ast.AST, table_name: str) -> str:
        if not (isinstance(node, ast.Constant) and isinstance(node.value, str)):
            raise ExpressionError(
                f"The lookup method for table '{table_name}' must be one of "
                '"exact", "truncate", "linear", "spline" as a quoted string'
            )
        method = node.value
        if method not in self._TABLE_METHODS:
            raise ExpressionError(
                f"Unknown lookup method '{method}' for table '{table_name}' "
                "(expected one of exact, truncate, linear, spline)"
            )
        return method

    def _table_default_result_column(
        self, table: Mapping[str, Any], columns: Sequence[str], table_name: str
    ) -> str:
        default_column = table.get("default_value_column")
        if default_column:
            if default_column not in columns:
                raise ExpressionError(
                    f"Table '{table_name}' default result column '{default_column}' does not exist"
                )
            return default_column
        if len(columns) == 2:
            return columns[1]
        raise ExpressionError(
            f"{table_name}[lookup_value] needs a default result column -- set one in "
            f"the table's settings, or specify it explicitly: {table_name}[lookup_value, result_column]"
        )

    def _eval_table_lookup(self, node: ast.Subscript) -> float:
        table_name, table = self._table_reference(node.value)
        columns = list(table.get("columns") or [])
        if not columns:
            raise ExpressionError(f"Table '{table_name}' has no columns")
        rows = table.get("rows") or []

        slice_node = node.slice
        args = list(slice_node.elts) if isinstance(slice_node, ast.Tuple) else [slice_node]
        if not 1 <= len(args) <= 4:
            raise ExpressionError(
                f"{table_name}[...] takes 1 to 4 arguments: lookup_value, "
                "[result_column] or [lookup_column, result_column], [, method]"
            )

        lookup_value = self._numeric(self._eval_node(args[0]))
        lookup_column = columns[0]
        method = "linear"

        if len(args) == 1:
            result_column = self._table_default_result_column(table, columns, table_name)
        elif len(args) == 2:
            result_column = self._table_column_selector(args[1], columns, table_name)
        else:
            lookup_column = self._table_column_selector(args[1], columns, table_name)
            result_column = self._table_column_selector(args[2], columns, table_name)
            if len(args) == 4:
                method = self._table_method_selector(args[3], table_name)

        pairs: list[tuple[float, float]] = []
        for row in rows:
            key_raw = row.get(lookup_column)
            value_raw = row.get(result_column)
            if key_raw is None or value_raw is None:
                continue
            try:
                pairs.append((float(key_raw), float(value_raw)))
            except (TypeError, ValueError):
                continue
        if not pairs:
            raise ExpressionError(
                f"Table '{table_name}' has no usable numeric rows for column "
                f"'{lookup_column}' -> '{result_column}'"
            )
        pairs.sort(key=lambda pair: pair[0])

        return self._table_lookup_value(
            table_name, lookup_column, result_column, pairs, lookup_value, method
        )

    def _table_lookup_value(
        self,
        table_name: str,
        lookup_column: str,
        result_column: str,
        pairs: list[tuple[float, float]],
        lookup_value: float,
        method: str,
    ) -> float:
        keys = [key for key, _ in pairs]
        values = [value for _, value in pairs]

        if method == "exact":
            for key, value in pairs:
                if math.isclose(key, lookup_value, abs_tol=1e-9, rel_tol=1e-9):
                    return value
            available = ", ".join(_format_number(key) for key in keys)
            raise ExpressionError(
                f"No exact match for {_format_number(lookup_value)} in table '{table_name}' "
                f"column '{lookup_column}' (available: {available})"
            )

        if method == "truncate":
            candidates = [key for key in keys if key <= lookup_value]
            target_key = max(candidates) if candidates else keys[0]
            for key, value in pairs:
                if key == target_key:
                    return value

        if method == "linear":
            if lookup_value <= keys[0]:
                return values[0]
            if lookup_value >= keys[-1]:
                return values[-1]
            for (key_low, value_low), (key_high, value_high) in zip(pairs, pairs[1:]):
                if key_low <= lookup_value <= key_high:
                    if key_high == key_low:
                        return value_low
                    weight = (lookup_value - key_low) / (key_high - key_low)
                    return value_low + weight * (value_high - value_low)
            return values[-1]

        if method == "spline":
            try:
                from scipy.interpolate import CubicSpline
            except ImportError as exc:
                raise ExpressionError("Spline table lookup requires scipy") from exc
            unique_keys = sorted(set(keys))
            if len(unique_keys) != len(keys):
                raise ExpressionError(
                    f"Table '{table_name}' column '{lookup_column}' has duplicate values; "
                    "spline lookup requires unique keys"
                )
            if len(pairs) < 3:
                raise ExpressionError(
                    f"Spline lookup on table '{table_name}' requires at least 3 rows"
                )
            clamped = min(max(lookup_value, keys[0]), keys[-1])
            spline = CubicSpline(keys, values)
            return float(spline(clamped))

        raise ExpressionError(f"Unknown lookup method '{method}'")

    def _call_function(self, name: str, args: list[Any]) -> float | bool:
        if name == "quantile":
            if len(args) != 2:
                raise ExpressionError("quantile({x1, x2, ...}, q) requires two arguments")
            return self._linear_quantile(self._sequence(args[0]), self._numeric(args[1]))
        if name in {"mean", "prod", "sd", "sum", "var"}:
            summary_values = self._summary_values(args)
            if name == "mean":
                return statistics.mean(summary_values)
            if name == "prod":
                return math.prod(summary_values)
            if name == "sd":
                return statistics.stdev(summary_values)
            if name == "sum":
                return sum(summary_values)
            return statistics.variance(summary_values)

        values = [self._numeric(argument) for argument in args]
        if name == "abs":
            return abs(values[0])
        if name == "arccos":
            return math.acos(values[0])
        if name == "arcsin":
            return math.asin(values[0])
        if name == "arctan":
            return math.atan(values[0])
        if name == "cbrt":
            return math.copysign(abs(values[0]) ** (1 / 3), values[0])
        if name == "ceil":
            return math.ceil(values[0])
        if name == "comb":
            return math.comb(self._integer(values[0]), self._integer(values[1]))
        if name == "cos":
            return math.cos(values[0])
        if name == "cosh":
            return math.cosh(values[0])
        if name == "erf":
            return math.erf(values[0])
        if name == "exp":
            return math.exp(values[0])
        if name == "factorial":
            return math.factorial(self._integer(values[0]))
        if name == "floor":
            return math.floor(values[0])
        if name == "gamma":
            return math.gamma(values[0])
        if name == "hypot":
            return math.hypot(*values)
        if name == "invErf":
            return statistics.NormalDist().inv_cdf((values[0] + 1) / 2) / math.sqrt(2)
        if name == "lgamma":
            return math.lgamma(values[0])
        if name in {"ln", "log"}:
            return math.log(*values)
        if name == "log10":
            return math.log10(values[0])
        if name == "logit":
            return math.log(values[0] / (1 - values[0]))
        if name == "logis":
            return 1 / (1 + math.exp(-values[0]))
        if name == "max":
            return max(values)
        if name == "min":
            return min(values)
        if name == "ProbToProb":
            return 1 - (1 - values[0]) ** (values[2] / values[1])
        if name == "ProbToRate":
            return -math.log1p(-values[0]) * values[2] / values[1]
        if name == "RateToProb":
            return 1 - math.exp(-values[0] * values[2] / values[1])
        if name == "round":
            # every argument arrives as a float, but round(x, ndigits) needs an integer ndigits
            return round(values[0], int(values[1])) if len(values) > 1 else round(values[0])
        if name == "sin":
            return math.sin(values[0])
        if name == "sinh":
            return math.sinh(values[0])
        if name == "sqrt":
            return math.sqrt(values[0])
        if name == "tan":
            return math.tan(values[0])
        if name == "tanh":
            return math.tanh(values[0])
        if name == "sign":
            return -1.0 if values[0] < 0 else 1.0 if values[0] > 0 else 0.0
        raise ExpressionError(f"Unknown function '{name}'" + _did_you_mean(name, _function_names()))

    @staticmethod
    def _summary_values(args: list[Any]) -> list[float]:
        if len(args) == 1 and isinstance(args[0], (list, tuple)):
            return [float(value) for value in args[0]]
        return [float(value) for value in args]

    @staticmethod
    def _sequence(value: Any) -> list[float]:
        if not isinstance(value, (list, tuple)):
            raise ExpressionError("quantile requires a list or tuple of values")
        return [float(item) for item in value]

    @staticmethod
    def _linear_quantile(values: list[float], q: float) -> float:
        if not values or not 0 <= q <= 1:
            raise ExpressionError("quantile requires values and q in [0, 1]")
        ordered = sorted(values)
        position = (len(ordered) - 1) * q
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return ordered[lower]
        weight = position - lower
        return ordered[lower] + weight * (ordered[upper] - ordered[lower])

    @staticmethod
    def _numeric(value: Any) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ExpressionError(f"Expected a numeric value, got {value!r}")
        return float(value)

    @staticmethod
    def _numeric_or_bool(value: Any) -> float | bool:
        if isinstance(value, bool):
            return value
        return _Evaluator._numeric(value)

    @staticmethod
    def _integer(value: float) -> int:
        if not value.is_integer():
            raise ExpressionError("Expected an integer argument")
        return int(value)


def evaluate_expression(
    value: Any,
    parameters: Mapping[str, Param],
    variables: Mapping[str, Any] | None = None,
    distributions: Mapping[str, Mapping[str, Any]] | None = None,
    tables: Mapping[str, Mapping[str, Any]] | None = None,
) -> float:
    """Evaluate one DECIDE expression against parameters and variables."""

    return _Evaluator(parameters, variables, distributions, tables).evaluate(value)


def resolve_variables(
    parameters: Mapping[str, Param],
    variables: Mapping[str, Any] | None = None,
    distributions: Mapping[str, Mapping[str, Any]] | None = None,
    tables: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Param]:
    """Return parameters plus evaluated variables as a single value namespace."""

    resolved = dict(parameters)
    if not variables:
        return resolved
    evaluator = _Evaluator(resolved, variables, distributions, tables)
    for name in variables:
        if name in resolved:
            raise ExpressionError(f"Variable '{name}' conflicts with a parameter")
        resolved[name] = Param(name=name, base=evaluator._resolve_name(name))
    return resolved


def resolve_parameter_definitions(
    parameters: Mapping[str, Param],
    definitions: Mapping[str, Any] | None = None,
    distributions: Mapping[str, Mapping[str, Any]] | None = None,
    tables: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Param]:
    """Resolve parameters whose definitions are expressions, then variables."""

    definitions = definitions or {}
    direct_parameters = {
        name: parameter for name, parameter in parameters.items() if name not in definitions
    }
    resolved = resolve_variables(direct_parameters, definitions, distributions, tables)
    for name, parameter in parameters.items():
        if name in definitions:
            calculated = resolved[name]
            resolved[name] = Param(
                name=name,
                base=calculated.base,
                low=parameter.low,
                high=parameter.high,
                dist=parameter.dist,
                dist_args=parameter.dist_args,
                source=parameter.source,
            )
    return resolved


def resolve_variables_partial(
    parameters: Mapping[str, Param],
    variables: Mapping[str, Any] | None = None,
    distributions: Mapping[str, Mapping[str, Any]] | None = None,
    tables: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[dict[str, Param], dict[str, str]]:
    """Like resolve_variables, but one broken variable does not block the rest.

    Returns ``(resolved, errors)``: every variable that evaluated
    successfully is in ``resolved`` alongside the input parameters; every
    variable that raised is left out of ``resolved`` and recorded in
    ``errors`` by name, with the message. A variable that depends on a
    broken one still fails (and is recorded too) -- only genuinely
    unrelated variables are unaffected. Intended for preview/UI code that
    wants each field's own result independently; resolve_variables() above
    is still what an actual model run uses, since a broken formula there
    should fail loudly rather than silently produce nothing.
    """

    resolved = dict(parameters)
    errors: dict[str, str] = {}
    if not variables:
        return resolved, errors
    evaluator = _Evaluator(resolved, variables, distributions, tables)
    for name in variables:
        if name in resolved:
            errors[name] = f"Variable '{name}' conflicts with a parameter"
            continue
        try:
            resolved[name] = Param(name=name, base=evaluator._resolve_name(name))
        except ValueError as exc:
            # Catches ExpressionError (bad formula) and DistributionError
            # (e.g. "Distribution functions require scipy") alike -- both
            # are ValueError subclasses, and either one should only take
            # this one field out, never crash the whole page.
            errors[name] = str(exc)
    return resolved, errors


def resolve_parameter_definitions_partial(
    parameters: Mapping[str, Param],
    definitions: Mapping[str, Any] | None = None,
    distributions: Mapping[str, Mapping[str, Any]] | None = None,
    tables: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[dict[str, Param], dict[str, str]]:
    """Like resolve_parameter_definitions, but one broken definition does
    not blank out every other parameter -- see resolve_variables_partial."""

    definitions = definitions or {}
    direct_parameters = {
        name: parameter for name, parameter in parameters.items() if name not in definitions
    }
    resolved, errors = resolve_variables_partial(direct_parameters, definitions, distributions, tables)
    for name, parameter in parameters.items():
        if name in definitions and name in resolved:
            calculated = resolved[name]
            resolved[name] = Param(
                name=name,
                base=calculated.base,
                low=parameter.low,
                high=parameter.high,
                dist=parameter.dist,
                dist_args=parameter.dist_args,
                source=parameter.source,
            )
    return resolved, errors


def build_tables_mapping(
    raw_tables: Sequence[Mapping[str, Any]] | None,
) -> dict[str, dict[str, Any]]:
    """Convert the document's ``tables`` list (as edited in the app's Tables
    UI -- each entry shaped like ``{"name", "data", "columns"?,
    "default_value_column"?}``) into the ``{table_name: {"columns": [...],
    "rows": [...], "default_value_column": ...}}`` shape the expression
    evaluator's ``table_name[...]`` Lookup syntax expects.

    ``columns`` is taken from the table's own explicit "columns" field when
    present (the source of truth, since dict-key order from a 0-row table
    can't be relied on); otherwise it's derived from the union of keys seen
    across "data" rows, in first-seen order, for backward compatibility with
    tables saved before that field existed.
    """

    tables: dict[str, dict[str, Any]] = {}
    for raw in raw_tables or []:
        name = str(raw.get("name") or "").strip()
        if not name:
            continue
        rows = list(raw.get("data") or [])
        columns = [str(column) for column in (raw.get("columns") or [])]
        if not columns:
            seen: list[str] = []
            for row in rows:
                for key in row.keys():
                    if key not in seen:
                        seen.append(key)
            columns = seen
        tables[name] = {
            "columns": columns,
            "rows": rows,
            "default_value_column": raw.get("default_value_column"),
        }
    return tables


def _function_names() -> list[str]:
    return sorted(n.rstrip("_") if n == "if_" else n for n in _FUNCTION_NAMES)


def _did_you_mean(name: str, candidates: Any) -> str:
    """' Did you mean ...?' hint for a mistyped name (case-insensitive match first)."""
    import difflib
    pool = sorted({str(c) for c in candidates if str(c) != name})
    if not pool:
        return ""
    lowered = {c.lower(): c for c in pool}
    if name.lower() in lowered:
        return f". Did you mean '{lowered[name.lower()]}'? (names are case-sensitive)"
    close = difflib.get_close_matches(name, pool, n=3, cutoff=0.6)
    if not close:
        close = [c for c in pool if name.lower() in c.lower() or c.lower() in name.lower()][:3] if len(name) >= 3 else []
    if not close:
        return ""
    return ". Did you mean " + " or ".join(f"'{c}'" for c in close) + "?"
