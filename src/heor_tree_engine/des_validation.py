"""Structural validation for DES models (see des_models.py)."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from .des_models import (
    NAMED_DISTRIBUTION_CLOSED_FORMS,
    NAMED_DISTRIBUTION_UNSUITABLE,
    DES_CLOCKS,
    DES_TTE_PARAMS,
    DESChance,
    DESLogic,
    DESModel,
    DESNode,
    DESTerminal,
    TimeNode,
    TimeToEvent,
    ValidationError,
)
from .des_sampling import build_table_cdf
from .des_models import ELSE_MARKERS
from .models import Param, is_complement

# A table referenced by TableToTime must have an identifier-style name
# (letters, digits, "_"; no spaces) -- e.g. life_table, never "life table"
# (a naming rule; also what table_name[...] Lookup needs).
TABLE_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

_POSITIVE_PARAMS = {
    "exponential": ("rate",),
    "weibull": ("shape", "scale"),
    "log_normal": ("sdlog",),
    "log_logistic": ("shape", "scale"),
    "gompertz": ("rate",),
    "generalized_gamma": ("a", "scale"),
}


def _check_probability_group(
    owner_id: str,
    branches,
    errors: list[ValidationError],
) -> None:
    numeric_sum = 0.0
    all_numeric = True
    complements = 0
    for branch in branches:
        value = branch.probability
        if value is None:
            errors.append(ValidationError(
                "missing_probability", f"Branch '{branch.label}' needs a probability", owner_id, branch.label,
            ))
            all_numeric = False
        elif is_complement(value):
            complements += 1
        elif isinstance(value, (int, float)):
            if not 0.0 <= value <= 1.0:
                errors.append(ValidationError(
                    "probability_out_of_range", f"Branch '{branch.label}' probability must be in [0, 1]",
                    owner_id, branch.label,
                ))
            numeric_sum += float(value)
        else:
            all_numeric = False  # an expression: checked at run time
    if complements > 1:
        errors.append(ValidationError(
            "multiple_complements", "Only one branch per node may use the complement '#'", owner_id,
        ))
    if all_numeric and complements == 0 and branches and abs(numeric_sum - 1.0) > 1e-9:
        errors.append(ValidationError(
            "probabilities_do_not_sum_to_one", f"Probabilities sum to {numeric_sum:g}, not 1", owner_id,
        ))
    if all_numeric and complements == 1 and numeric_sum > 1.0 + 1e-9:
        errors.append(ValidationError(
            "probabilities_exceed_one", f"Probabilities already sum to {numeric_sum:g} before '#'", owner_id,
        ))


def _check_tte(
    node_id: str,
    label: str,
    tte: TimeToEvent,
    tables: Mapping[str, Mapping[str, Any]],
    errors: list[ValidationError],
    distributions: Mapping[str, Mapping[str, Any]] | None = None,
) -> None:
    required = DES_TTE_PARAMS.get(tte.distribution)
    if required is None:
        errors.append(ValidationError(
            "unknown_distribution", f"Unknown time-to-event distribution '{tte.distribution}'", node_id, label,
        ))
        return
    missing = [name for name in required if tte.params.get(name) in (None, "")]
    if missing:
        errors.append(ValidationError(
            "missing_tte_parameter",
            f"Event '{label}' ({tte.distribution}) is missing: {', '.join(missing)}", node_id, label,
        ))
        return
    if tte.clock not in DES_CLOCKS:
        errors.append(ValidationError(
            "invalid_clock", f"clock must be one of {DES_CLOCKS}", node_id, label,
        ))
    for name in _POSITIVE_PARAMS.get(tte.distribution, ()):
        value = tte.params[name]
        if isinstance(value, (int, float)) and value <= 0:
            errors.append(ValidationError(
                "non_positive_parameter", f"Event '{label}': '{name}' must be > 0", node_id, label,
            ))
    if tte.distribution == "fixed" and isinstance(tte.params["time"], (int, float)) and tte.params["time"] < 0:
        errors.append(ValidationError(
            "negative_time", f"Event '{label}': fixed time must be >= 0", node_id, label,
        ))
    if tte.distribution == "named":
        name = str(tte.params["name"])
        definition = (distributions or {}).get(name)
        if definition is None:
            errors.append(ValidationError(
                "unknown_distribution_name",
                f"Event '{label}' uses distribution '{name}', which is not in the Distributions section",
                node_id, label,
            ))
        elif str(definition.get("type", "")) in NAMED_DISTRIBUTION_UNSUITABLE:
            errors.append(ValidationError(
                "unsuitable_distribution",
                f"Event '{label}': distribution '{name}' is {definition.get('type')} -- a time-to-event needs a "
                "continuous distribution (e.g. Weibull, Exponential, Gompertz, Lognormal)",
                node_id, label,
            ))
        else:
            closed = NAMED_DISTRIBUTION_CLOSED_FORMS.get(str(definition.get("type", "")))
            raw_params = definition.get("parameters") or {}
            missing = [p for p in (closed[1] if closed else ()) if raw_params.get(p) in (None, "")]
            if missing:
                errors.append(ValidationError(
                    "missing_distribution_parameter",
                    f"Distribution '{name}' ({definition.get('type')}) is missing: {', '.join(missing)}",
                    node_id, label,
                ))
        return
    if tte.distribution == "table_to_time":
        table_name = str(tte.params["table"])
        if not TABLE_NAME_PATTERN.match(table_name):
            errors.append(ValidationError(
                "invalid_table_name",
                f"Event '{label}': table name '{table_name}' is not allowed -- use letters, digits and '_' "
                f"only, no spaces (e.g. '{re.sub(r'[^A-Za-z0-9_]+', '_', table_name.strip()) or 'life_table'}')",
                node_id, label,
            ))
            return
        table = tables.get(table_name)
        if table is None:
            errors.append(ValidationError(
                "unknown_table", f"Event '{label}' references unknown table '{table_name}'", node_id, label,
            ))
            return
        try:
            build_table_cdf(
                table.get("rows", []), str(tte.params["index_column"]), str(tte.params["value_column"]),
                tte.prob_type, tte.interpolate,
            )
        except ValueError as exc:
            errors.append(ValidationError("invalid_table_to_time", f"Event '{label}': {exc}", node_id, label))


def validate_des(
    model: DESModel,
    parameters: Mapping[str, Param] | None = None,
    tables: Mapping[str, Mapping[str, Any]] | None = None,
    distributions: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[ValidationError]:
    errors: list[ValidationError] = []
    tables = tables or {}
    des: DESNode = model.des
    nodes = model.nodes

    if des.n_patients < 1:
        errors.append(ValidationError("invalid_n_patients", "N patients must be at least 1", des.id, "n_patients"))
    if des.trace_step <= 0:
        errors.append(ValidationError("invalid_trace_step", "Trace step must be > 0", des.id, "trace_step"))
    if isinstance(des.time_horizon, (int, float)) and des.time_horizon <= 0:
        errors.append(ValidationError("invalid_time_horizon", "Time horizon must be > 0", des.id, "time_horizon"))
    if des.event_log_sample_size < 0:
        errors.append(ValidationError(
            "invalid_event_log_sample_size", "Event log sample size cannot be negative", des.id,
        ))

    tracker_names = [tracker.name for tracker in des.trackers]
    if len(tracker_names) != len(set(tracker_names)):
        errors.append(ValidationError("duplicate_tracker", "Tracker names must be unique", des.id, "trackers"))
    known_trackers = set(tracker_names)

    if not des.branches:
        errors.append(ValidationError("no_starting_state", "A DES node needs at least one starting state", des.id))
    _check_probability_group(des.id, des.branches, errors)
    for branch in des.branches:
        if not isinstance(nodes.get(branch.child_id), (TimeNode, DESTerminal)):
            errors.append(ValidationError(
                "invalid_des_child", f"DES branch '{branch.label}' must lead to a Time node or a Terminal node",
                des.id, branch.label,
            ))

    for node in nodes.values():
        node_unknown = sorted(set(getattr(node, "tracker_updates", {}) or {}) - known_trackers)
        if node_unknown:
            errors.append(ValidationError(
                "unknown_tracker", f"Node '{node.id}' updates unknown tracker(s): {', '.join(node_unknown)}", node.id,
            ))
        for branch in getattr(node, "branches", []):
            unknown = sorted(set(branch.tracker_updates) - known_trackers)
            if unknown:
                errors.append(ValidationError(
                    "unknown_tracker", f"Branch '{branch.label}' updates unknown tracker(s): {', '.join(unknown)}",
                    node.id, branch.label,
                ))

        if isinstance(node, TimeNode):
            labels = [branch.label for branch in node.branches]
            if len(labels) != len(set(labels)):
                errors.append(ValidationError(
                    "duplicate_event_label", "Event (branch) labels must be unique within a Time node", node.id,
                ))
            for branch in node.branches:
                if branch.time_to_event is None:
                    errors.append(ValidationError(
                        "missing_time_to_event", f"Event '{branch.label}' needs a time-to-event", node.id, branch.label,
                    ))
                else:
                    _check_tte(node.id, branch.label, branch.time_to_event, tables, errors, distributions)
                if not isinstance(nodes.get(branch.child_id), (DESChance, DESLogic, DESTerminal)):
                    errors.append(ValidationError(
                        "invalid_event_child",
                        f"Event '{branch.label}' must lead to a Chance, Logic or Terminal node",
                        node.id, branch.label,
                    ))
        elif isinstance(node, DESChance):
            if not node.branches:
                errors.append(ValidationError("empty_chance", "A Chance node needs branches", node.id))
            _check_probability_group(node.id, node.branches, errors)
            for branch in node.branches:
                if not isinstance(nodes.get(branch.child_id), (DESChance, DESLogic, DESTerminal)):
                    errors.append(ValidationError(
                        "invalid_chance_child",
                        f"Chance branch '{branch.label}' must lead to a Chance, Logic or Terminal node "
                        "(use a Terminal with 'jump to' to go back to a state)",
                        node.id, branch.label,
                    ))
        elif isinstance(node, DESLogic):
            if not node.branches:
                errors.append(ValidationError("empty_logic", "A Logic node needs branches", node.id))
            else_count = 0
            for branch in node.branches:
                condition = branch.condition
                if condition in (None, ""):
                    errors.append(ValidationError(
                        "missing_condition",
                        f"Logic branch '{branch.label}' needs a condition (e.g. n_relapse >= 3) or '#' for 'else'",
                        node.id, branch.label,
                    ))
                elif isinstance(condition, str) and condition.strip().lower() in ELSE_MARKERS:
                    else_count += 1
                if not isinstance(nodes.get(branch.child_id), (DESChance, DESLogic, DESTerminal)):
                    errors.append(ValidationError(
                        "invalid_logic_child",
                        f"Logic branch '{branch.label}' must lead to a Chance, Logic or Terminal node",
                        node.id, branch.label,
                    ))
            if else_count > 1:
                errors.append(ValidationError("multiple_else", "Only one Logic branch may be 'else' (#)", node.id))
        elif isinstance(node, DESTerminal) and node.jump_to is not None:
            if not isinstance(nodes.get(node.jump_to), TimeNode):
                errors.append(ValidationError(
                    "invalid_jump_target", f"Terminal '{node.id}' jumps to '{node.jump_to}', which is not a Time node "
                    "in this DES model", node.id, "jump_to",
                ))
    return errors
