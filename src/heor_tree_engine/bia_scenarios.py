"""One scenario mechanism for budget impact models.

A scenario is a name plus a list of changes to the model's inputs:

    {"name": "Lower price, faster uptake",
     "overrides": [
        {"param": "@price",  "mode": "multiply", "value": 0.8},
        {"param": "new_mix_share/drug_new", "mode": "set", "value": 0.4, "years": "3-5"},
        {"param": "comparators/drug_a/cost_acquisition", "mode": "add", "value": 500}]}

* ``param`` is a path into the document (``/``-separated; a segment that names an item of a list matches the item whose
  ``id`` or ``name`` is that text) or one of the shortcuts ``@price`` (the new technology's price), ``@uptake`` (its uptake;
  the other treatments give up / take back the difference), and ``@population``.
* ``mode`` is ``set`` (replace), ``multiply`` or ``add``.
* ``years`` only matters when the parameter is a per-year list: empty = all years, or e.g. ``1,3`` / ``2-4``.

``list_parameters`` offers every changeable input of the current model (for the picker); ``apply_overrides`` builds the changed
copy of the document; ``run_scenarios`` and ``tornado`` run the budget impact calculation on it.  Pure Python (no Streamlit).
"""

from __future__ import annotations

import copy
from typing import Any, Callable

MODES = ("set", "multiply", "add")
SYNTHETIC = {
    "@price": "Price of the new technology (all cost categories)",
    "@uptake": "Uptake of the new technology (others give up the difference)",
    "@population": "Eligible population (all years)",
}
GENERIC_COSTS = ("cost_acquisition", "cost_administration", "cost_monitoring", "cost_adverse_events", "cost_disease_management")


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


# ---------------------------------------------------------------- paths
def _step(node: Any, segment: str) -> tuple[Any, Any]:
    """(container, key) addressing ``segment`` inside ``node`` (dict key or list item), or (None, None)."""

    if isinstance(node, dict):
        return (node, segment) if segment in node else (None, None)
    if isinstance(node, list):
        for index, item in enumerate(node):
            if isinstance(item, dict) and segment in (str(item.get("id")), str(item.get("name"))):
                return node, index
        if segment.lstrip("-").isdigit() and -len(node) <= int(segment) < len(node):
            return node, int(segment)
    return None, None


def resolve(document: dict[str, Any], path: str) -> tuple[Any, Any]:
    """(container, key) of the value at ``path`` or (None, None) when it does not exist."""

    container, key = None, None
    node: Any = document
    for segment in [s for s in path.split("/") if s]:
        container, key = _step(node, segment)
        if container is None:
            return None, None
        node = container[key]
    return container, key


def parse_years(text: Any, n: int) -> list[int]:
    """'' / 'all' -> every year; '1,3' or '2-4' -> zero-based indices inside 0..n-1."""

    raw = str(text or "").strip().lower()
    if raw in ("", "all", "*"):
        return list(range(n))
    out: set[int] = set()
    for part in raw.replace(";", ",").split(","):
        part = part.strip()
        if "-" in part:
            a, _, b = part.partition("-")
            if a.strip().isdigit() and b.strip().isdigit():
                out.update(range(int(a) - 1, int(b)))
        elif part.isdigit():
            out.add(int(part) - 1)
    return sorted(i for i in out if 0 <= i < n)


def _change(old: float, mode: str, value: float) -> float:
    if mode == "multiply":
        return old * value
    if mode == "add":
        return old + value
    return value


def apply_one(document: dict[str, Any], override: dict[str, Any], multipliers: dict[str, float]) -> None:
    """Apply one override in place (the document must already be a copy). Shortcuts only accumulate a multiplier."""

    param = str(override.get("param") or "")
    mode = str(override.get("mode") or "multiply")
    try:
        value = float(override.get("value"))
    except (TypeError, ValueError):
        return
    if param.startswith("@"):
        key = param[1:]
        current = multipliers.get(key, 1.0)
        multipliers[key] = current * value if mode == "multiply" else (value if mode == "set" else current + value)
        return
    container, key = resolve(document, param)
    if container is None:
        return
    target = container[key]
    if _is_number(target):
        container[key] = _change(float(target), mode, value)
    elif isinstance(target, list):
        for i in parse_years(override.get("years"), len(target)):
            if _is_number(target[i]) or target[i] in (None, ""):
                target[i] = _change(float(target[i] or 0.0), mode, value)


def apply_overrides(document: dict[str, Any], overrides: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, float]]:
    """(changed deep copy, {'price': x, 'uptake': y, ...} shortcut multipliers). Shortcuts are applied after the path edits."""

    changed = copy.deepcopy(document)
    multipliers: dict[str, float] = {}
    for override in overrides or []:
        apply_one(changed, override, multipliers)
    return changed, multipliers


# ---------------------------------------------------------------- parameter list (for the picker)
def list_parameters(document: dict[str, Any], template: str | None = None) -> list[tuple[str, str]]:
    """[(path, label)] of every input a scenario can change in this model."""

    out: list[tuple[str, str]] = list(SYNTHETIC.items())
    out.append(("eligible_population", "Eligible population (choose years)"))
    for comparator in document.get("comparators") or []:
        cid, label = comparator.get("id"), comparator.get("label") or comparator.get("id")
        for field in GENERIC_COSTS:
            out.append((f"comparators/{cid}/{field}", f"{label}: {field.replace('cost_', '').replace('_', ' ')} cost per patient"))
        out.append((f"current_mix_share/{cid}", f"{label}: market share, current mix (choose years)"))
        out.append((f"new_mix_share/{cid}", f"{label}: market share, new mix (choose years)"))
        if comparator.get("cost_by_treatment_year"):
            out.append((f"comparators/{cid}/cost_by_treatment_year", f"{label}: cost per patient by year of treatment (model profile; choose years)"))
    return out


def label_for(document: dict[str, Any], template: str | None, param: str) -> str:
    return dict(list_parameters(document, template)).get(param, param)


# ---------------------------------------------------------------- running
RunFn = Callable[[dict[str, Any], dict[str, float]], "dict[str, Any] | None"]


def scenario_document(document: dict[str, Any], price_multiplier: float = 1.0, uptake_multiplier: float = 1.0,
                      population_multiplier: float = 1.0) -> dict[str, Any]:
    """Copy of the document with the new technology's price, its uptake and the eligible population scaled.
    Uptake: the new technology's share is multiplied (capped at the total covered share of that year) and the
    comparators give up / take back the difference in proportion to their own new-mix shares."""

    from .bia import COST_CATEGORIES

    out = copy.deepcopy(document)
    for comparator in out.get("comparators") or []:
        if comparator.get("is_new"):
            for category in COST_CATEGORIES:
                if comparator.get(category) is not None:
                    comparator[category] = float(comparator[category]) * price_multiplier
            if comparator.get("cost_by_treatment_year"):
                comparator["cost_by_treatment_year"] = [float(v or 0.0) * price_multiplier for v in comparator["cost_by_treatment_year"]]
    out["eligible_population"] = [float(p or 0.0) * population_multiplier for p in out.get("eligible_population") or []]
    new_id = next((c.get("id") for c in out.get("comparators") or [] if c.get("is_new")), None)
    shares = out.get("new_mix_share") or {}
    if new_id is not None and new_id in shares and uptake_multiplier != 1.0:
        horizon = int(out.get("time_horizon_years") or 0)
        for y in range(horizon):
            total = sum(float((v[y] if y < len(v) else 0.0) or 0.0) for v in shares.values())
            old_new = float((shares[new_id][y] if y < len(shares[new_id]) else 0.0) or 0.0)
            wanted = min(old_new * uptake_multiplier, total)
            others_total = total - old_new
            if others_total <= 0:
                continue
            factor = (total - wanted) / others_total
            for cid, series in shares.items():
                if y >= len(series):
                    continue
                series[y] = wanted if cid == new_id else float(series[y] or 0.0) * factor
    return out


def generic_run(document: dict[str, Any], multipliers: dict[str, float]) -> dict[str, Any] | None:
    """Budget impact calculation: {'years': net budget impact per year, 'total': ...}."""

    from .bia import calculate_budget_impact

    doc = document
    if any(abs(multipliers.get(k, 1.0) - 1.0) > 1e-12 for k in ("price", "uptake", "population")):
        doc = scenario_document(document, multipliers.get("price", 1.0), multipliers.get("uptake", 1.0), multipliers.get("population", 1.0))
    result, errors = calculate_budget_impact(doc)
    if errors:
        return None
    return {"years": [y.budget_impact for y in result.years], "total": result.total_budget_impact}


def run_for(template: str | None = None) -> RunFn:
    return generic_run


def run_scenarios(document: dict[str, Any], scenarios: list[dict[str, Any]], template: str | None = None) -> list[dict[str, Any]]:
    """Base case first, then every named scenario: [{'name', 'years', 'total', ...}]."""

    run = run_for(template)
    rows: list[dict[str, Any]] = []
    base = run(document, {})
    if base is not None:
        rows.append({"name": "Base case", **base})
    for scenario in scenarios or []:
        name = str(scenario.get("name") or "").strip()
        if not name:
            continue
        changed, multipliers = apply_overrides(document, scenario.get("overrides") or [])
        result = run(changed, multipliers)
        if result is not None:
            rows.append({"name": name, **result})
    return rows


def tornado(document: dict[str, Any], params: list[str], pct: float = 20.0, template: str | None = None) -> list[dict[str, Any]]:
    """One-way sensitivity on the total: each parameter x (1 - pct%) and x (1 + pct%); sorted by swing, largest first."""

    run = run_for(template)
    base = run(document, {})
    if base is None:
        return []
    rows = []
    for param in params:
        totals = []
        for factor in (1.0 - pct / 100.0, 1.0 + pct / 100.0):
            changed, multipliers = apply_overrides(document, [{"param": param, "mode": "multiply", "value": factor}])
            result = run(changed, multipliers)
            totals.append(result["total"] if result else float("nan"))
        rows.append({"param": param, "low": totals[0], "high": totals[1], "base": base["total"],
                     "swing": abs(totals[1] - totals[0])})
    return sorted(rows, key=lambda r: -(r["swing"] if r["swing"] == r["swing"] else 0.0))


def preset_scenarios() -> list[dict[str, Any]]:
    """The three scenarios the old BIA panel offered, as ordinary scenarios the person can edit."""

    return [
        {"name": "Optimistic (new technology price -10%)", "overrides": [{"param": "@price", "mode": "multiply", "value": 0.9}]},
        {"name": "Pessimistic (new technology price +10%)", "overrides": [{"param": "@price", "mode": "multiply", "value": 1.1}]},
        {"name": "Higher uptake (x1.5)", "overrides": [{"param": "@uptake", "mode": "multiply", "value": 1.5}]},
    ]
