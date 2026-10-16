"""Export a DECIDE Microsimulation model as a standalone script.

``generate_microsim_export_code(document, language, toolkit)`` returns the
source of a self-contained patient-level simulation in Python, R, Stata
(Mata) or JavaScript.  The generated script simulates N patients one by one
through the model horizon for every strategy and prints the mean discounted
cost and effect (with Monte Carlo standard errors) and the ICER of each
strategy versus the first one.

This module is deliberately free of streamlit (and of numpy): the formula
transpiler and the Parameter/Variable context builders live in app.py next
to the Tree/Markov/PSM exporters, and are handed in through ``toolkit``
(``app.py`` passes ``globals()``; tests pass a namespace extracted from
app.py).  What is needed from the toolkit:

    _export_transpile, _ExportNamer, _export_context, _export_names_referenced,
    _export_freeze_resolver, _export_v_access, _is_complement_value, _num,
    _per_cycle_rate, _EXPORT_HELPER_SOURCE

Engine semantics mirrored (see microsim_engine.run_microsim):
  * per patient and cycle: resolve outgoing probabilities and cost/effect in
    the (state, cycle, tenure, tracker values) context, sample the
    transition (one uniform per hop; probabilities clipped at 0 and
    normalised), follow branch / terminal-leaf nodes down to the real state
    the cycle ends in while adding each hop's one-time cost/effect increment;
  * tenure-based cost (initial / recurring / terminal) blended with the exit
    probability exactly like the engine;
  * trackers: update formulas see the start-of-cycle context, reset rules
    apply on entry or one cycle later (tracker_reset_timing);
  * discounting per cycle, half-cycle correction (trapezoid on the last
    point), cycles_per_year conversion of annual discount rates.

Not exported (an explicit ValueError instead of wrong code): PSM-kind nodes,
Distribution rows used in formulas, stochastic functions (DistSample, ...),
loops among branch nodes.  Silently ignored but listed in the script's
header: extra outcomes beyond cost/effect, PSA settings, patient traces.
"""

from __future__ import annotations

import math
import re
from typing import Any

SUPPORTED_LANGUAGES = ("python", "r", "stata", "javascript")

_STOCHASTIC_FUNCTIONS = ("DistSample", "DistToTime", "TableToTime", "DistTrim")

# Names the generated scripts use themselves; a Parameter/Variable/Tracker
# with one of these names is renamed by the identifier sanitiser.
_RESERVED_NAMES = {
    "cycle", "tenure", "trk", "v", "state", "node", "rows", "exit_p", "rng", "S", "k", "i", "j", "n", "u",
    "math", "random", "sys", "Math", "process", "model_variables", "tracker_update", "prepare_row", "draw",
    "evaluate_context", "leave_probability", "simulate_patient", "run_strategy", "main", "report", "icer_text",
    "N_PATIENTS", "SEED", "CYCLES", "HALF_CYCLE_CORRECTION", "DISCOUNT_COST", "DISCOUNT_EFFECT",
    "RESET_ON_ENTRY", "EFFECT_SIGN", "TRACKERS", "TRACKER_INITIAL", "STRATEGIES", "N_TRACKERS",
    "mulberry32", "cache", "ctx", "raw", "cost", "effect", "total_cost", "total_effect",
}


def _comment_prefix(language: str) -> str:
    return {"python": "#", "r": "#", "javascript": "//", "stata": "//"}[language]


# ---------------------------------------------------------------------------
# IR
# ---------------------------------------------------------------------------


def _strategies_of(document: dict[str, Any]) -> list[dict[str, Any]]:
    strategies = document.get("strategies")
    if strategies:
        return [dict(s) for s in strategies]
    if document.get("states"):
        return [{
            "id": document.get("id", "strategy_1"),
            "name": document.get("name", "Strategy 1"),
            "states": document["states"],
            "transitions": document.get("transitions", []),
        }]
    raise ValueError("The Microsimulation model has no strategy with states to export.")


def _expr_texts(value: Any) -> list[str]:
    return [value] if isinstance(value, str) else []


def _build_strategy_ir(strategy: dict[str, Any], k: int) -> dict[str, Any]:
    sid = str(strategy.get("id", f"strategy_{k}"))
    sname = str(strategy.get("name") or sid)
    if strategy.get("canvas_kind") == "psm":
        raise ValueError(
            f"Strategy '{sname}': its root is a PSM node; exporting PSM-kind microsimulation nodes is not supported."
        )
    states = list(strategy.get("states", []))
    for s in states:
        if s.get("canvas_kind") == "psm":
            raise ValueError(
                f"Strategy '{sname}': state '{s.get('name') or s.get('id')}' is a PSM node; exporting PSM-kind "
                f"microsimulation nodes is not supported (use the PSM editor's export, or replace the node)."
            )
    if not states:
        raise ValueError(f"Strategy '{sname}' has no states.")
    ids = [str(s["id"]) for s in states]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Strategy '{sname}' has a duplicate state id.")
    index = {sid_: i for i, sid_ in enumerate(ids)}
    real = [not s.get("canvas_parent") for s in states]
    if not any(real):
        raise ValueError(f"Strategy '{sname}' has no real (non-branch) state.")

    grouped: dict[str, list[dict[str, Any]]] = {i: [] for i in ids}
    for t in strategy.get("transitions", []):
        frm, to = str(t["from_state"]), str(t["to_state"])
        if frm not in index or to not in index:
            raise ValueError(f"Strategy '{sname}': transition {frm} -> {to} refers to an unknown state.")
        grouped[frm].append(t)

    real_by_name = {str(s.get("name", s["id"])): str(s["id"]) for s, r in zip(states, real) if r}
    alias: list[int | None] = []
    for s, r in zip(states, real):
        if r or grouped[str(s["id"])]:
            alias.append(None)
            continue
        target = s.get("canvas_alias_of") or real_by_name.get(str(s.get("name", s["id"])))
        if not target or str(target) not in index or not real[index[str(target)]]:
            raise ValueError(
                f"Strategy '{sname}': branch/terminal node '{s.get('name') or s['id']}' has no outgoing "
                f"transition and doesn't link to a real state (use 'relabel as terminal' on it)."
            )
        alias.append(index[str(target)])

    # A chain of branch nodes must reach a real state (no loop among them).
    colour: dict[str, int] = {}

    def acyclic(node: str) -> None:
        colour[node] = 1
        for t in grouped[node]:
            to = str(t["to_state"])
            if real[index[to]]:
                continue
            if colour.get(to) == 1:
                raise ValueError(
                    f"Strategy '{sname}': loop entirely among branch nodes ('{to}' is reached again without "
                    f"passing through a real state)."
                )
            if colour.get(to) is None:
                acyclic(to)
        colour[node] = 2

    for s, r in zip(states, real):
        if not r and colour.get(str(s["id"])) is None:
            acyclic(str(s["id"]))

    branch_nodes = [i for i, (s, r) in enumerate(zip(states, real)) if not r and grouped[str(s["id"])]]
    tenure_states = [i for i, (s, r) in enumerate(zip(states, real)) if r and s.get("tenure_applied")]
    return {
        "id": sid, "name": sname, "states": states, "ids": ids, "index": index, "real": real,
        "grouped": grouped, "alias": alias, "branch_nodes": branch_nodes, "tenure_states": tenure_states,
    }


def _microsim_export_ir(document: dict[str, Any], tk: dict[str, Any]) -> dict[str, Any]:
    strategies = [_build_strategy_ir(s, k) for k, s in enumerate(_strategies_of(document), start=1)]

    parameters, variables_raw, distributions, tables = tk["_export_context"](document)
    all_trackers = [dict(t) for t in document.get("trackers") or []]
    # A Tracker row whose name is still empty is an unfinished draft row in the editor: skipped.
    trackers = [t for t in all_trackers if str(t.get("name", "")).strip()]
    tracker_names = [str(t["name"]).strip() for t in trackers]
    if len(set(tracker_names)) != len(tracker_names):
        raise ValueError("Duplicate tracker name.")
    for name in tracker_names:
        if name in parameters or name in variables_raw:
            raise ValueError(f"Tracker '{name}' has the same name as a Parameter/Variable.")
    reset_timing = str(document.get("tracker_reset_timing", "on_entry"))
    if reset_timing not in ("on_entry", "next_cycle"):
        raise ValueError(f"tracker_reset_timing must be 'on_entry' or 'next_cycle', got '{reset_timing}'.")

    # Every formula in the model, for the guard checks below.
    formulas: list[Any] = [v for v in variables_raw.values()]
    for t in trackers:
        formulas += [t.get("initial_value", 0.0), t.get("update_formula")] + list((t.get("reset_rules") or {}).values())
    for s in strategies:
        for st in s["states"]:
            formulas += [st.get(f) for f in ("cost", "effect", "initial_prob", "initial_cost", "recurring_cost", "terminal_cost")]
        for group in s["grouped"].values():
            for t in group:
                formulas += [t.get("probability"), t.get("cost_increment"), t.get("effect_increment")]
    for p in document.get("parameters", []):
        formulas.append(p.get("definition"))
    for text in (f for f in formulas if isinstance(f, str)):
        for fname in _STOCHASTIC_FUNCTIONS:
            if fname in text:
                raise ValueError(
                    f"Formula '{text}' uses {fname}(...), a random draw per call; the exporter can't reproduce "
                    f"that (it would need the engine's per-patient random streams)."
                )
        if distributions:
            used = tk["_export_names_referenced"](text) & set(distributions)
            if used:
                raise ValueError(
                    f"Formula '{text}' uses the Distribution '{sorted(used)[0]}' (resampled per model/patient/cycle); "
                    f"exporting Distribution rows is not supported."
                )

    # Dependency order of Variables + which ones are dynamic (depend,
    # directly or through other Variables, on _cycle/_tenure/a Tracker).
    special = {"_cycle", "_tenure"} | set(tracker_names)
    deps = {
        name: (tk["_export_names_referenced"](expr) & (set(variables_raw) | special)) - {name}
        for name, expr in variables_raw.items()
    }
    order: list[str] = []
    visiting: set[str] = set()
    done: set[str] = set()

    def visit(name: str) -> None:
        if name in done:
            return
        if name in visiting:
            raise ValueError(f"Circular variable reference involving '{name}'")
        visiting.add(name)
        for dep in deps.get(name, ()):
            if dep in variables_raw:
                visit(dep)
        visiting.discard(name)
        done.add(name)
        order.append(name)

    for name in variables_raw:
        visit(name)
    dynamic: set[str] = set()
    changed = True
    while changed:
        changed = False
        for name in order:
            if name not in dynamic and (deps[name] & special or deps[name] & dynamic):
                dynamic.add(name)
                changed = True

    all_text = " ".join(f for f in formulas if isinstance(f, str))
    uses_tenure = "_tenure" in all_text or any(
        "_tenure" in tk["_export_names_referenced"](f) for f in formulas if isinstance(f, str)
    )
    uses_cycle = any("_cycle" in tk["_export_names_referenced"](f) for f in formulas if isinstance(f, str))
    any_tenure_state = any(s["tenure_states"] for s in strategies)

    cycles_per_year = float(document.get("cycles_per_year", 1.0)) or 1.0
    cycles = int(document.get("cycles", 10))
    analysis = document.get("analysis") or {}
    caveats: list[str] = []
    if len(trackers) != len(all_trackers):
        caveats.append("Tracker rows with an empty name (unfinished rows in the editor) were skipped.")
    for s in strategies:
        for st in s["states"]:
            if st.get("extra_outcomes"):
                caveats.append(f"State '{st['id']}' ({s['name']}) has extra Outcomes-table rows beyond cost/effect; not exported.")
        for group in s["grouped"].values():
            for t in group:
                if t.get("extra_increment"):
                    caveats.append(
                        f"Transition {t.get('from_state')} -> {t.get('to_state')} ({s['name']}) has extra-outcome "
                        f"increments; not exported."
                    )
    return {
        "name": str(document.get("name") or "Microsimulation model"),
        "strategies": strategies,
        "parameters": parameters,
        "variables_raw": variables_raw,
        "var_order": order,
        "dynamic_vars": dynamic,
        "distributions": distributions,
        "tables": tables,
        "trackers": trackers,
        "tracker_names": tracker_names,
        "reset_on_entry": reset_timing == "on_entry",
        "cycles": cycles,
        "half_cycle_correction": bool(document.get("half_cycle_correction", True)),
        "n_patients": max(1, int(document.get("n_patients", 1000))),
        "seed": int(document.get("random_seed", 1)),
        "discount_cost": tk["_per_cycle_rate"](
            float(document.get("discount_cost_value", 0.0)), document.get("discount_cost_unit", "per cycle"), cycles_per_year),
        "discount_effect": tk["_per_cycle_rate"](
            float(document.get("discount_effect_value", 0.0)), document.get("discount_effect_unit", "per cycle"), cycles_per_year),
        "effect_sign": -1 if str(analysis.get("effect_objective", "maximize")).lower().startswith("min") else 1,
        # How the per-context cache is keyed: tenure only matters through
        # `tenure == 1` unless a formula reads _tenure.
        "tenure_key": "full" if uses_tenure else ("cap2" if any_tenure_state else "none"),
        "cycle_key": uses_cycle,
        "caveats": caveats,
    }


# ---------------------------------------------------------------------------
# Language-neutral code pieces (formulas already transpiled to `dialect`)
# ---------------------------------------------------------------------------


def _generic(ir: dict[str, Any], dialect: str, tk: dict[str, Any]) -> dict[str, Any]:
    transpile = tk["_export_transpile"]
    num = tk["_num"]
    parameters, variables_raw = ir["parameters"], ir["variables_raw"]
    namer = tk["_ExportNamer"](dialect)
    namer.used.update(_RESERVED_NAMES)
    for k in range(1, len(ir["strategies"]) + 1):
        namer.used.update({f"rows_{k}", f"cost_{k}", f"effect_{k}", f"reset_{k}", f"init_{k}", f"STRATEGY_{k}"})
    for name in parameters:
        namer.add(name)
    for name in ir["var_order"]:
        namer.add(name)
    for name in ir["tracker_names"]:
        namer.add(name)

    helpers: set[str] = set()
    frozen_notes: list[str] = []
    tracker_names = ir["tracker_names"]
    dynamic_vars = ir["dynamic_vars"]
    dynamic_names = set(tracker_names) | dynamic_vars | {"_cycle", "_tenure"}
    # Stata/Mata: trackers then dynamic variables live in one rowvector `v`.
    vpos = {name: n + 1 for n, name in enumerate(tracker_names + [x for x in ir["var_order"] if x in dynamic_vars])}

    def vref(name: str) -> str:
        if dialect == "stata":
            return f"v[{vpos[name]}]"
        return tk["_export_v_access"](dialect, namer.get(name))

    def name_v(name: str) -> str:
        if name in tracker_names or name in dynamic_vars:
            return vref(name)
        return namer.get(name)

    def name_static(name: str) -> str:
        if name in dynamic_names and name not in ("_cycle", "_tenure"):
            raise ValueError(f"'{name}' is a Tracker or a cycle/tenure-dependent Variable and can't be used here.")
        if name == "_cycle":
            return "0"
        if name == "_tenure":
            return "1"
        return namer.get(name)

    def freeze_cb(snippet: str) -> float:
        if tk["_export_names_referenced"](snippet) & dynamic_names:
            raise ValueError(
                f"'{snippet}' (a table lookup / distribution / aggregate call) depends on _cycle, _tenure or a "
                f"Tracker, so it can't be baked in as a constant; rewrite it with plain arithmetic."
            )
        value = tk["_export_freeze_resolver"](parameters, variables_raw, ir["distributions"], ir["tables"], 0, 1)(snippet)
        frozen_notes.append(f"{snippet!r} -> {value:.6g} (baked in; not representable as {dialect} source)")
        return value

    def tp(expr: Any, name_fn) -> str:
        if expr is None or (isinstance(expr, str) and not expr.strip()):
            return "0.0"
        code, _ = transpile(expr, dialect, name_fn, freeze_cb, helpers)
        return code

    # ---- parameters / variables / trackers -------------------------------
    param_rows = [(namer.get(n), n, float(p.base), p.source) for n, p in parameters.items()]
    static_rows, dynamic_rows = [], []
    for name in ir["var_order"]:
        raw = variables_raw[name]
        if name in dynamic_vars:
            dynamic_rows.append({"ident": namer.get(name), "name": name, "raw": raw, "code": tp(raw, name_v)})
        else:
            static_rows.append({"ident": namer.get(name), "name": name, "raw": raw, "code": tp(raw, name_static)})
    trackers = []
    for t, name in zip(ir["trackers"], tracker_names):
        update = t.get("update_formula")
        trackers.append({
            "ident": namer.get(name), "name": name,
            "init": tp(t.get("initial_value", 0.0), name_static),
            "update": tp(update, name_v) if update not in (None, "") else vref(name),
            "reset_rules": {str(sid): rule for sid, rule in (t.get("reset_rules") or {}).items()},
        })

    # ---- strategies ------------------------------------------------------
    def group_codes(entries: list[tuple[Any, Any, Any, Any]], name_fn) -> list[tuple[Any, str, str, str]]:
        """[(key, prob, cost_inc, effect_inc)] -> transpiled; the '#'/'C'
        complement becomes 1 - (sum of the others)."""
        codes = []
        known = []
        complement_at = None
        for pos, (key, prob, dc, de) in enumerate(entries):
            if tk["_is_complement_value"](prob):
                complement_at = pos
                codes.append([key, None, tp(dc, name_fn) if dc else "0.0", tp(de, name_fn) if de else "0.0"])
                continue
            code = tp(prob, name_fn)
            known.append(code)
            codes.append([key, code, tp(dc, name_fn) if dc else "0.0", tp(de, name_fn) if de else "0.0"])
        if complement_at is not None:
            codes[complement_at][1] = "(1.0 - (" + " + ".join(known) + "))" if known else "1.0"
        return [tuple(c) for c in codes]

    strategies = []
    for sir in ir["strategies"]:
        ids, index = sir["ids"], sir["index"]
        nodes_rows: dict[int, list[tuple[int, str, str, str]]] = {}
        for i, (st, sid_) in enumerate(zip(sir["states"], ids)):
            group = sir["grouped"][sid_]
            if group:
                entries = [(index[str(t["to_state"])], t.get("probability", 0.0), t.get("cost_increment"), t.get("effect_increment")) for t in group]
                nodes_rows[i] = group_codes(entries, name_v)
            elif sir["real"][i]:
                nodes_rows[i] = [(i, "1.0", "0.0", "0.0")]  # absorbing: implicit self-loop
        cost_code: dict[int, Any] = {}
        effect_code: dict[int, str] = {}
        for i, st in enumerate(sir["states"]):
            if not sir["real"][i]:
                continue
            effect_code[i] = tp(st.get("effect", 0.0), name_v)
            if st.get("tenure_applied"):
                recurring = st.get("recurring_cost")
                cost_code[i] = (
                    tp(st.get("initial_cost", 0.0), name_v),
                    tp(recurring if recurring is not None else st.get("cost", 0.0), name_v),
                    tp(st.get("terminal_cost", 0.0), name_v),
                )
            else:
                cost_code[i] = tp(st.get("cost", 0.0), name_v)
        init_entries = [
            (i, st.get("initial_prob", 0.0), None, None)
            for i, st in enumerate(sir["states"]) if sir["real"][i]
        ]
        init_codes = group_codes(init_entries, name_static)
        resets = []  # (tracker index, state index, code)
        for ti, t in enumerate(trackers):
            for sid_, rule in t["reset_rules"].items():
                if sid_ in index and sir["real"][index[sid_]]:
                    resets.append((ti, index[sid_], tp(rule, name_v)))
        strategies.append({
            "ir": sir, "rows": nodes_rows, "cost": cost_code, "effect": effect_code,
            "init": [(c[0], c[1]) for c in init_codes], "resets": resets,
        })
    return {
        "namer": namer, "helpers": helpers, "frozen_notes": frozen_notes,
        "param_rows": param_rows, "static_rows": static_rows, "dynamic_rows": dynamic_rows,
        "trackers": trackers, "strategies": strategies, "vref": vref, "num": num,
    }


def _header_lines(ir: dict[str, Any], g: dict[str, Any], run_hint: str) -> list[str]:
    names = ", ".join(s["name"] for s in ir["strategies"])
    lines = [
        f"Microsimulation model exported from DECIDE: {ir['name']}",
        f"Strategies ({len(ir['strategies'])}): {names}",
        "",
        f"Simulates {ir['n_patients']} patients one by one over {ir['cycles']} cycles for every strategy and prints",
        "the mean discounted cost and effect (with Monte Carlo standard error) and the ICER of each",
        "strategy versus the first one. A point-in-time snapshot of the model, not a live link: parameters",
        "are base-case values (no PSA). Edit freely.",
        run_hint,
        "",
        "Random numbers: patient i uses seed (SEED + i) in every strategy (common random numbers). The",
        "streams differ from DECIDE's, so results agree with the DECIDE app up to Monte Carlo error only.",
        "Run time grows with patients x strategies x cycles: try a small number of patients first.",
        "",
        "Reproduced: states, branch/terminal nodes, probabilities (incl. the '#' complement), one-time",
        "cost/effect increments on transitions, tenure-based cost (initial/recurring/terminal), Trackers",
        "(update formulas + reset rules, on entry or next cycle), Variables, per-cycle discounting,",
        "half-cycle correction, annual->per-cycle rates.",
        "NOT exported: PSM nodes (the export refuses), Distribution rows, random-draw functions, PSA /",
        "sensitivity analysis, outcomes beyond cost/effect, patient traces and plots.",
    ]
    if ir["caveats"]:
        lines.append("")
        lines.append("NOTE -- parts of the DECIDE model that this export leaves out:")
        lines += [f"  - {c}" for c in ir["caveats"]]
    if g["frozen_notes"]:
        lines.append("")
        lines.append("NOTE -- expressions using a construct this exporter can't translate (a table lookup, a")
        lines.append("distribution or an aggregate function) were evaluated once at export time and baked in")
        lines.append("as plain numbers:")
        lines += [f"  - {n}" for n in g["frozen_notes"]]
    return lines


def _comment_block(lines: list[str], prefix: str) -> list[str]:
    return [(f"{prefix} {ln}" if ln else prefix) for ln in lines]


def _py_str(s: Any) -> str:
    return repr(str(s))


def _js_str(s: Any) -> str:
    return '"' + str(s).replace("\\", "\\\\").replace('"', '\\"') + '"'


def _stata_str(s: Any) -> str:
    return '"' + str(s).replace('"', "'") + '"'


def _r_str(s: Any) -> str:
    return '"' + str(s).replace("\\", "\\\\").replace('"', '\\"') + '"'


def _fill(template: str, **kw: str) -> str:
    out = template
    for key, value in kw.items():
        out = out.replace(f"@@{key}@@", value)
    return out


def _tenure_key_expr(mode: str, dialect: str) -> str:
    if mode == "full":
        return "tenure"
    if mode == "cap2":
        return {"python": "min(tenure, 2)", "javascript": "Math.min(tenure, 2)", "r": "min(tenure, 2)", "stata": "min((tenure, 2))"}[dialect]
    return "0"


# ---------------------------------------------------------------------------
# Python
# ---------------------------------------------------------------------------

_PY_ENGINE = '''
# ---------------------------------------------------------------------------
# Simulation engine (generic: nothing below depends on the model)
# ---------------------------------------------------------------------------
def prepare_row(raw):
    """[(target, probability, cost_inc, effect_inc), ...] -> sampling table.
    Negative probabilities are clipped to 0 and the rest normalised, as DECIDE does."""
    probs = [max(float(r[1]), 0.0) for r in raw]
    total = sum(probs)
    probs = [p / total for p in probs] if total > 0 else [1.0 / len(raw)] * len(raw)
    cdf, running = [], 0.0
    for p in probs:
        running += p
        cdf.append(running)
    return ([r[0] for r in raw], cdf, [r[2] for r in raw], [r[3] for r in raw])


def draw(cdf, u):
    """Index of the first cumulative probability above u."""
    for i, c in enumerate(cdf):
        if u < c:
            return i
    return len(cdf) - 1


def leave_probability(S, state, raw_rows):
    """Probability of ending the cycle in a different state than `state`
    (through any chain of branch nodes) -- used for tenure-based cost."""
    leave = 0.0
    stack = [(state, 1.0)]
    while stack:
        node, prob = stack.pop()
        for target, p, _dc, _de in raw_rows[node]:
            q = prob * p
            if q == 0.0:
                continue
            if S["real"][target]:
                landed = target
            elif target in raw_rows:
                stack.append((target, q))
                continue
            else:
                landed = S["alias"][target]
            if landed != state:
                leave += q
    return leave


def evaluate_context(S, state, cycle, tenure, trk):
    """Everything one patient needs for one cycle, given (state, cycle, tenure, trackers)."""
    v = model_variables(cycle, tenure, trk)
    raw_rows = {node: S["rows"](node, cycle, tenure, v) for node in [state] + S["branch_nodes"]}
    exit_p = leave_probability(S, state, raw_rows) if state in S["tenure_states"] else 0.0
    cost = S["cost"](state, cycle, tenure, v, exit_p)
    effect = S["effect"](state, cycle, tenure, v)
    rows = {node: prepare_row(raw) for node, raw in raw_rows.items()}
    return (cost, effect, rows, tracker_update(cycle, tenure, v))


def simulate_patient(S, seed):
    """One patient's discounted total cost and effect (half-cycle correction included)."""
    rng = random.Random(seed)
    targets, cdf, _dc, _de = S["init_row"]
    state = targets[draw(cdf, rng.random())]
    tenure = 1
    trk = list(TRACKER_INITIAL)
    total_cost = total_effect = first_cost = first_effect = 0.0
    for cycle in range(CYCLES):
        key = (state, @@TENURE_KEY@@, @@CYCLE_KEY@@, tuple(trk))
        ctx = S["cache"].get(key)
        if ctx is None:
            if len(S["cache"]) > 200000:
                S["cache"].clear()
            ctx = S["cache"][key] = evaluate_context(S, state, cycle, tenure, trk)
        cost, effect, rows, updated = ctx
        node = state
        while True:  # follow branch nodes down to the real state this cycle ends in
            targets, cdf, d_cost, d_effect = rows[node]
            j = draw(cdf, rng.random())
            cost += d_cost[j]
            effect += d_effect[j]
            target = targets[j]
            if S["real"][target]:
                new_state = target
                break
            if target in rows:
                node = target
            else:
                new_state = S["alias"][target]
                break
        disc_cost = cost / (1.0 + DISCOUNT_COST) ** cycle
        disc_effect = effect / (1.0 + DISCOUNT_EFFECT) ** cycle
        total_cost += disc_cost
        total_effect += disc_effect
        if cycle == 0:
            first_cost, first_effect = disc_cost, disc_effect
        entering = new_state != state
        new_trk = list(updated)
        for k in range(N_TRACKERS):
            if RESET_ON_ENTRY:
                if entering and (k, new_state) in S["resets"]:
                    new_trk[k] = S["reset"](k, new_state, cycle, tenure, new_trk)
            elif (not entering) and tenure == 1 and (k, state) in S["resets"]:
                new_trk[k] = S["reset"](k, state, cycle, tenure, new_trk)
        trk = new_trk
        tenure = 1 if entering else tenure + 1
        state = new_state
    if HALF_CYCLE_CORRECTION and CYCLES > 0:
        v = model_variables(CYCLES, tenure, trk)
        total_cost += 0.5 * (S["cost"](state, CYCLES, tenure, v, 0.0) / (1.0 + DISCOUNT_COST) ** CYCLES - first_cost)
        total_effect += 0.5 * (S["effect"](state, CYCLES, tenure, v) / (1.0 + DISCOUNT_EFFECT) ** CYCLES - first_effect)
    return total_cost, total_effect


def run_strategy(S, n_patients, seed):
    S["cache"] = {}
    S["init_row"] = prepare_row([(s, w, 0.0, 0.0) for s, w in zip(S["init_states"], S["init_weights"]())])
    sum_c = sum_e = sq_c = sq_e = 0.0
    for i in range(n_patients):
        c, e = simulate_patient(S, seed + i)
        sum_c += c
        sum_e += e
        sq_c += c * c
        sq_e += e * e
    mean_c, mean_e = sum_c / n_patients, sum_e / n_patients
    if n_patients > 1:
        se_c = math.sqrt(max(sq_c / n_patients - mean_c ** 2, 0.0) * n_patients / (n_patients - 1) / n_patients)
        se_e = math.sqrt(max(sq_e / n_patients - mean_e ** 2, 0.0) * n_patients / (n_patients - 1) / n_patients)
    else:
        se_c = se_e = float("nan")
    return mean_c, mean_e, se_c, se_e


def icer_text(d_cost, d_effect):
    if d_effect == 0:
        return "n/a (no effect difference)"
    better = EFFECT_SIGN * d_effect > 0
    if better and d_cost <= 0:
        return "dominant (cheaper and better)"
    if (not better) and d_cost >= 0:
        return "dominated (costlier and not better)"
    return f"{d_cost / d_effect:,.2f}"


def main():
    n_patients = int(sys.argv[1]) if len(sys.argv) > 1 else N_PATIENTS
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else SEED
    print(f"Microsimulation: {n_patients} patients, {CYCLES} cycles, seed {seed}")
    results = [(S["name"],) + run_strategy(S, n_patients, seed) for S in STRATEGIES]
    print()
    print(f"{'Strategy':<28}{'Mean cost':>16}{'(SE)':>12}{'Mean effect':>14}{'(SE)':>10}")
    for name, mc, me, sc, se in results:
        print(f"{name:<28}{mc:>16,.2f}{sc:>12,.2f}{me:>14.4f}{se:>10.4f}")
    if len(results) > 1:
        print()
        print(f"Versus {results[0][0]}:")
        print(f"{'Strategy':<28}{'Delta cost':>16}{'Delta effect':>14}   ICER")
        for name, mc, me, _sc, _se in results[1:]:
            d_c, d_e = mc - results[0][1], me - results[0][2]
            print(f"{name:<28}{d_c:>16,.2f}{d_e:>14.4f}   {icer_text(d_c, d_e)}")


if __name__ == "__main__":
    main()
'''


def _render_python(ir: dict[str, Any], tk: dict[str, Any]) -> str:
    g = _generic(ir, "python", tk)
    num = g["num"]
    vref = g["vref"]
    L: list[str] = ['"""' + "\n".join(_header_lines(ir, g, "Run with: python3 <file>.py [n_patients] [seed]   (standard library only)")), '"""', "", "import math", "import random", "import sys", ""]

    if g["helpers"]:
        L.append("# ---- Functions ----")
        for name in ("bound3", "RateToProb", "ProbToProb", "ProbToRate"):
            if name in g["helpers"]:
                L.append(tk["_EXPORT_HELPER_SOURCE"][name]["python"])
        L.append("")
    L += [
        "# ---- Settings ----",
        f"N_PATIENTS = {ir['n_patients']}",
        f"SEED = {ir['seed']}",
        f"CYCLES = {ir['cycles']}",
        f"HALF_CYCLE_CORRECTION = {ir['half_cycle_correction']}",
        f"DISCOUNT_COST = {num(ir['discount_cost'])}   # per cycle",
        f"DISCOUNT_EFFECT = {num(ir['discount_effect'])}   # per cycle",
        f"RESET_ON_ENTRY = {ir['reset_on_entry']}   # Tracker reset timing",
        f"EFFECT_SIGN = {ir['effect_sign']}   # +1: higher effect is better",
        "",
    ]
    if g["param_rows"]:
        L.append("# ---- Parameters ----")
        for ident, name, base, source in g["param_rows"]:
            bits = ([] if ident == name else [name]) + ([source] if source else [])
            L.append(f"{ident} = {num(base)}" + (f"  # {' -- '.join(bits)}" if bits else ""))
        L.append("")
    if g["static_rows"]:
        L.append("# ---- Variables (static: same value every cycle) ----")
        for r in g["static_rows"]:
            if isinstance(r["raw"], str):
                L.append(f"# {r['name']} = {r['raw']}")
            L.append(f"{r['ident']} = {r['code']}")
        L.append("")
    L.append("# ---- Trackers (per-patient values carried across cycles) ----")
    L.append("TRACKERS = " + repr([t["name"] for t in g["trackers"]]))
    L.append("N_TRACKERS = len(TRACKERS)")
    L.append("TRACKER_INITIAL = [" + ", ".join(t["init"] for t in g["trackers"]) + "]")
    L.append("")
    L.append("def model_variables(cycle, tenure, trk):")
    L.append('    """Tracker values and the Variables that depend on cycle / tenure / Trackers."""')
    L.append("    v = {}")
    for k, t in enumerate(g["trackers"]):
        L.append(f"    {vref(t['name'])} = trk[{k}]")
    for r in g["dynamic_rows"]:
        L.append(f"    {vref(r['name'])} = {r['code']}   # {r['name']}")
    L.append("    return v")
    L.append("")
    L.append("def tracker_update(cycle, tenure, v):")
    L.append('    """Each Tracker\'s update formula (evaluated in the start-of-cycle context)."""')
    L.append("    return [" + ", ".join(t["update"] for t in g["trackers"]) + "]")
    L.append("")

    any_resets = False
    for k, s in enumerate(g["strategies"], start=1):
        sir = s["ir"]
        ids = sir["ids"]
        L.append(f"# ===== Strategy {k}: {sir['name']} (id {sir['id']}) =====")
        L.append(f"# node index -> DECIDE id: " + ", ".join(f"{i}={sid_}" + ("" if sir['real'][i] else " (branch)") for i, sid_ in enumerate(ids)))
        L.append(f"def rows_{k}(node, cycle, tenure, v):")
        L.append('    """Outgoing transitions of `node`: [(target, probability, cost increment, effect increment)]."""')
        for i, rows in s["rows"].items():
            L.append(f"    if node == {i}:  # {ids[i]}")
            L.append("        return [")
            for to, p, dc, de in rows:
                L.append(f"            ({to}, {p}, {dc}, {de}),  # -> {ids[to]}")
            L.append("        ]")
        L.append("    raise ValueError(node)")
        L.append("")
        L.append(f"def cost_{k}(state, cycle, tenure, v, exit_p):")
        L.append('    """Cost of one cycle in `state` (exit_p = probability of leaving it, for tenure-based cost)."""')
        for i, code in s["cost"].items():
            L.append(f"    if state == {i}:  # {ids[i]}")
            if isinstance(code, tuple):
                ini, rec, ter = code
                L.append(f"        base = {ini} if tenure == 1 else {rec} * (1.0 - exit_p)")
                L.append(f"        return base + (0.0 if tenure == 1 else {ter} * exit_p)")
            else:
                L.append(f"        return {code}")
        L.append("    raise ValueError(state)")
        L.append("")
        L.append(f"def effect_{k}(state, cycle, tenure, v):")
        for i, code in s["effect"].items():
            L.append(f"    if state == {i}:  # {ids[i]}")
            L.append(f"        return {code}")
        L.append("    raise ValueError(state)")
        L.append("")
        L.append(f"def reset_{k}(k, state, cycle, tenure, trk):")
        L.append('    """Tracker k\'s reset value when the rule for `state` fires."""')
        L.append("    v = model_variables(cycle, tenure, trk)")
        for ti, si, code in s["resets"]:
            any_resets = True
            L.append(f"    if k == {ti} and state == {si}:  # {g['trackers'][ti]['name']} @ {ids[si]}")
            L.append(f"        return {code}")
        L.append("    return trk[k]")
        L.append("")
        L.append(f"def init_{k}():")
        L.append('    """Initial-state weights (normalised when sampling)."""')
        L.append("    return [" + ", ".join(code for _i, code in s["init"]) + "]")
        L.append("")
    L.append("# ---- Strategy table ----")
    L.append("STRATEGIES = [")
    for k, s in enumerate(g["strategies"], start=1):
        sir = s["ir"]
        alias = [a for a in sir["alias"]]
        L.append("    {")
        L.append(f'        "id": {_py_str(sir["id"])}, "name": {_py_str(sir["name"])},')
        L.append(f'        "real": {sir["real"]!r},')
        L.append(f'        "alias": {alias!r},')
        L.append(f'        "branch_nodes": {sir["branch_nodes"]!r},')
        L.append(f'        "tenure_states": {sir["tenure_states"]!r},')
        L.append(f'        "init_states": {[i_ for i_, _c in s["init"]]!r}, "init_weights": init_{k},')
        L.append(f'        "rows": rows_{k}, "cost": cost_{k}, "effect": effect_{k}, "reset": reset_{k},')
        L.append(f'        "resets": {{{", ".join(f"({ti}, {si})" for ti, si, _c in s["resets"])}}},')
        L.append("    },")
    L.append("]")
    engine = _fill(
        _PY_ENGINE,
        TENURE_KEY=_tenure_key_expr(ir["tenure_key"], "python"),
        CYCLE_KEY="cycle" if ir["cycle_key"] else "0",
    )
    return "\n".join(L) + "\n" + engine


# ---------------------------------------------------------------------------
# JavaScript (Node)
# ---------------------------------------------------------------------------

_JS_ENGINE = '''
// ---------------------------------------------------------------------------
// Simulation engine (generic: nothing below depends on the model)
// ---------------------------------------------------------------------------
function mulberry32(seed) {
  // Small seedable generator (one per patient).
  let a = seed >>> 0;
  return function () {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function prepareRow(raw) {
  // [[target, probability, costInc, effectInc], ...] -> sampling table.
  // Negative probabilities are clipped to 0 and the rest normalised, as DECIDE does.
  let probs = raw.map((r) => Math.max(Number(r[1]), 0));
  const total = probs.reduce((a, b) => a + b, 0);
  probs = total > 0 ? probs.map((p) => p / total) : probs.map(() => 1 / probs.length);
  const cdf = [];
  let running = 0;
  for (const p of probs) {
    running += p;
    cdf.push(running);
  }
  return { targets: raw.map((r) => r[0]), cdf: cdf, dCost: raw.map((r) => r[2]), dEffect: raw.map((r) => r[3]) };
}

function draw(cdf, u) {
  for (let i = 0; i < cdf.length; i++) {
    if (u < cdf[i]) return i;
  }
  return cdf.length - 1;
}

function leaveProbability(S, state, rawRows) {
  // Probability of ending the cycle in a different state than `state`
  // (through any chain of branch nodes) -- used for tenure-based cost.
  let leave = 0;
  const stack = [[state, 1.0]];
  while (stack.length) {
    const [node, prob] = stack.pop();
    for (const [target, p] of rawRows[node]) {
      const q = prob * p;
      if (q === 0) continue;
      let landed;
      if (S.real[target]) landed = target;
      else if (rawRows[target] !== undefined) {
        stack.push([target, q]);
        continue;
      } else landed = S.alias[target];
      if (landed !== state) leave += q;
    }
  }
  return leave;
}

function evaluateContext(S, state, cycle, tenure, trk) {
  // Everything one patient needs for one cycle, given (state, cycle, tenure, trackers).
  const v = modelVariables(cycle, tenure, trk);
  const rawRows = {};
  for (const node of [state].concat(S.branchNodes)) rawRows[node] = S.rows(node, cycle, tenure, v);
  const exitP = S.tenureStates.includes(state) ? leaveProbability(S, state, rawRows) : 0.0;
  const rows = {};
  for (const node of Object.keys(rawRows)) rows[node] = prepareRow(rawRows[node]);
  return {
    cost: S.cost(state, cycle, tenure, v, exitP),
    effect: S.effect(state, cycle, tenure, v),
    rows: rows,
    updated: trackerUpdate(cycle, tenure, v),
  };
}

function simulatePatient(S, seed) {
  // One patient's discounted total cost and effect (half-cycle correction included).
  const rng = mulberry32(seed);
  let state = S.initRow.targets[draw(S.initRow.cdf, rng())];
  let tenure = 1;
  let trk = TRACKER_INITIAL.slice();
  let totalCost = 0, totalEffect = 0, firstCost = 0, firstEffect = 0;
  for (let cycle = 0; cycle < CYCLES; cycle++) {
    const key = [state, @@TENURE_KEY@@, @@CYCLE_KEY@@, trk.join(",")].join("|");
    let ctx = S.cache.get(key);
    if (ctx === undefined) {
      if (S.cache.size > 200000) S.cache.clear();
      ctx = evaluateContext(S, state, cycle, tenure, trk);
      S.cache.set(key, ctx);
    }
    let cost = ctx.cost, effect = ctx.effect;
    let node = state, newState;
    for (;;) { // follow branch nodes down to the real state this cycle ends in
      const row = ctx.rows[node];
      const j = draw(row.cdf, rng());
      cost += row.dCost[j];
      effect += row.dEffect[j];
      const target = row.targets[j];
      if (S.real[target]) { newState = target; break; }
      if (ctx.rows[target] !== undefined) node = target;
      else { newState = S.alias[target]; break; }
    }
    const discCost = cost / Math.pow(1 + DISCOUNT_COST, cycle);
    const discEffect = effect / Math.pow(1 + DISCOUNT_EFFECT, cycle);
    totalCost += discCost;
    totalEffect += discEffect;
    if (cycle === 0) { firstCost = discCost; firstEffect = discEffect; }
    const entering = newState !== state;
    const newTrk = ctx.updated.slice();
    for (let k = 0; k < N_TRACKERS; k++) {
      if (RESET_ON_ENTRY) {
        if (entering && S.resets.has(k + "," + newState)) newTrk[k] = S.reset(k, newState, cycle, tenure, newTrk);
      } else if (!entering && tenure === 1 && S.resets.has(k + "," + state)) {
        newTrk[k] = S.reset(k, state, cycle, tenure, newTrk);
      }
    }
    trk = newTrk;
    tenure = entering ? 1 : tenure + 1;
    state = newState;
  }
  if (HALF_CYCLE_CORRECTION && CYCLES > 0) {
    const v = modelVariables(CYCLES, tenure, trk);
    totalCost += 0.5 * (S.cost(state, CYCLES, tenure, v, 0.0) / Math.pow(1 + DISCOUNT_COST, CYCLES) - firstCost);
    totalEffect += 0.5 * (S.effect(state, CYCLES, tenure, v) / Math.pow(1 + DISCOUNT_EFFECT, CYCLES) - firstEffect);
  }
  return [totalCost, totalEffect];
}

function runStrategy(S, nPatients, seed) {
  S.cache = new Map();
  const weights = S.initWeights();
  S.initRow = prepareRow(S.initStates.map((s, i) => [s, weights[i], 0, 0]));
  let sumC = 0, sumE = 0, sqC = 0, sqE = 0;
  for (let i = 0; i < nPatients; i++) {
    const [c, e] = simulatePatient(S, seed + i);
    sumC += c; sumE += e; sqC += c * c; sqE += e * e;
  }
  const meanC = sumC / nPatients, meanE = sumE / nPatients;
  const se = (sq, mean) => (nPatients > 1 ? Math.sqrt((Math.max(sq / nPatients - mean * mean, 0) * nPatients) / (nPatients - 1) / nPatients) : NaN);
  return { name: S.name, meanCost: meanC, meanEffect: meanE, seCost: se(sqC, meanC), seEffect: se(sqE, meanE) };
}

function fmt(x, digits) {
  return x.toLocaleString("en-US", { minimumFractionDigits: digits, maximumFractionDigits: digits });
}

function icerText(dCost, dEffect) {
  if (dEffect === 0) return "n/a (no effect difference)";
  const better = EFFECT_SIGN * dEffect > 0;
  if (better && dCost <= 0) return "dominant (cheaper and better)";
  if (!better && dCost >= 0) return "dominated (costlier and not better)";
  return fmt(dCost / dEffect, 2);
}

function main() {
  const args = typeof process !== "undefined" ? process.argv.slice(2) : [];
  const nPatients = args.length > 0 ? parseInt(args[0], 10) : N_PATIENTS;
  const seed = args.length > 1 ? parseInt(args[1], 10) : SEED;
  console.log(`Microsimulation: ${nPatients} patients, ${CYCLES} cycles, seed ${seed}`);
  const results = STRATEGIES.map((S) => runStrategy(S, nPatients, seed));
  console.log("");
  console.log("Strategy".padEnd(28) + "Mean cost".padStart(16) + "(SE)".padStart(12) + "Mean effect".padStart(14) + "(SE)".padStart(10));
  for (const r of results) {
    console.log(r.name.padEnd(28) + fmt(r.meanCost, 2).padStart(16) + fmt(r.seCost, 2).padStart(12) + r.meanEffect.toFixed(4).padStart(14) + r.seEffect.toFixed(4).padStart(10));
  }
  if (results.length > 1) {
    console.log("");
    console.log(`Versus ${results[0].name}:`);
    console.log("Strategy".padEnd(28) + "Delta cost".padStart(16) + "Delta effect".padStart(14) + "   ICER");
    for (const r of results.slice(1)) {
      const dC = r.meanCost - results[0].meanCost, dE = r.meanEffect - results[0].meanEffect;
      console.log(r.name.padEnd(28) + fmt(dC, 2).padStart(16) + dE.toFixed(4).padStart(14) + "   " + icerText(dC, dE));
    }
  }
}

main();
'''


def _render_javascript(ir: dict[str, Any], tk: dict[str, Any]) -> str:
    g = _generic(ir, "javascript", tk)
    num = g["num"]
    vref = g["vref"]
    L: list[str] = ["/*"] + _header_lines(ir, g, "Run with: node <file>.js [n_patients] [seed]   (no dependencies)") + ["*/", ""]
    # '*/' inside the header would end the comment; formulas never contain it.
    if g["helpers"]:
        L.append("// ---- Functions ----")
        for name in ("bound3", "RateToProb", "ProbToProb", "ProbToRate"):
            if name in g["helpers"]:
                L.append(tk["_EXPORT_HELPER_SOURCE"][name]["javascript"])
        L.append("")
    L += [
        "// ---- Settings ----",
        f"const N_PATIENTS = {ir['n_patients']};",
        f"const SEED = {ir['seed']};",
        f"const CYCLES = {ir['cycles']};",
        f"const HALF_CYCLE_CORRECTION = {'true' if ir['half_cycle_correction'] else 'false'};",
        f"const DISCOUNT_COST = {num(ir['discount_cost'])}; // per cycle",
        f"const DISCOUNT_EFFECT = {num(ir['discount_effect'])}; // per cycle",
        f"const RESET_ON_ENTRY = {'true' if ir['reset_on_entry'] else 'false'}; // Tracker reset timing",
        f"const EFFECT_SIGN = {ir['effect_sign']}; // +1: higher effect is better",
        "",
    ]
    if g["param_rows"]:
        L.append("// ---- Parameters ----")
        for ident, name, base, source in g["param_rows"]:
            bits = ([] if ident == name else [name]) + ([source] if source else [])
            L.append(f"const {ident} = {num(base)};" + (f" // {' -- '.join(bits)}" if bits else ""))
        L.append("")
    if g["static_rows"]:
        L.append("// ---- Variables (static: same value every cycle) ----")
        for r in g["static_rows"]:
            if isinstance(r["raw"], str):
                L.append(f"// {r['name']} = {r['raw']}")
            L.append(f"const {r['ident']} = {r['code']};")
        L.append("")
    L.append("// ---- Trackers (per-patient values carried across cycles) ----")
    L.append("const TRACKERS = [" + ", ".join(_js_str(t["name"]) for t in g["trackers"]) + "];")
    L.append("const N_TRACKERS = TRACKERS.length;")
    L.append("const TRACKER_INITIAL = [" + ", ".join(t["init"] for t in g["trackers"]) + "];")
    L.append("")
    L.append("function modelVariables(cycle, tenure, trk) {")
    L.append("  // Tracker values and the Variables that depend on cycle / tenure / Trackers.")
    L.append("  const v = {};")
    for k, t in enumerate(g["trackers"]):
        L.append(f"  {vref(t['name'])} = trk[{k}];")
    for r in g["dynamic_rows"]:
        L.append(f"  {vref(r['name'])} = {r['code']}; // {r['name']}")
    L.append("  return v;")
    L.append("}")
    L.append("")
    L.append("function trackerUpdate(cycle, tenure, v) {")
    L.append("  // Each Tracker's update formula (evaluated in the start-of-cycle context).")
    L.append("  return [" + ", ".join(t["update"] for t in g["trackers"]) + "];")
    L.append("}")
    L.append("")
    for k, s in enumerate(g["strategies"], start=1):
        sir = s["ir"]
        ids = sir["ids"]
        L.append(f"// ===== Strategy {k}: {sir['name']} (id {sir['id']}) =====")
        L.append("// node index -> DECIDE id: " + ", ".join(f"{i}={sid_}" + ("" if sir["real"][i] else " (branch)") for i, sid_ in enumerate(ids)))
        L.append(f"function rows_{k}(node, cycle, tenure, v) {{")
        L.append("  // Outgoing transitions of `node`: [target, probability, cost increment, effect increment].")
        for i, rows in s["rows"].items():
            L.append(f"  if (node === {i}) {{ // {ids[i]}")
            L.append("    return [")
            for to, p, dc, de in rows:
                L.append(f"      [{to}, {p}, {dc}, {de}], // -> {ids[to]}")
            L.append("    ];")
            L.append("  }")
        L.append("  throw new Error('unknown node ' + node);")
        L.append("}")
        L.append("")
        L.append(f"function cost_{k}(state, cycle, tenure, v, exitP) {{")
        L.append("  // Cost of one cycle in `state` (exitP = probability of leaving it, for tenure-based cost).")
        for i, code in s["cost"].items():
            L.append(f"  if (state === {i}) {{ // {ids[i]}")
            if isinstance(code, tuple):
                ini, rec, ter = code
                L.append(f"    const base = tenure === 1 ? {ini} : {rec} * (1.0 - exitP);")
                L.append(f"    return tenure === 1 ? base : base + {ter} * exitP;")
            else:
                L.append(f"    return {code};")
            L.append("  }")
        L.append("  throw new Error('unknown state ' + state);")
        L.append("}")
        L.append("")
        L.append(f"function effect_{k}(state, cycle, tenure, v) {{")
        for i, code in s["effect"].items():
            L.append(f"  if (state === {i}) return {code}; // {ids[i]}")
        L.append("  throw new Error('unknown state ' + state);")
        L.append("}")
        L.append("")
        L.append(f"function reset_{k}(k, state, cycle, tenure, trk) {{")
        L.append("  // Tracker k's reset value when the rule for `state` fires.")
        L.append("  const v = modelVariables(cycle, tenure, trk);")
        for ti, si, code in s["resets"]:
            L.append(f"  if (k === {ti} && state === {si}) return {code}; // {g['trackers'][ti]['name']} @ {ids[si]}")
        L.append("  return trk[k];")
        L.append("}")
        L.append("")
        L.append(f"function init_{k}() {{")
        L.append("  // Initial-state weights (normalised when sampling).")
        L.append("  return [" + ", ".join(code for _i, code in s["init"]) + "];")
        L.append("}")
        L.append("")
    L.append("// ---- Strategy table ----")
    L.append("const STRATEGIES = [")
    for k, s in enumerate(g["strategies"], start=1):
        sir = s["ir"]
        js_bool = lambda xs: "[" + ", ".join("true" if x else "false" for x in xs) + "]"  # noqa: E731
        js_alias = "[" + ", ".join("null" if a is None else str(a) for a in sir["alias"]) + "]"
        L.append("  {")
        L.append(f"    id: {_js_str(sir['id'])}, name: {_js_str(sir['name'])},")
        L.append(f"    real: {js_bool(sir['real'])},")
        L.append(f"    alias: {js_alias},")
        L.append(f"    branchNodes: {sir['branch_nodes']!r},")
        L.append(f"    tenureStates: {sir['tenure_states']!r},")
        L.append(f"    initStates: {[i for i, _c in s['init']]!r}, initWeights: init_{k},")
        L.append(f"    rows: rows_{k}, cost: cost_{k}, effect: effect_{k}, reset: reset_{k},")
        L.append("    resets: new Set([" + ", ".join(f'"{ti},{si}"' for ti, si, _c in s["resets"]) + "]),")
        L.append("  },")
    L.append("];")
    engine = _fill(
        _JS_ENGINE,
        TENURE_KEY=_tenure_key_expr(ir["tenure_key"], "javascript"),
        CYCLE_KEY="cycle" if ir["cycle_key"] else "0",
    )
    return "\n".join(L) + "\n" + engine


# ---------------------------------------------------------------------------
# R (base R)
# ---------------------------------------------------------------------------

_R_ENGINE = '''
# ---------------------------------------------------------------------------
# Simulation engine (generic: nothing below depends on the model)
# ---------------------------------------------------------------------------
prepare_row <- function(raw) {
  # list(targets, p, dc, de) -> sampling table. Negative probabilities are
  # clipped to 0 and the rest normalised, as DECIDE does.
  probs <- pmax(as.numeric(raw$p), 0)
  total <- sum(probs)
  probs <- if (total > 0) probs / total else rep(1 / length(probs), length(probs))
  list(targets = raw$targets, cdf = cumsum(probs), dc = raw$dc, de = raw$de)
}

draw <- function(cdf, u) {
  # Index of the first cumulative probability above u.
  idx <- which(u < cdf)[1]
  if (is.na(idx)) length(cdf) else idx
}

leave_probability <- function(S, state, raw_rows) {
  # Probability of ending the cycle in a different state than `state`
  # (through any chain of branch nodes) -- used for tenure-based cost.
  leave <- 0
  stack <- list(list(node = state, prob = 1))
  while (length(stack) > 0) {
    item <- stack[[length(stack)]]
    stack[[length(stack)]] <- NULL
    row <- raw_rows[[as.character(item$node)]]
    for (m in seq_along(row$targets)) {
      target <- row$targets[m]
      q <- item$prob * row$p[m]
      if (q == 0) next
      if (S$real[target]) {
        landed <- target
      } else if (!is.null(raw_rows[[as.character(target)]])) {
        stack[[length(stack) + 1]] <- list(node = target, prob = q)
        next
      } else {
        landed <- S$alias[target]
      }
      if (landed != state) leave <- leave + q
    }
  }
  leave
}

evaluate_context <- function(S, state, cycle, tenure, trk) {
  # Everything one patient needs for one cycle, given (state, cycle, tenure, trackers).
  v <- model_variables(cycle, tenure, trk)
  raw_rows <- list()
  for (node in c(state, S$branch_nodes)) raw_rows[[as.character(node)]] <- S$rows(node, cycle, tenure, v)
  exit_p <- if (state %in% S$tenure_states) leave_probability(S, state, raw_rows) else 0
  list(
    cost = S$cost(state, cycle, tenure, v, exit_p),
    effect = S$effect(state, cycle, tenure, v),
    rows = lapply(raw_rows, prepare_row),
    updated = tracker_update(cycle, tenure, v)
  )
}

simulate_patient <- function(S, seed) {
  # One patient's discounted total cost and effect (half-cycle correction included).
  set.seed(seed)
  state <- S$init_row$targets[draw(S$init_row$cdf, runif(1))]
  tenure <- 1
  trk <- TRACKER_INITIAL
  total_cost <- 0; total_effect <- 0; first_cost <- 0; first_effect <- 0
  for (cycle in seq_len(CYCLES) - 1L) {
    key <- paste(state, @@TENURE_KEY@@, @@CYCLE_KEY@@, paste(sprintf("%.17g", trk), collapse = ","), sep = "|")
    ctx <- S$cache[[key]]
    if (is.null(ctx)) {
      ctx <- evaluate_context(S, state, cycle, tenure, trk)
      assign(key, ctx, envir = S$cache)
    }
    cost <- ctx$cost; effect <- ctx$effect
    node <- state
    repeat {  # follow branch nodes down to the real state this cycle ends in
      row <- ctx$rows[[as.character(node)]]
      j <- draw(row$cdf, runif(1))
      cost <- cost + row$dc[j]
      effect <- effect + row$de[j]
      target <- row$targets[j]
      if (S$real[target]) { new_state <- target; break }
      if (!is.null(ctx$rows[[as.character(target)]])) {
        node <- target
      } else {
        new_state <- S$alias[target]; break
      }
    }
    disc_cost <- cost / (1 + DISCOUNT_COST)^cycle
    disc_effect <- effect / (1 + DISCOUNT_EFFECT)^cycle
    total_cost <- total_cost + disc_cost
    total_effect <- total_effect + disc_effect
    if (cycle == 0) { first_cost <- disc_cost; first_effect <- disc_effect }
    entering <- new_state != state
    new_trk <- ctx$updated
    for (k in seq_len(N_TRACKERS)) {
      if (RESET_ON_ENTRY) {
        if (entering && any(S$resets[, 1] == k & S$resets[, 2] == new_state))
          new_trk[k] <- S$reset(k, new_state, cycle, tenure, new_trk)
      } else if (!entering && tenure == 1 && any(S$resets[, 1] == k & S$resets[, 2] == state)) {
        new_trk[k] <- S$reset(k, state, cycle, tenure, new_trk)
      }
    }
    trk <- new_trk
    tenure <- if (entering) 1 else tenure + 1
    state <- new_state
  }
  if (HALF_CYCLE_CORRECTION && CYCLES > 0) {
    v <- model_variables(CYCLES, tenure, trk)
    total_cost <- total_cost + 0.5 * (S$cost(state, CYCLES, tenure, v, 0) / (1 + DISCOUNT_COST)^CYCLES - first_cost)
    total_effect <- total_effect + 0.5 * (S$effect(state, CYCLES, tenure, v) / (1 + DISCOUNT_EFFECT)^CYCLES - first_effect)
  }
  c(total_cost, total_effect)
}

run_strategy <- function(S, n_patients, seed) {
  S$cache <- new.env(hash = TRUE)
  weights <- S$init_weights()
  S$init_row <- prepare_row(list(targets = S$init_states, p = weights, dc = rep(0, length(weights)), de = rep(0, length(weights))))
  costs <- numeric(n_patients)
  effects <- numeric(n_patients)
  for (i in seq_len(n_patients)) {
    res <- simulate_patient(S, seed + i - 1)
    costs[i] <- res[1]
    effects[i] <- res[2]
  }
  list(name = S$name, mean_cost = mean(costs), mean_effect = mean(effects),
       se_cost = if (n_patients > 1) sd(costs) / sqrt(n_patients) else NA,
       se_effect = if (n_patients > 1) sd(effects) / sqrt(n_patients) else NA)
}

icer_text <- function(d_cost, d_effect) {
  if (d_effect == 0) return("n/a (no effect difference)")
  better <- EFFECT_SIGN * d_effect > 0
  if (better && d_cost <= 0) return("dominant (cheaper and better)")
  if (!better && d_cost >= 0) return("dominated (costlier and not better)")
  format(d_cost / d_effect, big.mark = ",", nsmall = 2, digits = 10)
}

main <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  n_patients <- if (length(args) > 0) as.integer(args[1]) else N_PATIENTS
  seed <- if (length(args) > 1) as.integer(args[2]) else SEED
  cat(sprintf("Microsimulation: %d patients, %d cycles, seed %d\\n", n_patients, CYCLES, seed))
  results <- lapply(STRATEGIES, run_strategy, n_patients = n_patients, seed = seed)
  cat("\\n")
  cat(sprintf("%-28s%16s%12s%14s%10s\\n", "Strategy", "Mean cost", "(SE)", "Mean effect", "(SE)"))
  for (r in results) {
    cat(sprintf("%-28s%16s%12s%14.4f%10.4f\\n", r$name, format(round(r$mean_cost, 2), big.mark = ",", nsmall = 2),
                format(round(r$se_cost, 2), big.mark = ",", nsmall = 2), r$mean_effect, r$se_effect))
  }
  if (length(results) > 1) {
    cat(sprintf("\\nVersus %s:\\n", results[[1]]$name))
    cat(sprintf("%-28s%16s%14s   %s\\n", "Strategy", "Delta cost", "Delta effect", "ICER"))
    for (r in results[-1]) {
      d_c <- r$mean_cost - results[[1]]$mean_cost
      d_e <- r$mean_effect - results[[1]]$mean_effect
      cat(sprintf("%-28s%16s%14.4f   %s\\n", r$name, format(round(d_c, 2), big.mark = ",", nsmall = 2), d_e, icer_text(d_c, d_e)))
    }
  }
}

main()
'''


def _render_r(ir: dict[str, Any], tk: dict[str, Any]) -> str:
    g = _generic(ir, "r", tk)
    num = g["num"]
    vref = g["vref"]
    L: list[str] = _comment_block(_header_lines(ir, g, "Run with: Rscript <file>.R [n_patients] [seed]   (base R only)"), "#") + [""]
    if g["helpers"]:
        L.append("# ---- Functions ----")
        for name in ("bound3", "RateToProb", "ProbToProb", "ProbToRate"):
            if name in g["helpers"]:
                L.append(tk["_EXPORT_HELPER_SOURCE"][name]["r"])
        L.append("")
    L += [
        "# ---- Settings ----",
        f"N_PATIENTS <- {ir['n_patients']}L",
        f"SEED <- {ir['seed']}L",
        f"CYCLES <- {ir['cycles']}L",
        f"HALF_CYCLE_CORRECTION <- {'TRUE' if ir['half_cycle_correction'] else 'FALSE'}",
        f"DISCOUNT_COST <- {num(ir['discount_cost'])}   # per cycle",
        f"DISCOUNT_EFFECT <- {num(ir['discount_effect'])}   # per cycle",
        f"RESET_ON_ENTRY <- {'TRUE' if ir['reset_on_entry'] else 'FALSE'}   # Tracker reset timing",
        f"EFFECT_SIGN <- {ir['effect_sign']}   # +1: higher effect is better",
        "",
    ]
    if g["param_rows"]:
        L.append("# ---- Parameters ----")
        for ident, name, base, source in g["param_rows"]:
            bits = ([] if ident == name else [name]) + ([source] if source else [])
            L.append(f"{ident} <- {num(base)}" + (f"  # {' -- '.join(bits)}" if bits else ""))
        L.append("")
    if g["static_rows"]:
        L.append("# ---- Variables (static: same value every cycle) ----")
        for r in g["static_rows"]:
            if isinstance(r["raw"], str):
                L.append(f"# {r['name']} = {r['raw']}")
            L.append(f"{r['ident']} <- {r['code']}")
        L.append("")
    L.append("# ---- Trackers (per-patient values carried across cycles) ----")
    L.append("TRACKERS <- c(" + ", ".join(_r_str(t["name"]) for t in g["trackers"]) + ")")
    L.append("N_TRACKERS <- length(TRACKERS)")
    L.append("TRACKER_INITIAL <- c(" + ", ".join(t["init"] for t in g["trackers"]) + ")")
    L.append("")
    L.append("model_variables <- function(cycle, tenure, trk) {")
    L.append("  # Tracker values and the Variables that depend on cycle / tenure / Trackers.")
    L.append("  v <- list()")
    for k, t in enumerate(g["trackers"], start=1):
        L.append(f"  {vref(t['name'])} <- trk[{k}]")
    for r in g["dynamic_rows"]:
        L.append(f"  {vref(r['name'])} <- {r['code']}   # {r['name']}")
    L.append("  v")
    L.append("}")
    L.append("")
    L.append("tracker_update <- function(cycle, tenure, v) {")
    L.append("  # Each Tracker's update formula (evaluated in the start-of-cycle context).")
    L.append("  c(" + ", ".join(t["update"] for t in g["trackers"]) + ")")
    L.append("}")
    L.append("")

    for k, s in enumerate(g["strategies"], start=1):
        sir = s["ir"]
        ids = sir["ids"]
        L.append(f"# ===== Strategy {k}: {sir['name']} (id {sir['id']}) =====")
        L.append("# node index -> DECIDE id: " + ", ".join(f"{i + 1}={sid_}" + ("" if sir["real"][i] else " (branch)") for i, sid_ in enumerate(ids)))
        L.append(f"rows_{k} <- function(node, cycle, tenure, v) {{")
        L.append("  # Outgoing transitions of `node`: targets, probabilities, cost / effect increments.")
        for i, rows in s["rows"].items():
            L.append(f"  if (node == {i + 1}) {{  # {ids[i]}")
            L.append("    return(list(")
            L.append("      targets = c(" + ", ".join(str(to + 1) for to, _p, _c, _e in rows) + "),  # -> " + ", ".join(ids[to] for to, _p, _c, _e in rows))
            L.append("      p = c(" + ", ".join(p for _t, p, _c, _e in rows) + "),")
            L.append("      dc = c(" + ", ".join(c for _t, _p, c, _e in rows) + "),")
            L.append("      de = c(" + ", ".join(e for _t, _p, _c, e in rows) + ")")
            L.append("    ))")
            L.append("  }")
        L.append('  stop(paste("unknown node", node))')
        L.append("}")
        L.append("")
        L.append(f"cost_{k} <- function(state, cycle, tenure, v, exit_p) {{")
        L.append("  # Cost of one cycle in `state` (exit_p = probability of leaving it, for tenure-based cost).")
        for i, code in s["cost"].items():
            L.append(f"  if (state == {i + 1}) {{  # {ids[i]}")
            if isinstance(code, tuple):
                ini, rec, ter = code
                L.append(f"    base <- if (tenure == 1) {ini} else {rec} * (1 - exit_p)")
                L.append(f"    return(if (tenure == 1) base else base + {ter} * exit_p)")
            else:
                L.append(f"    return({code})")
            L.append("  }")
        L.append('  stop(paste("unknown state", state))')
        L.append("}")
        L.append("")
        L.append(f"effect_{k} <- function(state, cycle, tenure, v) {{")
        for i, code in s["effect"].items():
            L.append(f"  if (state == {i + 1}) return({code})  # {ids[i]}")
        L.append('  stop(paste("unknown state", state))')
        L.append("}")
        L.append("")
        L.append(f"reset_{k} <- function(k, state, cycle, tenure, trk) {{")
        L.append("  # Tracker k's reset value when the rule for `state` fires.")
        L.append("  v <- model_variables(cycle, tenure, trk)")
        for ti, si, code in s["resets"]:
            L.append(f"  if (k == {ti + 1} && state == {si + 1}) return({code})  # {g['trackers'][ti]['name']} @ {ids[si]}")
        L.append("  trk[k]")
        L.append("}")
        L.append("")
        L.append(f"init_{k} <- function() {{")
        L.append("  # Initial-state weights (normalised when sampling).")
        L.append("  c(" + ", ".join(code for _i, code in s["init"]) + ")")
        L.append("}")
        L.append("")
    L.append("# ---- Strategy table ----")
    L.append("STRATEGIES <- list(")
    entries = []
    for k, s in enumerate(g["strategies"], start=1):
        sir = s["ir"]
        real = "c(" + ", ".join("TRUE" if x else "FALSE" for x in sir["real"]) + ")"
        alias = "c(" + ", ".join("NA" if a is None else str(a + 1) for a in sir["alias"]) + ")"
        resets_rows = ", ".join(f"{ti + 1}, {si + 1}" for ti, si, _c in s["resets"])
        resets = f"matrix(c({resets_rows}), ncol = 2, byrow = TRUE)" if s["resets"] else "matrix(numeric(0), ncol = 2)"
        entries.append(
            "  list(\n"
            f"    id = {_r_str(sir['id'])}, name = {_r_str(sir['name'])},\n"
            f"    real = {real},\n"
            f"    alias = {alias},\n"
            f"    branch_nodes = c({', '.join(str(i + 1) for i in sir['branch_nodes'])}),\n"
            f"    tenure_states = c({', '.join(str(i + 1) for i in sir['tenure_states'])}),\n"
            f"    init_states = c({', '.join(str(i + 1) for i, _c in s['init'])}), init_weights = init_{k},\n"
            f"    rows = rows_{k}, cost = cost_{k}, effect = effect_{k}, reset = reset_{k},\n"
            f"    resets = {resets}\n"
            "  )"
        )
    L.append(",\n".join(entries))
    L.append(")")
    engine = _fill(
        _R_ENGINE,
        TENURE_KEY=_tenure_key_expr(ir["tenure_key"], "r"),
        CYCLE_KEY="cycle" if ir["cycle_key"] else "0",
    )
    return "\n".join(L) + "\n" + engine


# ---------------------------------------------------------------------------
# Stata (Mata)
# ---------------------------------------------------------------------------

_STATA_ENGINE = '''
// ---------------------------------------------------------------------------
// Simulation engine (generic: nothing below depends on the model)
// Nodes are numbered 1..n as listed in each strategy's header comment.
// ---------------------------------------------------------------------------
real matrix prepare_row(real matrix raw)
{
    // raw: 4 x m = (targets \\ probabilities \\ cost increments \\ effect increments).
    // Returns the same matrix with row 2 replaced by the cumulative distribution
    // (negative probabilities clipped to 0, the rest normalised, as DECIDE does).
    real matrix out
    real rowvector probs
    real scalar total
    out = raw
    probs = colmax((raw[2, .] \\ J(1, cols(raw), 0)))
    total = sum(probs)
    if (total > 0) probs = probs / total
    else probs = J(1, cols(raw), 1 / cols(raw))
    out[2, .] = runningsum(probs)
    return(out)
}

real scalar draw(real rowvector cdf, real scalar u)
{
    real scalar i
    for (i = 1; i <= cols(cdf); i++) {
        if (u < cdf[i]) return(i)
    }
    return(cols(cdf))
}

real scalar leave_probability(real scalar k, real scalar state, real scalar node, real scalar prob, real scalar cycle, real scalar tenure, real rowvector v)
{
    // Probability of ending the cycle in a different state than `state` (through
    // any chain of branch nodes) -- used for tenure-based cost.
    real matrix raw
    real scalar m, q, landed, leave
    raw = node_rows(k, node, cycle, tenure, v)
    leave = 0
    for (m = 1; m <= cols(raw); m++) {
        q = prob * raw[2, m]
        if (q == 0) continue
        if (NT[1, raw[1, m]] == 1) landed = raw[1, m]
        else if (NT[3, raw[1, m]] == 1) {
            leave = leave + leave_probability(k, state, raw[1, m], q, cycle, tenure, v)
            continue
        }
        else landed = NT[2, raw[1, m]]
        if (landed != state) leave = leave + q
    }
    return(leave)
}

// Returns (cost, effect) of one patient; half-cycle correction included.
real rowvector simulate_patient(real scalar k, real scalar seed)
{
    real matrix raw, row
    real rowvector trk, v, new_trk, result
    real scalar state, tenure, cycle, node, new_state, j, target, kk, entering
    real scalar cost, effect, exit_p, disc_cost, disc_effect
    real scalar total_cost, total_effect, first_cost, first_effect

    rseed(seed)
    row = prepare_row(init_row(k))
    state = row[1, draw(row[2, .], runiform(1, 1))]
    tenure = 1
    trk = TRACKER_INITIAL
    total_cost = 0; total_effect = 0; first_cost = 0; first_effect = 0
    for (cycle = 0; cycle <= CYCLES - 1; cycle++) {
        v = model_variables(trk, cycle, tenure)
        exit_p = 0
        if (NT[4, state] == 1) exit_p = leave_probability(k, state, state, 1, cycle, tenure, v)
        cost = state_cost(k, state, cycle, tenure, v, exit_p)
        effect = state_effect(k, state, cycle, tenure, v)
        node = state
        for (;;) {   // follow branch nodes down to the real state this cycle ends in
            row = prepare_row(node_rows(k, node, cycle, tenure, v))
            j = draw(row[2, .], runiform(1, 1))
            cost = cost + row[3, j]
            effect = effect + row[4, j]
            target = row[1, j]
            if (NT[1, target] == 1) {
                new_state = target
                break
            }
            if (NT[3, target] == 1) node = target
            else {
                new_state = NT[2, target]
                break
            }
        }
        disc_cost = cost / (1 + DISCOUNT_COST)^cycle
        disc_effect = effect / (1 + DISCOUNT_EFFECT)^cycle
        total_cost = total_cost + disc_cost
        total_effect = total_effect + disc_effect
        if (cycle == 0) {
            first_cost = disc_cost
            first_effect = disc_effect
        }
        entering = (new_state != state)
        new_trk = tracker_update(cycle, tenure, v)
        for (kk = 1; kk <= N_TRACKERS; kk++) {
            if (RESET_ON_ENTRY) {
                if (entering & RT[kk, new_state] == 1) new_trk[kk] = tracker_reset(k, kk, new_state, cycle, tenure, new_trk)
            }
            else if (!entering & tenure == 1 & RT[kk, state] == 1) {
                new_trk[kk] = tracker_reset(k, kk, state, cycle, tenure, new_trk)
            }
        }
        trk = new_trk
        tenure = cond(entering, 1, tenure + 1)
        state = new_state
    }
    if (HALF_CYCLE_CORRECTION & CYCLES > 0) {
        v = model_variables(trk, CYCLES, tenure)
        total_cost = total_cost + 0.5 * (state_cost(k, state, CYCLES, tenure, v, 0) / (1 + DISCOUNT_COST)^CYCLES - first_cost)
        total_effect = total_effect + 0.5 * (state_effect(k, state, CYCLES, tenure, v) / (1 + DISCOUNT_EFFECT)^CYCLES - first_effect)
    }
    return((total_cost, total_effect))
}

// Returns (mean cost, mean effect, SE cost, SE effect) for strategy k.
real rowvector run_strategy(real scalar k, real scalar n_patients, real scalar seed)
{
    real scalar i, sum_c, sum_e, sq_c, sq_e, mean_c, mean_e
    real rowvector r
    NT = node_table(k)
    RT = reset_table(k)
    sum_c = 0; sum_e = 0; sq_c = 0; sq_e = 0
    for (i = 1; i <= n_patients; i++) {
        r = simulate_patient(k, seed + i - 1)
        sum_c = sum_c + r[1]
        sum_e = sum_e + r[2]
        sq_c = sq_c + r[1]^2
        sq_e = sq_e + r[2]^2
    }
    mean_c = sum_c / n_patients
    mean_e = sum_e / n_patients
    if (n_patients > 1) {
        return((mean_c, mean_e, sqrt(max((sq_c / n_patients - mean_c^2, 0)) * n_patients / (n_patients - 1) / n_patients), sqrt(max((sq_e / n_patients - mean_e^2, 0)) * n_patients / (n_patients - 1) / n_patients)))
    }
    return((mean_c, mean_e, ., .))
}

string scalar icer_text(real scalar d_cost, real scalar d_effect)
{
    real scalar better
    if (d_effect == 0) return("n/a (no effect difference)")
    better = (EFFECT_SIGN * d_effect > 0)
    if (better & d_cost <= 0) return("dominant (cheaper and better)")
    if (!better & d_cost >= 0) return("dominated (costlier and not better)")
    return(strofreal(d_cost / d_effect, "%15.2fc"))
}

void main()
{
    real scalar k, n_strat
    real matrix res
    n_strat = cols(STRATEGY_NAMES)
    res = J(n_strat, 4, .)
    printf("Microsimulation: %g patients, %g cycles, seed %g\\n", N_PATIENTS, CYCLES, SEED)
    for (k = 1; k <= n_strat; k++) {
        res[k, .] = run_strategy(k, N_PATIENTS, SEED)
    }
    printf("\\n%-28s %16s %12s %14s %10s\\n", "Strategy", "Mean cost", "(SE)", "Mean effect", "(SE)")
    for (k = 1; k <= n_strat; k++) {
        printf("%-28s %16s %12s %14.4f %10.4f\\n", STRATEGY_NAMES[k], strofreal(res[k, 1], "%15.2fc"), strofreal(res[k, 3], "%11.2fc"), res[k, 2], res[k, 4])
    }
    if (n_strat > 1) {
        printf("\\nVersus %s:\\n", STRATEGY_NAMES[1])
        printf("%-28s %16s %14s    %s\\n", "Strategy", "Delta cost", "Delta effect", "ICER")
        for (k = 2; k <= n_strat; k++) {
            printf("%-28s %16s %14.4f    %s\\n", STRATEGY_NAMES[k], strofreal(res[k, 1] - res[1, 1], "%15.2fc"), res[k, 2] - res[1, 2], icer_text(res[k, 1] - res[1, 1], res[k, 2] - res[1, 2]))
        }
    }
}

main()

end
'''


def _stata_add_externals(text: str, scalars: list[str], others: list[tuple[str, str]]) -> str:
    """Mata functions only see session-level variables that they declare
    `external`; add the declarations each function body needs."""

    out: list[str] = []
    lines = text.split("\n")
    header = re.compile(r"^(void|real|string|transmorphic)\b.*\)\s*$")
    i = 0
    while i < len(lines):
        line = lines[i]
        out.append(line)
        if header.match(line) and i + 1 < len(lines) and lines[i + 1].strip() == "{" and not line.startswith("external"):
            # collect the body up to the closing brace at column 0
            j = i + 2
            while j < len(lines) and lines[j] != "}":
                j += 1
            body = "\n".join(re.sub(r"//.*", "", ln) for ln in lines[i + 2:j])
            words = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", body))
            out.append("{")
            used_scalars = [n for n in scalars if n in words]
            if used_scalars:
                out.append("    external real scalar " + ", ".join(used_scalars))
            for typ, name in others:
                if name in words:
                    out.append(f"    external {typ} {name}")
            i += 2
            continue
        i += 1
    return "\n".join(out)


def _render_stata(ir: dict[str, Any], tk: dict[str, Any]) -> str:
    g = _generic(ir, "stata", tk)
    num = g["num"]
    n_trk = len(g["trackers"])
    n_dyn = len(g["dynamic_rows"])
    L: list[str] = _comment_block(_header_lines(ir, g, "Run with: do <file>.do   (all code is Mata; edit N_PATIENTS / SEED below)"), "//") + [""]
    L.append("mata set matastrict off")
    L.append("mata:")
    L.append("")
    if g["helpers"]:
        L.append("// ---- Functions ----")
        for name in ("bound3", "RateToProb", "ProbToProb", "ProbToRate"):
            if name in g["helpers"]:
                L.append(tk["_EXPORT_HELPER_SOURCE"][name]["stata"])
        L.append("")
    L += [
        "// ---- Settings ----",
        "real scalar N_PATIENTS, SEED, CYCLES, HALF_CYCLE_CORRECTION, DISCOUNT_COST, DISCOUNT_EFFECT, RESET_ON_ENTRY, EFFECT_SIGN",
        f"N_PATIENTS = {ir['n_patients']}",
        f"SEED = {ir['seed']}",
        f"CYCLES = {ir['cycles']}",
        f"HALF_CYCLE_CORRECTION = {1 if ir['half_cycle_correction'] else 0}",
        f"DISCOUNT_COST = {num(ir['discount_cost'])}   // per cycle",
        f"DISCOUNT_EFFECT = {num(ir['discount_effect'])}   // per cycle",
        f"RESET_ON_ENTRY = {1 if ir['reset_on_entry'] else 0}   // Tracker reset timing",
        f"EFFECT_SIGN = {ir['effect_sign']}   // +1: higher effect is better",
        "",
    ]
    if g["param_rows"]:
        L.append("// ---- Parameters ----")
        for ident, name, base, source in g["param_rows"]:
            bits = ([] if ident == name else [name]) + ([source] if source else [])
            L.append(f"real scalar {ident}\n{ident} = {num(base)}" + (f"   // {' -- '.join(bits)}" if bits else ""))
        L.append("")
    if g["static_rows"]:
        L.append("// ---- Variables (static: same value every cycle) ----")
        for r in g["static_rows"]:
            if isinstance(r["raw"], str):
                L.append(f"// {r['name']} = {r['raw']}")
            L.append(f"real scalar {r['ident']}\n{r['ident']} = {r['code']}")
        L.append("")
    L.append("// ---- Trackers (per-patient values carried across cycles) ----")
    L.append(f"real scalar N_TRACKERS\nN_TRACKERS = {n_trk}")
    L.append("real rowvector TRACKER_INITIAL")
    L.append("TRACKER_INITIAL = " + ("(" + ", ".join(t["init"] for t in g["trackers"]) + ")" if n_trk else "J(1, 0, .)"))
    L.append("")
    L.append("// v = (tracker values, then the Variables that depend on cycle / tenure / Trackers):")
    names_v = [t["name"] for t in g["trackers"]] + [r["name"] for r in g["dynamic_rows"]]
    L.append("//   " + (", ".join(f"v[{i + 1}]={n}" for i, n in enumerate(names_v)) if names_v else "(empty)"))
    L.append("real rowvector model_variables(real rowvector trk, real scalar cycle, real scalar tenure)")
    L.append("{")
    L.append("    real rowvector v")
    L.append(f"    v = J(1, {n_trk + n_dyn}, 0)")
    for k in range(n_trk):
        L.append(f"    v[{k + 1}] = trk[{k + 1}]")
    for r in g["dynamic_rows"]:
        L.append(f"    {g['vref'](r['name'])} = {r['code']}   // {r['name']}")
    L.append("    return(v)")
    L.append("}")
    L.append("")
    L.append("real rowvector tracker_update(real scalar cycle, real scalar tenure, real rowvector v)")
    L.append("{")
    L.append("    // Each Tracker's update formula (evaluated in the start-of-cycle context).")
    L.append("    return(" + ("(" + ", ".join(t["update"] for t in g["trackers"]) + ")" if n_trk else "J(1, 0, .)") + ")")
    L.append("}")
    L.append("")

    n_strat = len(g["strategies"])
    node_tables, reset_tables = [], []
    for k, s in enumerate(g["strategies"], start=1):
        sir = s["ir"]
        ids = sir["ids"]
        n_nodes = len(ids)
        L.append(f"// ===== Strategy {k}: {sir['name']} (id {sir['id']}) =====")
        L.append("// node -> DECIDE id: " + ", ".join(f"{i + 1}={sid_}" + ("" if sir["real"][i] else " (branch)") for i, sid_ in enumerate(ids)))
        L.append(f"real matrix rows_{k}(real scalar node, real scalar cycle, real scalar tenure, real rowvector v)")
        L.append("{")
        L.append("    // Outgoing transitions of `node` as 4 x m: targets \\ probabilities \\ cost increments \\ effect increments.")
        for i, rows in s["rows"].items():
            L.append(f"    if (node == {i + 1}) {{   // {ids[i]}")
            L.append("        return((" + ", ".join(str(to + 1) for to, _p, _c, _e in rows) + " \\")
            L.append("                " + ", ".join(p for _t, p, _c, _e in rows) + " \\")
            L.append("                " + ", ".join(c for _t, _p, c, _e in rows) + " \\")
            L.append("                " + ", ".join(e for _t, _p, _c, e in rows) + "))")
            L.append("    }")
        L.append('    _error("unknown node")')
        L.append("}")
        L.append("")
        L.append(f"real scalar cost_{k}(real scalar state, real scalar cycle, real scalar tenure, real rowvector v, real scalar exit_p)")
        L.append("{")
        L.append("    // Cost of one cycle in `state` (exit_p = probability of leaving it, for tenure-based cost).")
        if any(isinstance(c, tuple) for c in s["cost"].values()):
            L.append("    real scalar base")
        for i, code in s["cost"].items():
            L.append(f"    if (state == {i + 1}) {{   // {ids[i]}")
            if isinstance(code, tuple):
                ini, rec, ter = code
                L.append(f"        base = cond(tenure == 1, {ini}, {rec} * (1 - exit_p))")
                L.append(f"        return(base + cond(tenure == 1, 0, {ter} * exit_p))")
            else:
                L.append(f"        return({code})")
            L.append("    }")
        L.append('    _error("unknown state")')
        L.append("}")
        L.append("")
        L.append(f"real scalar effect_{k}(real scalar state, real scalar cycle, real scalar tenure, real rowvector v)")
        L.append("{")
        for i, code in s["effect"].items():
            L.append(f"    if (state == {i + 1}) return({code})   // {ids[i]}")
        L.append('    _error("unknown state")')
        L.append("}")
        L.append("")
        L.append(f"real scalar reset_{k}(real scalar kk, real scalar state, real scalar cycle, real scalar tenure, real rowvector trk)")
        L.append("{")
        L.append("    // Tracker kk's reset value when the rule for `state` fires.")
        L.append("    real rowvector v")
        L.append("    v = model_variables(trk, cycle, tenure)")
        for ti, si, code in s["resets"]:
            L.append(f"    if (kk == {ti + 1} & state == {si + 1}) return({code})   // {g['trackers'][ti]['name']} @ {ids[si]}")
        L.append("    return(trk[kk])")
        L.append("}")
        L.append("")
        L.append(f"real matrix init_{k}()")
        L.append("{")
        L.append("    // Initial-state targets \\ weights (normalised when sampling).")
        L.append("    return((" + ", ".join(str(i + 1) for i, _c in s["init"]) + " \\ " + ", ".join(c for _i, c in s["init"]) + " \\ " + ", ".join("0" for _ in s["init"]) + " \\ " + ", ".join("0" for _ in s["init"]) + "))")
        L.append("}")
        L.append("")
        real_row = ", ".join("1" if x else "0" for x in sir["real"])
        alias_row = ", ".join("0" if a is None else str(a + 1) for a in sir["alias"])
        branch_row = ", ".join("1" if i in sir["branch_nodes"] else "0" for i in range(n_nodes))
        tenure_row = ", ".join("1" if i in sir["tenure_states"] else "0" for i in range(n_nodes))
        node_tables.append((k, f"({real_row} \\ {alias_row} \\ {branch_row} \\ {tenure_row})"))
        if n_trk:
            reset_rows = []
            for ti in range(n_trk):
                reset_rows.append(", ".join("1" if any(r[0] == ti and r[1] == i for r in s["resets"]) else "0" for i in range(n_nodes)))
            reset_tables.append((k, "(" + " \\ ".join(reset_rows) + ")"))
        else:
            reset_tables.append((k, f"J(0, {n_nodes}, 0)"))

    L.append("// ---- Per-strategy dispatch (generic wrappers) ----")
    L.append("string rowvector STRATEGY_NAMES")
    L.append("STRATEGY_NAMES = (" + ", ".join(_stata_str(s["ir"]["name"]) for s in g["strategies"]) + ")")
    L.append("// NT: 4 x nodes = (is real \\ alias target (0 = none) \\ is branch node with transitions \\ tenure-based cost)")
    L.append("real matrix NT, RT")
    L.append("")

    def dispatch(sig: str, call: str, ret: str) -> None:
        L.append(f"{ret} {sig}")
        L.append("{")
        for k in range(1, n_strat + 1):
            L.append(f"    if (k == {k}) return({call.replace('@', str(k))})")
        L.append('    _error("unknown strategy")')
        L.append("}")
        L.append("")

    dispatch("node_rows(real scalar k, real scalar node, real scalar cycle, real scalar tenure, real rowvector v)", "rows_@(node, cycle, tenure, v)", "real matrix")
    dispatch("state_cost(real scalar k, real scalar state, real scalar cycle, real scalar tenure, real rowvector v, real scalar exit_p)", "cost_@(state, cycle, tenure, v, exit_p)", "real scalar")
    dispatch("state_effect(real scalar k, real scalar state, real scalar cycle, real scalar tenure, real rowvector v)", "effect_@(state, cycle, tenure, v)", "real scalar")
    dispatch("tracker_reset(real scalar k, real scalar kk, real scalar state, real scalar cycle, real scalar tenure, real rowvector trk)", "reset_@(kk, state, cycle, tenure, trk)", "real scalar")
    dispatch("init_row(real scalar k)", "init_@()", "real matrix")
    for name, tables in (("node_table", node_tables), ("reset_table", reset_tables)):
        L.append(f"real matrix {name}(real scalar k)")
        L.append("{")
        for k, expr in tables:
            L.append(f"    if (k == {k}) return({expr})")
        L.append('    _error("unknown strategy")')
        L.append("}")
        L.append("")
    scalars = ["N_PATIENTS", "SEED", "CYCLES", "HALF_CYCLE_CORRECTION", "DISCOUNT_COST", "DISCOUNT_EFFECT",
               "RESET_ON_ENTRY", "EFFECT_SIGN", "N_TRACKERS"]
    scalars += [ident for ident, _n, _b, _s in g["param_rows"]] + [r["ident"] for r in g["static_rows"]]
    others = [("real rowvector", "TRACKER_INITIAL"), ("string rowvector", "STRATEGY_NAMES"), ("real matrix", "NT"), ("real matrix", "RT")]
    return _stata_add_externals("\n".join(L) + "\n" + _STATA_ENGINE, scalars, others)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def generate_microsim_export_code(document: dict[str, Any], language: str, toolkit: Any) -> str:
    """Standalone simulation script for a Microsimulation document (the full
    document with its ``strategies`` list). Raises ValueError with a clear
    message when the model uses something that can't be exported."""

    if language not in SUPPORTED_LANGUAGES:
        raise ValueError(f"Unsupported export language: {language}")
    tk = toolkit if isinstance(toolkit, dict) else {k: getattr(toolkit, k) for k in dir(toolkit)}
    ir = _microsim_export_ir(document, tk)
    renderers = {
        "python": _render_python,
        "r": _render_r,
        "stata": _render_stata,
        "javascript": _render_javascript,
    }
    return renderers[language](ir, tk)
