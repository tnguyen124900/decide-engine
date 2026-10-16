"""Incremental cost-effectiveness analysis across several strategies.

`incremental_analysis` ranks strategies by cost and compares each one with the
previous strategy that is *not dominated*, not simply with the row above it:

* **Strict dominance** -- a strategy that costs at least as much as a cheaper
  one but does not add effect is dominated.
* **Extended (weak) dominance** -- a strategy whose ICER against the previous
  non-dominated strategy is higher than the ICER of the next, more effective
  strategy is also removed, because a mix of its neighbours beats it.

Effects are assumed to be "higher is better" (QALYs, life-years); pass costs and
effects as plain floats.  This is the standard ICER table of CEA reporting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Iterable, Sequence

REFERENCE = "reference"
FRONTIER = "frontier"
DOMINATED = "dominated"
EXTENDED_DOMINATED = "extended_dominated"


@dataclass(frozen=True)
class IncrementalRow:
    """One strategy in the ranked table."""

    strategy_id: Hashable
    cost: float
    effect: float
    status: str  # REFERENCE | FRONTIER | DOMINATED | EXTENDED_DOMINATED
    compared_with: Hashable | None  # strategy the increments are measured against
    incremental_cost: float | None
    incremental_effect: float | None
    icer: float | None  # None for the reference and for dominated strategies


def incremental_analysis(strategies: Iterable[tuple[Hashable, float, float]]) -> list[IncrementalRow]:
    """Return one row per strategy, ordered by cost (cheapest first).

    `strategies` is an iterable of ``(id, total_cost, total_effect)``.  The
    cheapest strategy is the reference.  Dominated strategies are listed with
    their increments measured against the last non-dominated strategy that is
    cheaper, and no ICER.
    """

    items = sorted(strategies, key=lambda item: (item[1], -item[2]))
    if not items:
        return []

    # 1. strict dominance
    kept: list[tuple[Hashable, float, float]] = []
    status: dict[Hashable, str] = {}
    for sid, cost, effect in items:
        if kept and effect <= kept[-1][2]:
            status[sid] = DOMINATED
        else:
            kept.append((sid, cost, effect))
            status[sid] = FRONTIER

    # 2. extended dominance: ICERs must increase along the frontier
    changed = True
    while changed and len(kept) > 2:
        changed = False
        icers = [
            (kept[i][1] - kept[i - 1][1]) / (kept[i][2] - kept[i - 1][2])
            for i in range(1, len(kept))
        ]
        for i in range(1, len(icers)):
            if icers[i] < icers[i - 1]:
                status[kept[i][0]] = EXTENDED_DOMINATED
                del kept[i]
                changed = True
                break
    status[kept[0][0]] = REFERENCE

    rows: list[IncrementalRow] = []
    frontier_so_far: tuple[Hashable, float, float] | None = None
    final_frontier = {sid for sid, _, _ in kept}
    for sid, cost, effect in items:
        if frontier_so_far is None:
            rows.append(IncrementalRow(sid, cost, effect, REFERENCE, None, None, None, None))
            frontier_so_far = (sid, cost, effect)
            continue
        ref_id, ref_cost, ref_effect = frontier_so_far
        d_cost, d_effect = cost - ref_cost, effect - ref_effect
        if sid in final_frontier:
            rows.append(IncrementalRow(sid, cost, effect, FRONTIER, ref_id, d_cost, d_effect, d_cost / d_effect))
            frontier_so_far = (sid, cost, effect)
        else:
            rows.append(IncrementalRow(sid, cost, effect, status[sid], ref_id, d_cost, d_effect, None))
    return rows


def format_rows(rows: Sequence[IncrementalRow]) -> dict[Hashable, IncrementalRow]:
    """Index the rows by strategy id (helper for UI code)."""

    return {row.strategy_id: row for row in rows}
