"""Reconstruct individual patient data from a published Kaplan-Meier curve (Guyot et al. 2012, BMC Med Res Methodol
12:9). Inputs per arm: digitised curve points (time, survival), the numbers at risk printed under the plot (times
and counts) and, when reported, the total number of events. Output: pseudo-IPD (time, event) and the events and
person-time per time interval, which feed the Poisson likelihood of the survival NMA (nma_survival).

Method: the curve is split into the intervals between the at-risk times. Within an interval the number of
censored patients is first estimated from the change of the at-risk count and the curve, censoring times are
spread uniformly, the events at each step of the curve follow from d = n (1 - S_j / S_{j-1}), and the censor count
is adjusted until the reconstructed number at risk at the start of the next interval equals the reported one.
The last interval has no next at-risk count: censoring there is set from the reported total events if given,
otherwise from the remaining patients at the last curve time. Accuracy depends on the quality of the digitised
points: report it and prefer real IPD when available."""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class ReconstructedArm:
    times: list[float] = field(default_factory=list)
    events: list[int] = field(default_factory=list)  # 1 = event, 0 = censored
    n: int = 0
    n_events: int = 0
    warnings: list[str] = field(default_factory=list)


def _clean_curve(points) -> list[tuple[float, float]]:
    pts = sorted((float(t), float(s)) for t, s in points)
    if not pts or pts[0][0] > 0:
        pts.insert(0, (0.0, 1.0))
    out: list[tuple[float, float]] = []
    for t, s in pts:
        s = min(1.0, max(0.0, s))
        if out and s > out[-1][1]:
            s = out[-1][1]  # enforce a non-increasing curve
        out.append((t, s))
    return out


def reconstruct_arm(curve, risk_times, risk_counts, total_events: int | None = None) -> ReconstructedArm:
    """curve: [(t, S)], risk_times / risk_counts: at-risk table (first time is 0 with the arm size)."""

    out = ReconstructedArm()
    pts = _clean_curve(curve)
    times = [float(t) for t in risk_times]
    counts = [float(n) for n in risk_counts]
    if len(times) != len(counts) or len(times) < 1 or times[0] != 0:
        out.warnings.append("The at-risk table must start at time 0 with the arm size")
        return out
    surv = lambda t: next((s for (tt, s) in reversed(pts) if tt <= t), 1.0)  # noqa: E731
    horizon = max(pts[-1][0], times[-1])
    bounds = times + ([horizon] if times[-1] < horizon else [])
    n_at = counts[0]
    out.n = int(round(n_at))
    ev_times: list[float] = []
    cen_times: list[float] = []
    n_int = len(bounds) - 1
    for i in range(n_int):
        a, b = bounds[i], bounds[i + 1]
        steps = [(t, s) for (t, s) in pts if a < t <= b]
        s_a = surv(a)
        has_next = i + 1 < len(counts)
        n_start = n_at
        # initial censoring estimate for the interval (Guyot eq. 1)
        if has_next and s_a > 0:
            n_cen = max(0, int(round(n_start * surv(b) / s_a - counts[i + 1])))
        else:
            n_cen = 0
        for _ in range(60):
            n_now, s_prev, ev, cens = n_start, s_a, [], []
            cen_left = n_cen
            cen_edges = [a + (b - a) * (k + 1) / (n_cen + 1) for k in range(n_cen)] if n_cen > 0 else []
            ci = 0
            for (t, s) in steps + [(b, surv(b))]:
                # censors occurring before this step
                while ci < len(cen_edges) and cen_edges[ci] < t and cen_left > 0:
                    cens.append(cen_edges[ci]); n_now -= 1; ci += 1; cen_left -= 1
                if t == b and (t, s) not in steps:
                    break
                d = int(round(n_now * (1 - s / s_prev))) if s_prev > 0 and n_now > 0 else 0
                d = max(0, min(d, int(n_now)))
                ev.extend([t] * d)
                n_now -= d
                s_prev = s if s < s_prev else s_prev
            n_end = n_now - (n_cen - len(cens))  # any censors not yet used
            if not has_next:
                break
            diff = int(round(n_end - counts[i + 1]))
            if diff == 0:
                break
            n_cen = max(0, n_cen + diff)
        if not has_next:
            # last interval: leftover after the curve; censor them at the horizon
            pass
        ev_times.extend(ev)
        cen_times.extend(cens)
        n_at = counts[i + 1] if has_next else max(0.0, n_start - len(ev) - len(cens))
    # everyone still at risk at the end is censored at the last time
    remaining = int(round(max(0.0, out.n - len(ev_times) - len(cen_times))))
    if total_events is not None and len(ev_times) != int(total_events):
        out.warnings.append(f"Reconstructed {len(ev_times)} events but {int(total_events)} were reported: check the digitised curve")
    cen_times.extend([horizon] * remaining)
    out.times = ev_times + cen_times
    out.events = [1] * len(ev_times) + [0] * len(cen_times)
    out.n_events = len(ev_times)
    return out


def interval_counts(arm: ReconstructedArm, breaks) -> list[tuple[float, float, int, float]]:
    """Events and person-time per interval of `breaks` (a list of cut points starting at 0):
    [(t_from, t_to, events, person_time)]."""

    edges = [float(b) for b in breaks]
    rows = []
    for a, b in zip(edges[:-1], edges[1:]):
        events, person = 0, 0.0
        for t, e in zip(arm.times, arm.events):
            if t > a:
                person += min(t, b) - a
                if e and t <= b:
                    events += 1
        rows.append((a, b, events, person))
    return rows


def kaplan_meier(times, events) -> list[tuple[float, float]]:
    """Product-limit estimate of (times, events) for checking a reconstruction."""

    order = sorted(range(len(times)), key=lambda i: times[i])
    n, s, out = len(times), 1.0, [(0.0, 1.0)]
    i = 0
    while i < len(order):
        t = times[order[i]]
        d = c = 0
        while i < len(order) and times[order[i]] == t:
            d += events[order[i]]
            c += 1 - events[order[i]]
            i += 1
        if d:
            s *= 1 - d / n
            out.append((t, s))
        n -= d + c
    return out
