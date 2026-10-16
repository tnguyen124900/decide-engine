"""Graph structure of an NMA: edges, connectivity, closed loops.

Pure Python (no networkx) so it can run anywhere the engine runs. Feeds the
network plot (node size = patients, edge width = number of studies), the
"network connected" banner, and - later - node-splitting, which needs the
closed loops.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from .nma_models import NetworkMetaAnalysis


@dataclass
class NetworkEdge:
    treatments: tuple[str, str]  # sorted pair
    study_ids: list[str] = field(default_factory=list)
    patients: float = 0.0  # patients in the two arms, summed over studies

    @property
    def n_studies(self) -> int:
        return len(self.study_ids)


@dataclass
class ClosedLoop:
    treatments: list[str]  # cycle in order, first treatment not repeated
    # True when every edge of the loop comes from one multi-arm study: such a
    # loop is consistent by design and cannot be checked for inconsistency.
    single_study: bool = False


@dataclass
class NetworkSummary:
    treatments: list[str] = field(default_factory=list)
    edges: dict[tuple[str, str], NetworkEdge] = field(default_factory=dict)
    patients: dict[str, float] = field(default_factory=dict)
    n_studies: dict[str, int] = field(default_factory=dict)
    components: list[list[str]] = field(default_factory=list)
    loops: list[ClosedLoop] = field(default_factory=list)
    multi_arm_studies: list[str] = field(default_factory=list)
    n_arms: int = 0

    @property
    def is_connected(self) -> bool:
        return len(self.components) <= 1

    @property
    def n_independent_loops(self) -> int:
        """Loops that carry independent evidence (excludes single-study loops)."""

        return sum(1 for loop in self.loops if not loop.single_study)

    def neighbours(self, treatment: str) -> list[str]:
        out = []
        for a, b in self.edges:
            if a == treatment:
                out.append(b)
            elif b == treatment:
                out.append(a)
        return out


def _pair(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def _arm_patients(model: NetworkMetaAnalysis) -> dict[tuple[str, str], float]:
    """Patients per (study, treatment); contrast-level studies contribute 0."""

    counts: dict[tuple[str, str], float] = {}
    for study in model.studies:
        for arm in study.arms:
            if arm.n is not None:
                counts[(study.id, arm.treatment)] = counts.get((study.id, arm.treatment), 0.0) + arm.n
    return counts


def summarize_network(model: NetworkMetaAnalysis) -> NetworkSummary:
    summary = NetworkSummary(treatments=list(model.treatment_names))
    for name in summary.treatments:
        summary.patients[name] = 0.0
        summary.n_studies[name] = 0
    arm_patients = _arm_patients(model)

    for study in model.studies:
        names = list(dict.fromkeys(study.treatments))
        summary.n_arms += len(names)
        if len(names) > 2:
            summary.multi_arm_studies.append(study.id)
        for name in names:
            if name not in summary.patients:
                summary.treatments.append(name)
                summary.patients[name] = 0.0
                summary.n_studies[name] = 0
            summary.n_studies[name] += 1
            summary.patients[name] += arm_patients.get((study.id, name), 0.0)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                key = _pair(a, b)
                edge = summary.edges.setdefault(key, NetworkEdge(treatments=key))
                edge.study_ids.append(study.id)
                edge.patients += arm_patients.get((study.id, a), 0.0) + arm_patients.get((study.id, b), 0.0)

    summary.components = _components(summary)
    summary.loops = _fundamental_loops(summary, model)
    return summary


def _adjacency(summary: NetworkSummary) -> dict[str, list[str]]:
    adjacency: dict[str, list[str]] = {name: [] for name in summary.treatments}
    for a, b in summary.edges:
        adjacency[a].append(b)
        adjacency[b].append(a)
    return adjacency


def _components(summary: NetworkSummary) -> list[list[str]]:
    adjacency = _adjacency(summary)
    seen: set[str] = set()
    components: list[list[str]] = []
    for start in summary.treatments:
        if start in seen:
            continue
        component = []
        queue = deque([start])
        seen.add(start)
        while queue:
            node = queue.popleft()
            component.append(node)
            for nxt in adjacency[node]:
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)
        components.append(component)
    return components


def _fundamental_loops(summary: NetworkSummary, model: NetworkMetaAnalysis) -> list[ClosedLoop]:
    """One cycle per non-tree edge of a BFS spanning forest (a cycle basis)."""

    adjacency = _adjacency(summary)
    parent: dict[str, str | None] = {}
    depth: dict[str, int] = {}
    tree_edges: set[tuple[str, str]] = set()
    for start in summary.treatments:
        if start in parent:
            continue
        parent[start] = None
        depth[start] = 0
        queue = deque([start])
        while queue:
            node = queue.popleft()
            for nxt in adjacency[node]:
                if nxt not in parent:
                    parent[nxt] = node
                    depth[nxt] = depth[node] + 1
                    tree_edges.add(_pair(node, nxt))
                    queue.append(nxt)

    study_sets = {study.id: set(study.treatments) for study in model.studies}
    loops: list[ClosedLoop] = []
    for a, b in summary.edges:
        if (a, b) in tree_edges:
            continue
        # Walk both ends up to their lowest common ancestor.
        path_a, path_b = [a], [b]
        x, y = a, b
        while depth[x] > depth[y]:
            x = parent[x]  # type: ignore[assignment]
            path_a.append(x)
        while depth[y] > depth[x]:
            y = parent[y]  # type: ignore[assignment]
            path_b.append(y)
        while x != y:
            x = parent[x]  # type: ignore[assignment]
            y = parent[y]  # type: ignore[assignment]
            path_a.append(x)
            path_b.append(y)
        cycle = path_a + list(reversed(path_b[:-1]))
        members = set(cycle)
        cycle_edges = [_pair(cycle[i], cycle[(i + 1) % len(cycle)]) for i in range(len(cycle))]
        # Consistent by design when every study on every edge of the loop
        # contains all of the loop's treatments (i.e. only multi-arm studies
        # spanning the whole loop supply it).
        single = all(
            members <= study_sets.get(study_id, set())
            for edge in cycle_edges
            for study_id in summary.edges[edge].study_ids
        )
        loops.append(ClosedLoop(treatments=cycle, single_study=single))
    return loops
