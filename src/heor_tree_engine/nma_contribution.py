"""Contribution matrix for NMA (Papakonstantinou et al. 2018, Res Synth
Methods 9:1-12), the basis of CINeMA's within-study-bias and indirectness
judgements.

For every network estimate (treatment A vs B) the row of the hat matrix,
    h = x' (X' W X)^-1 X' W,     W = (V + tau^2 C)^-1,
gives the weight of each observed study contrast in the estimate. Seen as
a flow on the network graph, h carries one unit of "evidence flow" from B
(source) to A (sink) and satisfies flow conservation. The flow is split into
paths (shortest paths first); a path of length L carrying flow phi gives
phi / L to each of its edges. The contribution of a direct comparison is the
sum over the paths that use it; contributions of one row add up to 100 %.

Multi-arm studies are represented, as in netmeta (Ruecker & Schwarzer 2014),
by all pairwise comparisons of their arms with adjusted weights: with arm
variances v_i (plus tau^2/2 each for random effects) the pairwise weight is
w_ij = g_i g_j / sum(g), g = 1/v (the star-to-mesh transform of the arm
resistances), which makes the diagonal-weight pairwise model equivalent to
the GLS fit with the full multi-arm covariance. The hat matrix
H = X_p (X_p' W X_p)^-1 X_p' W of that pairwise model gives the flow of each
pairwise comparison directly, so the contributions match netmeta's
netcontrib(method = "shortestpath") up to tie-breaking between equally short
paths.

Study contributions split each edge's contribution among the studies on that
edge in proportion to the absolute flow they carry.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .models import ValidationError
from .nma_freq import build_study_blocks, run_nma_frequentist
from .nma_models import NetworkMetaAnalysis, NMAResult
from .nma_validation import validate_nma

Edge = tuple[str, str]  # canonical: (later, earlier) in the treatment order, i.e. "later vs earlier"


@dataclass
class ContributionRow:
    treatment: str
    versus: str
    edges: dict[Edge, float] = field(default_factory=dict)  # proportion (0-1)
    studies: dict[str, float] = field(default_factory=dict)  # proportion (0-1)
    direct_share: float = 0.0  # contribution of the direct comparison itself
    unexplained: float = 0.0  # flow left in circulations (normally ~0)

    @property
    def label(self) -> str:
        return f"{self.treatment} vs {self.versus}"


@dataclass
class ContributionResult:
    model: str = "random"
    tau2: float = 0.0
    treatments: list[str] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)  # direct comparisons
    edge_studies: dict[Edge, list[str]] = field(default_factory=dict)
    rows: list[ContributionRow] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[ValidationError] = field(default_factory=list)

    def row(self, treatment: str, versus: str) -> ContributionRow | None:
        for row in self.rows:
            if (row.treatment, row.versus) == (treatment, versus):
                return row
        return None

    def edge_label(self, edge: Edge) -> str:
        return f"{edge[0]} vs {edge[1]}"


def _decompose(flows: dict[Edge, float], source: str, sink: str, tolerance: float = 1e-10):
    """Shortest-path-first decomposition of an edge flow into source->sink
    paths. `flows` maps canonical edges to signed flow (positive = from
    edge[1] to edge[0]). Returns (edge contributions, total path flow)."""

    residual: dict[tuple[str, str], float] = {}
    for (a, b), value in flows.items():
        if value > tolerance:
            residual[(b, a)] = value  # directed b -> a
        elif value < -tolerance:
            residual[(a, b)] = -value
    contributions: dict[Edge, float] = {edge: 0.0 for edge in flows}
    total = 0.0
    for _ in range(10_000):
        adjacency: dict[str, list[str]] = {}
        for (u, v), value in residual.items():
            if value > tolerance:
                adjacency.setdefault(u, []).append(v)
        for u in adjacency:
            adjacency[u].sort()
        previous: dict[str, str | None] = {source: None}
        queue = deque([source])
        while queue and sink not in previous:
            u = queue.popleft()
            for v in adjacency.get(u, []):
                if v not in previous:
                    previous[v] = u
                    queue.append(v)
        if sink not in previous:
            break
        path = []
        node = sink
        while previous[node] is not None:
            path.append((previous[node], node))
            node = previous[node]  # type: ignore[assignment]
        phi = min(residual[step] for step in path)
        for u, v in path:
            residual[(u, v)] -= phi
            edge = (u, v) if (u, v) in contributions else (v, u)
            contributions[edge] += phi / len(path)
        total += phi
    return contributions, total


def contribution_matrix(model: NetworkMetaAnalysis, effects: str = "random", tau2: float | None = None,
                        nma: NMAResult | None = None) -> ContributionResult:
    result = ContributionResult(model=effects)
    result.errors = validate_nma(model)
    if result.errors:
        return result
    if tau2 is None:
        if effects == "random":
            nma = nma or run_nma_frequentist(model)
            if nma.errors:
                result.errors = nma.errors
                return result
            tau2 = nma.heterogeneity.tau2
        else:
            tau2 = 0.0
    tau2 = tau2 if effects == "random" else 0.0
    result.tau2 = tau2

    scratch = NMAResult()
    blocks = build_study_blocks(model, scratch)
    result.warnings.extend(scratch.warnings)
    used = {name for block in blocks for name in (block.baseline, *block.treatments)}
    treatments = [name for name in model.treatment_names if name in used]
    reference = model.reference
    if reference not in used:
        result.errors.append(ValidationError(code="reference_excluded", message="The reference has no usable data"))
        return result
    parameters = [name for name in treatments if name != reference]
    result.treatments = treatments
    position = {name: i for i, name in enumerate(treatments)}

    def canonical(a: str, b: str) -> tuple[Edge, float]:
        """Edge key and the sign of a flow from b to a along it."""
        return ((a, b), 1.0) if position[a] > position[b] else ((b, a), -1.0)

    # Pairwise-comparison design with adjusted weights (one row per pair of arms).
    index = {name: i for i, name in enumerate(parameters)}
    rows_design: list[np.ndarray] = []
    weights: list[float] = []
    pair_rows: list[tuple[str, str, str]] = []  # (study, a, b): row is the contrast b - a
    edge_studies: dict[Edge, list[str]] = {}
    for block in blocks:
        arms = [block.baseline, *block.treatments]
        arm_var = {block.baseline: block.baseline_variance + tau2 / 2.0}
        for name, variance in zip(block.treatments, block.variances):
            arm_var[name] = variance - block.baseline_variance + tau2 / 2.0
        if len(arms) == 2:
            pair_weight = {(arms[0], arms[1]): 1.0 / (block.variances[0] + tau2)}
        else:
            g = {name: 1.0 / max(arm_var[name], 1e-12) for name in arms}
            total_g = sum(g.values())
            pair_weight = {(a, b): g[a] * g[b] / total_g for i, a in enumerate(arms) for b in arms[i + 1:]}
        for (a, b), weight in pair_weight.items():
            row = np.zeros(len(parameters))
            if b in index:
                row[index[b]] += 1.0
            if a in index:
                row[index[a]] -= 1.0
            rows_design.append(row)
            weights.append(weight)
            pair_rows.append((block.study_id, a, b))
            edge, _ = canonical(a, b)
            edge_studies.setdefault(edge, [])
            if block.study_id not in edge_studies[edge]:
                edge_studies[edge].append(block.study_id)
    X = np.array(rows_design)
    if np.linalg.matrix_rank(X) < len(parameters):
        result.errors.append(ValidationError(code="not_estimable", message="The network is disconnected"))
        return result
    W = np.diag(weights)
    projector = np.linalg.inv(X.T @ W @ X) @ X.T @ W  # parameters x pairwise rows
    result.edges = sorted(edge_studies, key=lambda edge: (position[edge[1]], position[edge[0]]))
    result.edge_studies = edge_studies

    for j, later in enumerate(treatments):
        for earlier in treatments[:j]:
            x = np.zeros(len(parameters))
            if later in index:
                x[index[later]] += 1.0
            if earlier in index:
                x[index[earlier]] -= 1.0
            h = x @ projector
            edge_flow = {edge: 0.0 for edge in result.edges}
            study_flow: dict[Edge, dict[str, float]] = {edge: {} for edge in result.edges}
            for (study_id, a, b), value in zip(pair_rows, h):
                flow_a_to_b = float(value)  # weight of the contrast b - a = flow a -> b
                edge, sign = canonical(b, a)
                edge_flow[edge] += sign * flow_a_to_b
                study_flow[edge][study_id] = study_flow[edge].get(study_id, 0.0) + abs(flow_a_to_b)
            contributions, total = _decompose(edge_flow, source=earlier, sink=later)
            scale = total if total > 1e-12 else 1.0
            record = ContributionRow(treatment=later, versus=earlier)
            record.unexplained = max(0.0, 1.0 - total)
            for edge in result.edges:
                share = contributions.get(edge, 0.0) / scale
                record.edges[edge] = share
                carried = study_flow[edge]
                carried_total = sum(carried.values())
                if share > 0 and carried_total > 0:
                    for study_id, value in carried.items():
                        record.studies[study_id] = record.studies.get(study_id, 0.0) + share * value / carried_total
            record.direct_share = record.edges.get((later, earlier), 0.0)
            result.rows.append(record)
    return result
