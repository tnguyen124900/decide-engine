"""Inconsistency checks for a frequentist NMA.

Two complementary checks, both run on the same study contrasts as
nma_freq.run_nma_frequentist:

1. Node-splitting (local). For every comparison A-B that has direct evidence
   AND an independent indirect route, the studies containing both A and B get
   an extra parameter theta on their B-vs-A contrast ("separating indirect
   from direct evidence", SIDE; Dias et al. 2010, Stat Med 29:932). Then
       indirect = d_B - d_A          (all other evidence)
       direct   = d_B - d_A + theta  (the A-B studies)
       difference = theta,  z = theta / se(theta)
   A multi-arm study containing A and B is re-expressed against A, so only
   its B-vs-A row carries theta and its other arms still inform the
   consistency part. Random effects reuse the consistency model's tau^2.

2. Design-by-treatment interaction (global; Higgins et al. 2012,
   Krahn et al. 2013). A "design" is the set of treatments a study compares.
   Fitting every design its own effects (the inconsistency model) splits
       Q_total = Q_within designs (heterogeneity) + Q_between designs (inconsistency).
   Q_between is also reported after allowing for heterogeneity: tau^2 is
   estimated within designs and Q_between recomputed with V + tau^2 C.

Only numpy is needed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .models import ValidationError
from .nma_freq import (
    _StudyBlock,
    _assemble,
    _gls,
    build_study_blocks,
    chi2_sf,
    normal_cdf,
)
from .nma_models import NetworkMetaAnalysis, NMAResult, is_ratio_measure
from .nma_parser import _normal_quantile
from .nma_validation import validate_nma


@dataclass
class NodeSplit:
    treatment: str
    versus: str
    model: str  # "fixed" | "random"
    n_direct_studies: int
    direct: float
    direct_se: float
    indirect: float
    indirect_se: float
    network: float
    network_se: float
    difference: float  # direct - indirect (analysis scale)
    difference_se: float
    z: float
    p_value: float
    # Natural scale (exp for ratio measures) and CIs of direct / indirect.
    natural_direct: float = 0.0
    natural_indirect: float = 0.0
    natural_network: float = 0.0
    direct_lower: float = 0.0
    direct_upper: float = 0.0
    indirect_lower: float = 0.0
    indirect_upper: float = 0.0
    # Ratio of indirect to direct estimate on the natural scale (ROR for OR).
    natural_difference: float = 0.0


@dataclass
class DesignQ:
    design: tuple[str, ...]
    n_studies: int
    q: float
    df: int


@dataclass
class DesignByTreatment:
    q_total: float
    df_total: int
    q_within: float  # heterogeneity within designs
    df_within: int
    q_between: float  # inconsistency between designs (fixed effect)
    df_between: int
    p_within: float | None
    p_between: float | None
    tau2_within: float
    q_between_random: float | None  # after allowing for heterogeneity
    p_between_random: float | None
    designs: list[DesignQ] = field(default_factory=list)


@dataclass
class InconsistencyResult:
    measure: str = "OR"
    method: str = "side"  # node-splitting method: "side" | "sidde"
    splits_fixed: list[NodeSplit] = field(default_factory=list)
    splits_random: list[NodeSplit] = field(default_factory=list)
    not_splittable: list[tuple[str, str]] = field(default_factory=list)
    design_by_treatment: DesignByTreatment | None = None
    warnings: list[str] = field(default_factory=list)
    errors: list[ValidationError] = field(default_factory=list)


# --------------------------------------------------------------------------
# Block helpers
# --------------------------------------------------------------------------


def _all_treatments(block: _StudyBlock) -> list[str]:
    return [block.baseline, *block.treatments]


def rebase_block(block: _StudyBlock, new_baseline: str) -> _StudyBlock:
    """Re-express a study's contrasts against another of its arms.

    With arm values x and arm variances v: y_k = x_k - x_o, Var = v_k + v_o,
    Cov(y_k, y_l) = v_o. Against a new baseline a: y'_k = y_k - y_a,
    Var = v_k + v_a, Cov = v_a. Arm variances are recovered from the block.
    """

    if new_baseline == block.baseline:
        return block
    old = block.baseline
    value = {old: 0.0, **dict(zip(block.treatments, block.estimates))}
    arm_var = {old: block.baseline_variance}
    for name, variance in zip(block.treatments, block.variances):
        arm_var[name] = variance - block.baseline_variance
    others = [name for name in _all_treatments(block) if name != new_baseline]
    return _StudyBlock(
        block.study_id,
        new_baseline,
        others,
        [value[name] - value[new_baseline] for name in others],
        [arm_var[name] + arm_var[new_baseline] for name in others],
        arm_var[new_baseline],
    )


def _prepare(model: NetworkMetaAnalysis):
    result = NMAResult(reference=model.reference, measure=model.spec.measure)
    blocks = build_study_blocks(model, result)
    used = {name for block in blocks for name in _all_treatments(block)}
    treatments = [name for name in model.treatment_names if name in used]
    reference = model.reference if model.reference in used else (treatments[0] if treatments else None)
    parameters = [name for name in treatments if name != reference]
    return blocks, treatments, reference, parameters, result.warnings


def _consistency_tau2(y, X, V, C) -> tuple[float, float, int]:
    d, cov, W = _gls(y, X, V)
    r = y - X @ d
    q = float(r @ W @ r)
    df = len(y) - X.shape[1]
    P = W - W @ X @ cov @ X.T @ W
    trace = float(np.trace(P @ C))
    tau2 = max(0.0, (q - df) / trace) if df > 0 and trace > 0 else 0.0
    return tau2, q, df


# --------------------------------------------------------------------------
# Node-splitting
# --------------------------------------------------------------------------


def _natural(value: float, ratio: bool) -> float:
    return math.exp(value) if ratio else value


def _split_one(blocks, parameters, a: str, b: str, tau2: float, model_name: str,
               ratio: bool, z_ci: float) -> NodeSplit | None:
    rebased = [rebase_block(block, a) if (a in _all_treatments(block) and b in _all_treatments(block)) else block
               for block in blocks]
    y, X, V, C = _assemble(rebased, parameters)
    Z = np.zeros((len(y), 1))
    row = 0
    n_direct = 0
    for block in rebased:
        if block.baseline == a and b in block.treatments:
            Z[row + block.treatments.index(b), 0] = 1.0
            n_direct += 1
        row += len(block.treatments)
    if n_direct == 0:
        return None
    XZ = np.hstack([X, Z])
    if np.linalg.matrix_rank(XZ) < XZ.shape[1]:
        return None  # no indirect route independent of the A-B studies
    params, cov, _ = _gls(y, XZ, V + tau2 * C)

    # Contrast vector for d_B - d_A (reference has no column: d_ref = 0).
    contrast = np.zeros(XZ.shape[1])
    index = {name: i for i, name in enumerate(parameters)}
    if b in index:
        contrast[index[b]] += 1.0
    if a in index:
        contrast[index[a]] -= 1.0
    theta = np.zeros(XZ.shape[1])
    theta[-1] = 1.0
    direct_vec = contrast + theta

    indirect = float(contrast @ params)
    indirect_se = math.sqrt(max(float(contrast @ cov @ contrast), 0.0))
    direct = float(direct_vec @ params)
    direct_se = math.sqrt(max(float(direct_vec @ cov @ direct_vec), 0.0))
    difference = float(params[-1])
    difference_se = math.sqrt(max(float(cov[-1, -1]), 0.0))
    z = difference / difference_se if difference_se > 0 else 0.0
    p_value = 2.0 * (1.0 - normal_cdf(abs(z)))

    # Network (consistency) estimate with the same tau^2.
    y0, X0, V0, C0 = _assemble(blocks, parameters)
    d0, cov0, _ = _gls(y0, X0, V0 + tau2 * C0)
    contrast0 = contrast[:-1]
    network = float(contrast0 @ d0)
    network_se = math.sqrt(max(float(contrast0 @ cov0 @ contrast0), 0.0))

    return NodeSplit(
        treatment=b, versus=a, model=model_name, n_direct_studies=n_direct,
        direct=direct, direct_se=direct_se, indirect=indirect, indirect_se=indirect_se,
        network=network, network_se=network_se,
        difference=difference, difference_se=difference_se, z=z, p_value=p_value,
        natural_direct=_natural(direct, ratio), natural_indirect=_natural(indirect, ratio),
        natural_network=_natural(network, ratio),
        direct_lower=_natural(direct - z_ci * direct_se, ratio),
        direct_upper=_natural(direct + z_ci * direct_se, ratio),
        indirect_lower=_natural(indirect - z_ci * indirect_se, ratio),
        indirect_upper=_natural(indirect + z_ci * indirect_se, ratio),
        natural_difference=_natural(difference, ratio),
    )


# --------------------------------------------------------------------------
# Design-by-treatment interaction
# --------------------------------------------------------------------------


def _design_key(block: _StudyBlock, order: dict[str, int]) -> tuple[str, ...]:
    return tuple(sorted(set(_all_treatments(block)), key=lambda name: order.get(name, len(order))))


def _inconsistency_design(blocks, treatments) -> tuple[list[_StudyBlock], np.ndarray, list[tuple[str, ...]], list[int]]:
    """Blocks re-based to each design's first treatment, and the design
    matrix with one free effect per (design, non-baseline treatment)."""

    order = {name: i for i, name in enumerate(treatments)}
    designs: list[tuple[str, ...]] = []
    rebased: list[_StudyBlock] = []
    design_of_block: list[int] = []
    for block in blocks:
        key = _design_key(block, order)
        if key not in designs:
            designs.append(key)
        design_of_block.append(designs.index(key))
        rebased.append(rebase_block(block, key[0]))
    columns: dict[tuple[int, str], int] = {}
    for d_index, key in enumerate(designs):
        for name in key[1:]:
            columns[(d_index, name)] = len(columns)
    n_obs = sum(len(block.treatments) for block in rebased)
    X = np.zeros((n_obs, len(columns)))
    row = 0
    for block, d_index in zip(rebased, design_of_block):
        for offset, name in enumerate(block.treatments):
            X[row + offset, columns[(d_index, name)]] = 1.0
        row += len(block.treatments)
    return rebased, X, designs, design_of_block


def design_by_treatment(blocks, treatments, parameters) -> DesignByTreatment:
    y, Xc, V, C = _assemble(blocks, parameters)
    d_c, _, W = _gls(y, Xc, V)
    r_c = y - Xc @ d_c
    q_total = float(r_c @ W @ r_c)
    df_total = len(y) - Xc.shape[1]

    rebased, Xi, designs, design_of_block = _inconsistency_design(blocks, treatments)
    yi, _, Vi, Ci = _assemble(rebased, parameters)
    d_i, cov_i, Wi = _gls(yi, Xi, Vi)
    r_i = yi - Xi @ d_i
    q_within = float(r_i @ Wi @ r_i)
    df_within = len(yi) - Xi.shape[1]
    q_between = max(q_total - q_within, 0.0)
    df_between = df_total - df_within

    # Per-design contributions (V is block-diagonal by study).
    per_design: dict[int, list[float]] = {}
    row = 0
    for block, d_index in zip(rebased, design_of_block):
        k = len(block.treatments)
        rows = slice(row, row + k)
        contribution = float(r_i[rows] @ Wi[rows, rows] @ r_i[rows])
        entry = per_design.setdefault(d_index, [0.0, 0, 0])
        entry[0] += contribution
        entry[1] += 1
        entry[2] += k
        row += k
    design_rows = [
        DesignQ(design=designs[i], n_studies=int(n), q=q, df=int(obs - (len(designs[i]) - 1)))
        for i, (q, n, obs) in per_design.items()
    ]

    # tau^2 within designs, then Q between designs allowing for it.
    P = Wi - Wi @ Xi @ cov_i @ Xi.T @ Wi
    trace = float(np.trace(P @ Ci))
    tau2_within = max(0.0, (q_within - df_within) / trace) if df_within > 0 and trace > 0 else 0.0
    q_between_random = p_between_random = None
    if df_between > 0:
        dc_r, _, Wc_r = _gls(y, Xc, V + tau2_within * C)
        rc_r = y - Xc @ dc_r
        di_r, _, Wi_r = _gls(yi, Xi, Vi + tau2_within * Ci)
        ri_r = yi - Xi @ di_r
        q_between_random = max(float(rc_r @ Wc_r @ rc_r) - float(ri_r @ Wi_r @ ri_r), 0.0)
        p_between_random = chi2_sf(q_between_random, df_between)

    return DesignByTreatment(
        q_total=q_total, df_total=df_total,
        q_within=q_within, df_within=df_within,
        q_between=q_between, df_between=df_between,
        p_within=chi2_sf(q_within, df_within) if df_within > 0 else None,
        p_between=chi2_sf(q_between, df_between) if df_between > 0 else None,
        tau2_within=tau2_within,
        q_between_random=q_between_random, p_between_random=p_between_random,
        designs=design_rows,
    )


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


SPLIT_METHODS = ("side", "sidde")


def _sub_network_effect(blocks, a: str, b: str, tau2: float) -> tuple[float, float] | None:
    """Estimate of b vs a (analysis scale) and its SE from an NMA of `blocks`
    alone, with a as the reference; None when b is not connected to a."""

    if not blocks:
        return None
    used = []
    for block in blocks:
        for name in _all_treatments(block):
            if name not in used:
                used.append(name)
    if a not in used or b not in used:
        return None
    parameters = [name for name in used if name != a]
    y, X, V, C = _assemble(blocks, parameters)
    if np.linalg.matrix_rank(X) < len(parameters):
        # Keep only the component connected to a.
        component = {a}
        changed = True
        while changed:
            changed = False
            for block in blocks:
                names = set(_all_treatments(block))
                if names & component and not names <= component:
                    component |= names
                    changed = True
        if b not in component:
            return None
        blocks = [block for block in blocks if set(_all_treatments(block)) <= component]
        parameters = [name for name in used if name != a and name in component]
        y, X, V, C = _assemble(blocks, parameters)
        if np.linalg.matrix_rank(X) < len(parameters):
            return None
    d, cov, _ = _gls(y, X, V + tau2 * C)
    i = parameters.index(b)
    return float(d[i]), math.sqrt(max(float(cov[i, i]), 0.0))


def _split_sidde(blocks, parameters, a: str, b: str, tau2: float, model_name: str,
                 ratio: bool, z_ci: float) -> NodeSplit | None:
    """SIDDE (Efthimiou et al. 2019), as netmeta's netsplit(): the direct
    estimate is an NMA of the studies (designs) that contain both a and b
    (tau^2 from the full network); the indirect estimate is back-calculated
    from the network and direct estimates on the precision scale,
        1/se_ind^2 = 1/se_net^2 - 1/se_dir^2,
        indirect = (network/se_net^2 - direct/se_dir^2) * se_ind^2,
    and the test treats direct and indirect as independent."""

    with_both = [block for block in blocks if a in _all_treatments(block) and b in _all_treatments(block)]
    if not with_both:
        return None
    direct = _sub_network_effect(with_both, a, b, tau2)
    if direct is None:
        return None
    direct, direct_se = direct
    y0, X0, V0, C0 = _assemble(blocks, parameters)
    d0, cov0, _ = _gls(y0, X0, V0 + tau2 * C0)
    contrast = np.zeros(len(parameters))
    index = {name: i for i, name in enumerate(parameters)}
    if b in index:
        contrast[index[b]] += 1.0
    if a in index:
        contrast[index[a]] -= 1.0
    network = float(contrast @ d0)
    network_se = math.sqrt(max(float(contrast @ cov0 @ contrast), 0.0))
    precision_indirect = 1.0 / network_se ** 2 - 1.0 / direct_se ** 2 if network_se > 0 and direct_se > 0 else 0.0
    if precision_indirect <= 1e-12:
        return None  # all evidence is direct: nothing to split
    indirect_se = math.sqrt(1.0 / precision_indirect)
    indirect = (network / network_se ** 2 - direct / direct_se ** 2) / precision_indirect
    difference = direct - indirect
    difference_se = math.sqrt(direct_se ** 2 + indirect_se ** 2)
    z = difference / difference_se if difference_se > 0 else 0.0
    p_value = 2.0 * (1.0 - normal_cdf(abs(z)))
    return NodeSplit(
        treatment=b, versus=a, model=model_name, n_direct_studies=len(with_both),
        direct=direct, direct_se=direct_se, indirect=indirect, indirect_se=indirect_se,
        network=network, network_se=network_se,
        difference=difference, difference_se=difference_se, z=z, p_value=p_value,
        natural_direct=_natural(direct, ratio), natural_indirect=_natural(indirect, ratio),
        natural_network=_natural(network, ratio),
        direct_lower=_natural(direct - z_ci * direct_se, ratio),
        direct_upper=_natural(direct + z_ci * direct_se, ratio),
        indirect_lower=_natural(indirect - z_ci * indirect_se, ratio),
        indirect_upper=_natural(indirect + z_ci * indirect_se, ratio),
        natural_difference=_natural(difference, ratio),
    )


def run_inconsistency(model: NetworkMetaAnalysis, method: str = "side") -> InconsistencyResult:
    """method: "side" (Dias 2010 / NICE TSD 4 - the direct parameter is
    attached to the a-b contrast of every study containing both, so
    multi-arm studies still feed the indirect estimate through their other
    arms) or "sidde" (Efthimiou 2019, netmeta's default - whole designs
    containing a and b form the direct evidence, the indirect estimate is
    back-calculated from the network and direct estimates)."""

    result = InconsistencyResult(measure=model.spec.measure, method=method)
    if method not in SPLIT_METHODS:
        result.errors.append(ValidationError(code="unknown_split_method", message=f"Unknown node-split method '{method}'"))
        return result
    result.errors = validate_nma(model)
    if result.errors:
        return result
    blocks, treatments, reference, parameters, warnings = _prepare(model)
    result.warnings.extend(warnings)
    if reference is None or len(treatments) < 2:
        result.errors.append(ValidationError(code="not_estimable", message="Not enough treatments with usable data"))
        return result

    y, X, V, C = _assemble(blocks, parameters)
    if np.linalg.matrix_rank(X) < len(parameters):
        result.errors.append(ValidationError(
            code="not_estimable",
            message="Some treatment effects cannot be estimated (the network is disconnected after exclusions)",
        ))
        return result
    tau2, _, _ = _consistency_tau2(y, X, V, C)
    ratio = is_ratio_measure(model.spec.measure)
    z_ci = _normal_quantile(0.5 + model.spec.ci_level / 2)
    order = {name: i for i, name in enumerate(treatments)}

    pairs: list[tuple[str, str]] = []
    for block in blocks:
        names = sorted(set(_all_treatments(block)), key=lambda name: order[name])
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                if (a, b) not in pairs:
                    pairs.append((a, b))
    pairs.sort(key=lambda pair: (order[pair[0]], order[pair[1]]))

    split = _split_sidde if method == "sidde" else _split_one
    for a, b in pairs:
        fixed = split(blocks, parameters, a, b, 0.0, "fixed", ratio, z_ci)
        if fixed is None:
            result.not_splittable.append((b, a))
            continue
        result.splits_fixed.append(fixed)
        random = split(blocks, parameters, a, b, tau2, "random", ratio, z_ci)
        if random is not None:
            result.splits_random.append(random)

    if not result.splits_fixed:
        result.warnings.append("No comparison has both direct and independent indirect evidence (no closed loops), so node-splitting is not possible.")
    result.design_by_treatment = design_by_treatment(blocks, treatments, parameters)
    return result
