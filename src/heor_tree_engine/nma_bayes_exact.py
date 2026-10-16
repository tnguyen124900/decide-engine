"""Bayesian NMA with the exact arm-level likelihood (NICE DSU TSD 2) by MCMC,
plus the unrelated mean effects (UME) inconsistency model (TSD 4).

Arm-based model (study i, arm j, baseline arm b = 0):
    binary      r_ij ~ Binomial(n_ij, p_ij),  logit(p_ij) = mu_i + delta_ij   (OR)
    count       r_ij ~ Poisson(E_ij lam_ij),  log(lam_ij) = mu_i + delta_ij   (rate ratio)
    continuous  y_ij ~ N(theta_ij, sd_ij^2/n_ij), theta_ij = mu_i + delta_ij  (MD)
    delta_i0 = 0;  delta_i ~ N(X_i beta, tau^2 Sigma)  (random effects,
    Sigma = 1 on the diagonal, 1/2 off it for multi-arm studies)
    or delta_i = X_i beta (fixed effect).
Contrast-level studies in the same network enter with a normal likelihood
on their reported contrasts: y_i ~ N(delta_i, V_i) (TSD 2 "shared parameter"
approach). Priors: mu_i, beta ~ N(0, prior_sd^2) (for continuous data the mu
prior is widened to 10 x the largest |mean|); tau ~ Uniform(0, U) or
Half-Normal(0, s).

Consistency model: X_i maps beta = d (effects of every treatment vs the
reference) to d_arm - d_baseline. UME model: one free parameter per observed
comparison (baseline, arm), so no consistency is assumed; comparing its
fit with the consistency model (DIC, and the deviance of every data point) is
the TSD 4 check for inconsistency.

Sampler (no JAGS / Stan needed; all conditionals are log-concave):
  1. per study, (mu_i, delta_i) jointly by an independence Metropolis-Hastings
     step with a multivariate-t proposal from the Laplace approximation of
     its conditional (Newton to the mode from a fixed start), batched over
     studies with the same number of arms;
  2. contrast-level deltas and beta | delta, tau: exact Gaussian (Gibbs) draws;
  3. a "shift" move beta -> beta + e, delta_i -> delta_i + X_i e (Laplace
     independence proposal for e) that removes the beta-delta correlation when
     tau is small;
  4. tau by slice sampling, plus a joint scale move (tau, delta - X beta) ->
     s (tau, delta - X beta) against the tau-delta "funnel".
Fixed effect: step 1 on mu_i only, beta by a Laplace independence step.
Convergence: split R-hat and ESS per parameter across independent chains.

Deviance (TSD 2): residual deviance of every data point (arm, or contrast
vector of a contrast-level study), posterior mean Dres, pD = sum of leverages
(Dres minus the deviance at the posterior mean of the fitted values), DIC =
Dres + pD, and the leverage plot data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .models import ValidationError
from .nma_bayes import BayesResult, BayesSettings, ParameterDiagnostics, _posterior_fit, _rankings, _summary
from .nma_freq import _choose_baseline, _uninformative_binary, build_study_blocks
from .nma_inconsistency import rebase_block
from .nma_models import NetworkMetaAnalysis, NMAResult, RelativeEffect, is_ratio_measure
from .nma_validation import validate_nma

EXACT_LIKELIHOOD = {
    ("binary", "OR"): "binomial",  # logit link
    ("binary", "RR"): "binomial-log",  # log link: p = exp(eta), eta < 0 (TSD 2 §4.2)
    ("binary", "RD"): "binomial-identity",  # identity link: p = eta in (0, 1)
    ("rate", "RATE_RATIO"): "poisson",
    ("continuous", "MD"): "normal",
    ("continuous", "SMD"): "normal-standardised",  # normal on means / pooled SD (SD treated as known)
}
MODEL_TYPES = ("consistency", "ume", "nodesplit")
BINOMIAL = ("binomial", "binomial-log", "binomial-identity")
LINK = {"binomial": "logit", "binomial-log": "log", "binomial-identity": "identity", "poisson": "log",
        "normal": "identity", "normal-standardised": "identity"}
LIKELIHOOD_LABELS = {"binomial": "binomial (logit link)", "binomial-log": "binomial (log link)",
                     "binomial-identity": "binomial (identity link)", "poisson": "Poisson (log link)",
                     "normal": "normal", "normal-standardised": "normal on standardised means (pooled SD known)"}
_EPS = 1e-9


def exact_likelihood(model: NetworkMetaAnalysis) -> str | None:
    """The exact arm-level likelihood for this outcome / measure, or None:
    'binomial' (logit, OR), 'binomial-log' (RR), 'binomial-identity' (RD),
    'poisson' (rate ratio), 'normal' (MD), 'normal-standardised' (SMD)."""

    return EXACT_LIKELIHOOD.get((model.spec.outcome, model.spec.measure))


@dataclass
class DataPoint:
    study_id: str
    label: str  # treatment of the arm, or "A, B vs P" for a contrast-level study
    n_points: int
    dev_bar: float  # posterior mean residual deviance
    dev_at_mean: float
    leverage: float
    sign: float  # sign of (observed - fitted at the posterior mean)


@dataclass
class BayesNodeSplit:
    """Bayesian node-split (TSD 4 / Dias 2010) for treatment vs versus."""

    treatment: str
    versus: str
    n_direct_studies: int
    direct: RelativeEffect
    indirect: RelativeEffect
    difference: float  # posterior median of direct - indirect (analysis scale)
    difference_lower: float
    difference_upper: float
    p_value: float  # 2 * min(P(diff > 0), P(diff < 0))
    dic: float | None = None


@dataclass
class ExactResult(BayesResult):
    likelihood: str = "binomial"
    model_type: str = "consistency"
    warmup: int = 1000
    data_points: list[DataPoint] = field(default_factory=list)
    # UME: effect of each observed comparison (arm vs baseline), keyed (arm, baseline)
    ume_effects: dict[tuple[str, str], RelativeEffect] = field(default_factory=dict)
    # Node-split (model_type "nodesplit"): direct, indirect and their difference
    split: "BayesNodeSplit | None" = None
    acceptance: dict[str, float] = field(default_factory=dict)
    # Per-chain draws for convergence plots: (chains, kept draws, parameters) and (chains, kept draws)
    parameter_names: list[str] = field(default_factory=list)
    chain_draws: np.ndarray | None = None
    chain_tau: np.ndarray | None = None
    # Treatment classes ("common" / "exchangeable"): members of each class used in the network, the class
    # effect versus the reference, sigma_class (exchangeable) and SUCRA across classes
    class_model: str = "none"
    class_members: dict[str, list[str]] = field(default_factory=dict)
    class_effects: dict[str, RelativeEffect] = field(default_factory=dict)
    class_sd: ParameterDiagnostics | None = None
    class_sucra: dict[str, float] = field(default_factory=dict)


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------


class _ArmGroup:
    """Arm-level studies with the same number of arms k (baseline first)."""

    def __init__(self, k: int):
        self.k = k
        self.ids: list[str] = []
        self.arms: list[list[str]] = []
        self.a: list[list[float]] = []  # events / y
        self.b: list[list[float]] = []  # n / exposure / variance
        self.rows: list[np.ndarray] = []  # (k-1, p) design

    def freeze(self):
        self.a = np.array(self.a, dtype=float)
        self.b = np.array(self.b, dtype=float)
        self.X = np.array(self.rows, dtype=float)  # (S, k-1, p)
        self.S = len(self.ids)


class _ContrastGroup:
    def __init__(self, m: int):
        self.m = m
        self.ids: list[str] = []
        self.labels: list[str] = []
        self.y: list[np.ndarray] = []
        self.V: list[np.ndarray] = []
        self.rows: list[np.ndarray] = []

    def freeze(self):
        self.y = np.array(self.y, dtype=float)
        self.V = np.array(self.V, dtype=float)
        self.Vinv = np.linalg.inv(self.V)
        self.X = np.array(self.rows, dtype=float)
        self.S = len(self.ids)


def _sigma(m: int) -> np.ndarray:
    return 0.5 * (np.eye(m) + np.ones((m, m)))


# --------------------------------------------------------------------------
# Arm log-likelihoods: value, gradient and Hessian diagonal w.r.t. eta
# --------------------------------------------------------------------------


def _arm_loglik(kind: str, eta: np.ndarray, a: np.ndarray, b: np.ndarray):
    """Log-likelihood, gradient and Hessian diagonal w.r.t. the linear
    predictor eta of each arm. The log and identity links constrain eta
    (p < 1, resp. 0 < p < 1): outside the range the value is -inf while the
    gradient / Hessian are those of the nearest feasible point, so Newton
    steps still point back into the feasible region."""

    if kind == "binomial":
        p = 1.0 / (1.0 + np.exp(-eta))
        ll = a * eta - b * np.logaddexp(0.0, eta)
        return ll, a - b * p, -b * p * (1.0 - p)
    if kind == "binomial-log":
        feasible = eta < 0.0
        e = np.minimum(eta, -_EPS)
        p = np.exp(e)
        q = -np.expm1(e)  # 1 - p, accurate near 0
        ll = np.where(feasible, a * e + (b - a) * np.log(q), -np.inf)
        return ll, a - (b - a) * p / q, -(b - a) * p / q ** 2
    if kind == "binomial-identity":
        feasible = (eta > 0.0) & (eta < 1.0)
        p = np.clip(eta, _EPS, 1.0 - _EPS)
        ll = np.where(feasible, a * np.log(p) + (b - a) * np.log1p(-p), -np.inf)
        return ll, a / p - (b - a) / (1.0 - p), -a / p ** 2 - (b - a) / (1.0 - p) ** 2
    if kind == "poisson":
        mu = b * np.exp(np.minimum(eta, 50.0))
        return a * eta - mu, a - mu, -mu
    return -0.5 * (a - eta) ** 2 / b, (a - eta) / b, -1.0 / b


def _arm_deviance(kind: str, fitted: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Residual deviance per arm; `fitted` = p (binomial), rate (Poisson) or mean."""

    def xlogy(x, y):
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(x > 0, x * np.log(np.where(x > 0, x, 1.0) / y), 0.0)

    if kind in BINOMIAL:
        rhat = np.clip(b * fitted, 1e-300, None)
        return 2.0 * (xlogy(a, rhat) + xlogy(b - a, np.clip(b - rhat, 1e-300, None)))
    if kind == "poisson":
        rhat = np.clip(b * fitted, 1e-300, None)
        return 2.0 * ((rhat - a) + xlogy(a, rhat))
    return (a - fitted) ** 2 / b


def _fitted(kind: str, eta: np.ndarray) -> np.ndarray:
    if kind == "binomial":
        return 1.0 / (1.0 + np.exp(-eta))
    if kind == "binomial-log":
        return np.exp(np.minimum(eta, 0.0))
    if kind == "binomial-identity":
        return np.clip(eta, 0.0, 1.0)
    if kind == "poisson":
        return np.exp(np.minimum(eta, 50.0))
    return eta


# --------------------------------------------------------------------------
# Batched Laplace independence Metropolis-Hastings
# --------------------------------------------------------------------------


def _newton(fn, start: np.ndarray, iters: int = 30):
    """Batched Newton ascent (with step halving) to the mode of concave fn.
    fn(x) -> (lp (S,), grad (S,d), hess (S,d,d))."""

    x = start.copy()
    lp, g, H = fn(x)
    for _ in range(iters):
        step = np.linalg.solve(-H, g[..., None])[..., 0]
        step = np.clip(step, -5.0, 5.0)
        t = np.ones(len(x))
        for _ in range(12):
            trial = x + t[:, None] * step
            lp_t = fn(trial)[0]
            worse = ~(lp_t >= lp - 1e-10)
            if not worse.any():
                break
            t = np.where(worse, t * 0.5, t)
        x = x + t[:, None] * step
        lp_new, g, H = fn(x)
        done = bool(np.all(np.abs(t[:, None] * step) < 1e-6))
        lp = lp_new
        if done:
            break
    return x, lp, H


def _laplace_mh(fn, current: np.ndarray, start: np.ndarray, rng: np.random.Generator, nu: float = 6.0):
    """One independence MH step per row; returns (new x, accepted mask)."""

    mode, _, H = _newton(fn, start)
    precision = -H
    d = mode.shape[1]
    try:
        L = np.linalg.cholesky(precision)
    except np.linalg.LinAlgError:
        return current, np.zeros(len(current), dtype=bool)
    z = rng.standard_normal(mode.shape)
    w = rng.chisquare(nu, size=len(mode)) / nu
    u_new = z / np.sqrt(w)[:, None]
    proposal = mode + np.linalg.solve(np.swapaxes(L, -1, -2), u_new[..., None])[..., 0]
    u_cur = np.einsum("sji,sj->si", L, current - mode)  # L' (x - mode)

    def log_q(u):
        return -0.5 * (nu + d) * np.log1p(np.sum(u * u, axis=1) / nu)

    log_alpha = fn(proposal)[0] - fn(current)[0] + log_q(u_cur) - log_q(u_new)
    accept = np.log(rng.random(len(current))) < np.nan_to_num(log_alpha, nan=-np.inf)
    return np.where(accept[:, None], proposal, current), accept


def _slice(logf, x0: float, rng: np.random.Generator, width: float = 0.3, lower: float = 0.0,
           upper: float = math.inf, max_steps: int = 50) -> float:
    """Univariate slice sampler with stepping out (Neal 2003)."""

    level = logf(x0) + math.log(rng.random())
    left = x0 - width * rng.random()
    right = left + width
    steps = 0
    while left > lower and logf(left) > level and steps < max_steps:
        left -= width
        steps += 1
    steps = 0
    while right < upper and logf(right) > level and steps < max_steps:
        right += width
        steps += 1
    left, right = max(left, lower), min(right, upper)
    for _ in range(200):
        x1 = left + (right - left) * rng.random()
        if logf(x1) > level:
            return x1
        if x1 < x0:
            left = x1
        else:
            right = x1
    return x0


def _class_structure(model: NetworkMetaAnalysis, treatments: list[str]) -> dict[str, list[str]]:
    """Treatment classes with at least two of the treatments used in the network
    (members in network order)."""

    label = {t.name: t.treatment_class for t in model.treatments if t.treatment_class}
    groups: dict[str, list[str]] = {}
    for name in treatments:
        if label.get(name):
            groups.setdefault(label[name], []).append(name)
    return {cls: members for cls, members in groups.items() if len(members) >= 2}


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def run_nma_bayesian_exact(model: NetworkMetaAnalysis, settings: BayesSettings | None = None,
                           model_type: str = "consistency", warmup: int | None = None,
                           split: tuple[str, str] | None = None) -> ExactResult:
    """model_type: "consistency" (NMA), "ume" (unrelated mean effects, TSD 4)
    or "nodesplit" with `split` = (treatment, versus): studies comparing the
    two get their own direct parameter (they are re-baselined to `versus`),
    the rest of the network gives the indirect estimate through the
    consistency parameters d."""
    settings = settings or BayesSettings()
    warmup = settings.draws_per_chain if warmup is None else int(warmup)
    kind = exact_likelihood(model)
    result = ExactResult(reference=model.reference, measure=model.spec.measure, settings=settings,
                         likelihood=kind or "", model_type=model_type, warmup=warmup)
    result.errors = validate_nma(model)
    if kind is None:
        result.errors.append(ValidationError(
            code="no_exact_likelihood",
            message="The exact likelihood is available for binary data (OR, RR, RD), counts with rate ratio and "
                    "continuous data (MD, SMD); use the normal approximation for this outcome / measure."))
    if model_type not in MODEL_TYPES:
        result.errors.append(ValidationError(code="unknown_model_type", message=f"Unknown model '{model_type}'"))
    if model_type == "nodesplit" and (not split or len(split) != 2):
        result.errors.append(ValidationError(code="missing_split", message="Give the comparison to split"))
    if settings.chains < 1 or settings.draws_per_chain < 10:
        result.errors.append(ValidationError(code="too_few_draws", message="Use at least 1 chain and 10 draws per chain"))
    if result.errors:
        return result

    random_effects = settings.effects == "random"
    order = {name: i for i, name in enumerate(model.treatment_names)}
    reference = model.reference
    split_pair = tuple(split) if model_type == "nodesplit" else None

    def choose_baseline(names: list[str]) -> str:
        if split_pair and split_pair[0] in names and split_pair[1] in names:
            return split_pair[1]
        return _choose_baseline(names, order, reference)

    # ---- which studies, and the parameters -----------------------------
    arm_studies, contrast_studies, pairs = [], [], []
    for study in model.studies:
        if study.is_contrast_level:
            contrast_studies.append(study)
            continue
        if kind in ("binomial", "binomial-log") and _uninformative_binary(study):
            result.warnings.append(f"{study.id}: excluded - no events (or all events) in every arm: "
                                   "no information on the " + ("odds ratio" if kind == "binomial" else "risk ratio"))
            continue
        arm_studies.append(study)
    blocks = {block.study_id: block for block in build_study_blocks(model, NMAResult())}
    used: list[str] = []
    for study in arm_studies:
        names = [arm.treatment for arm in study.arms]
        baseline = choose_baseline(names)
        for name in names:
            if name not in used:
                used.append(name)
            if name != baseline and (name, baseline) not in pairs:
                pairs.append((name, baseline))
    for study in contrast_studies:
        block = blocks.get(study.id)
        if block is None:
            continue
        for name in (block.baseline, *block.treatments):
            if name not in used:
                used.append(name)
        for name in block.treatments:
            if (name, block.baseline) not in pairs:
                pairs.append((name, block.baseline))
    treatments = [name for name in model.treatment_names if name in used]
    result.treatments = treatments
    if reference not in used:
        result.errors.append(ValidationError(code="reference_excluded",
                                             message=f"The reference '{reference}' has no usable data"))
        return result
    if model_type in ("consistency", "nodesplit"):
        parameters = [name for name in treatments if name != reference]
        if model_type == "nodesplit":
            if not all(name in used for name in split_pair):
                result.errors.append(ValidationError(code="split_not_in_network",
                                                     message="The comparison to split is not in the network"))
                return result
            parameters = parameters + [f"direct[{split_pair[0]} vs {split_pair[1]}]"]
        column = {name: i for i, name in enumerate(parameters)}
        direct_column = len(parameters) - 1 if model_type == "nodesplit" else None

        def design_row(arm: str, base: str) -> np.ndarray:
            row = np.zeros(len(parameters))
            if direct_column is not None and {arm, base} == set(split_pair):
                row[direct_column] = 1.0 if arm == split_pair[0] else -1.0
                return row
            if column.get(arm) is not None:
                row[column[arm]] += 1.0
            if column.get(base) is not None:
                row[column[base]] -= 1.0
            return row
    else:
        pairs.sort(key=lambda pair: (order[pair[1]], order[pair[0]]))
        parameters = [f"{a} vs {b}" for a, b in pairs]
        column = {pair: i for i, pair in enumerate(pairs)}

        def design_row(arm: str, base: str) -> np.ndarray:
            row = np.zeros(len(parameters))
            row[column[(arm, base)]] = 1.0
            return row
    report_parameters = list(parameters)  # treatment effects that are reported
    class_mode, class_groups, hier = "none", {}, None
    expand_index: list[int] = []
    sampled_names: list[str] = []
    if settings.class_model in ("common", "exchangeable"):
        if model_type != "consistency":
            result.warnings.append("Class effects apply to the consistency model only and are ignored here.")
        else:
            class_groups = _class_structure(model, treatments)
            if settings.class_model == "exchangeable":  # needs two non-reference members to share information
                class_groups = {c: m for c, m in class_groups.items()
                                if sum(1 for name in m if name in report_parameters) >= 2}
            if class_groups:
                class_mode = settings.class_model
            else:
                result.warnings.append("No treatment class with enough treatments in this network: "
                                       "the model was run without class effects.")
    result.class_model = class_mode
    result.class_members = {c: list(m) for c, m in class_groups.items()} if class_mode != "none" else {}
    ref_class = next((c for c, m in class_groups.items() if reference in m), None) if class_mode == "common" else None
    if class_mode == "common":
        cls_of = {name: c for c, m in class_groups.items() for name in m}
        column = {}
        for name in report_parameters:
            c = cls_of.get(name)
            if c is not None and c == ref_class:
                column[name] = None  # same class as the reference: constrained equal to it
                continue
            key = f"class[{c}]" if c is not None else name
            if key not in sampled_names:
                sampled_names.append(key)
            column[name] = sampled_names.index(key)
        if ref_class is not None:
            result.warnings.append(f"Common class effect: the other treatments of {ref_class} (the class of the "
                                   f"reference {reference}) are constrained equal to the reference.")
        parameters = list(sampled_names)
        expand_index = [-1 if column[name] is None else column[name] for name in report_parameters]
    p = len(parameters)
    if class_mode == "exchangeable":
        member_of = {name: c for c, m in class_groups.items() for name in m if name in parameters}
        class_names = list(class_groups)
        Mc = np.zeros((p, len(class_names)))
        for name, c in member_of.items():
            Mc[parameters.index(name), class_names.index(c)] = 1.0
        member_mask = Mc.sum(axis=1) > 0
        class_counts = Mc.sum(axis=0)
        n_member = int(member_mask.sum())

    # ---- data groups ----------------------------------------------------
    arm_groups: dict[int, _ArmGroup] = {}
    max_abs_mean = 0.0
    for study in arm_studies:
        names = [arm.treatment for arm in study.arms]
        baseline = choose_baseline(names)
        arms = sorted(study.arms, key=lambda arm: (arm.treatment != baseline, names.index(arm.treatment)))
        group = arm_groups.setdefault(len(arms), _ArmGroup(len(arms)))
        group.ids.append(study.id)
        group.arms.append([arm.treatment for arm in arms])
        if kind in BINOMIAL:
            group.a.append([float(arm.events) for arm in arms])
            group.b.append([float(arm.n) for arm in arms])
        elif kind == "poisson":
            group.a.append([float(arm.events) for arm in arms])
            group.b.append([float(arm.exposure) for arm in arms])
        elif kind == "normal-standardised":
            # Means and variances divided by the study's pooled SD (df = N - k)
            # times Hedges' small-sample factor J, so delta is Hedges' g; the
            # pooled SD is treated as known (as in the frequentist SMD).
            total_n = sum(float(arm.n) for arm in arms)
            df = total_n - len(arms)
            pooled = math.sqrt(sum((float(arm.n) - 1.0) * float(arm.sd) ** 2 for arm in arms) / df)
            j = 1.0 - 3.0 / (4.0 * df - 1.0) if df > 1 else 1.0
            group.a.append([j * float(arm.mean) / pooled for arm in arms])
            group.b.append([j * j * float(arm.sd) ** 2 / (pooled ** 2 * float(arm.n)) for arm in arms])
            max_abs_mean = max(max_abs_mean, *(abs(j * float(arm.mean) / pooled) for arm in arms))
        else:
            group.a.append([float(arm.mean) for arm in arms])
            group.b.append([float(arm.sd) ** 2 / float(arm.n) for arm in arms])
            max_abs_mean = max(max_abs_mean, *(abs(float(arm.mean)) for arm in arms))
        group.rows.append(np.array([design_row(arm.treatment, baseline) for arm in arms[1:]]).reshape(len(arms) - 1, p))
    contrast_groups: dict[int, _ContrastGroup] = {}
    for study in contrast_studies:
        block = blocks.get(study.id)
        if block is None:
            continue
        if split_pair and split_pair[1] in block.treatments and split_pair[0] in (block.baseline, *block.treatments):
            block = rebase_block(block, split_pair[1])
        m = len(block.treatments)
        group = contrast_groups.setdefault(m, _ContrastGroup(m))
        group.ids.append(study.id)
        group.labels.append(", ".join(block.treatments) + " vs " + block.baseline)
        group.y.append(np.array(block.estimates))
        V = np.full((m, m), block.baseline_variance) if m > 1 else np.zeros((1, 1))
        V[np.diag_indices(m)] = block.variances
        group.V.append(V)
        group.rows.append(np.array([design_row(name, block.baseline) for name in block.treatments]).reshape(m, p))
    for group in (*arm_groups.values(), *contrast_groups.values()):
        group.freeze()
    arm_list = list(arm_groups.values())
    con_list = list(contrast_groups.values())
    X_all = np.concatenate([g.X.reshape(-1, p) for g in (*arm_list, *con_list)], axis=0)
    if np.linalg.matrix_rank(X_all) < p:
        result.errors.append(ValidationError(code="not_estimable",
                                             message="Some treatment effects cannot be estimated (disconnected network)"))
        return result
    m_total = sum(g.S * (g.k - 1) for g in arm_list) + sum(g.S * g.m for g in con_list)

    prior_prec = 1.0 / settings.prior_sd ** 2
    mu_sd = max(settings.prior_sd, 10.0 * max_abs_mean) if kind.startswith("normal") else settings.prior_sd
    mu_prec = 1.0 / mu_sd ** 2
    tau_upper = settings.tau_max if settings.tau_prior == "uniform" else math.inf

    def log_tau_prior(tau: float) -> float:
        if tau <= 0 or tau > tau_upper:
            return -math.inf
        if settings.tau_prior == "half-normal":
            return -0.5 * (tau / settings.tau_scale) ** 2
        if settings.tau_prior == "lognormal":
            z = (2.0 * math.log(tau) - settings.tau_mu) / settings.tau_sigma
            if settings.tau_df is None:
                return -0.5 * z * z
            return -0.5 * (settings.tau_df + 1.0) * math.log1p(z * z / settings.tau_df)
        return 0.0

    # Data-based start for mu (fixed, so the Laplace proposals depend only on
    # the conditioning parameters).
    for g in arm_list:
        base_a, base_b = g.a[:, 0], g.b[:, 0]
        if kind == "binomial":
            g.mu_start = np.log((base_a + 0.5) / (base_b - base_a + 0.5))
        elif kind == "binomial-log":
            g.mu_start = np.log((base_a + 0.5) / (base_b + 1.0))
        elif kind == "binomial-identity":
            g.mu_start = (base_a + 0.5) / (base_b + 1.0)
        elif kind == "poisson":
            g.mu_start = np.log((base_a + 0.5) / base_b)
        else:
            g.mu_start = base_a.copy()
        g.sigma_inv = np.linalg.inv(_sigma(g.k - 1))
    for g in con_list:
        g.sigma_inv = np.linalg.inv(_sigma(g.m))

    rng = np.random.default_rng(settings.seed)
    n_chains, n_keep = settings.chains, settings.draws_per_chain
    C = n_chains
    beta_draws = np.empty((n_chains, n_keep, p))
    tau_draws = np.zeros((n_chains, n_keep))
    # deviance accumulators over kept draws of every chain
    dev_sum = {id(g): np.zeros((g.S, g.k)) for g in arm_list}
    fit_sum = {id(g): np.zeros((g.S, g.k)) for g in arm_list}
    cdev_sum = {id(g): np.zeros(g.S) for g in con_list}
    cfit_sum = {id(g): np.zeros((g.S, g.m)) for g in con_list}
    accepted = {"study": [0, 0], "shift": [0, 0], "scale": [0, 0], "beta": [0, 0]}
    # sum_i X_i' Sigma^-1 X_i (beta | delta precision = prior + G / tau^2)
    G = np.zeros((p, p))
    for g in (*arm_list, *con_list):
        G += np.einsum("sjp,jk,skq->pq", g.X, g.sigma_inv, g.X)

    # Chains run side by side (leading axis c), so every numpy call serves all of them.
    constrained = kind in ("binomial-log", "binomial-identity")
    beta = np.zeros((C, p)) if constrained else rng.normal(0.0, 0.5, (C, p))
    tau_start_hi = min(1.0, tau_upper)
    if settings.tau_prior == "lognormal":
        tau_start_hi = min(tau_start_hi, math.exp(0.5 * (settings.tau_mu + settings.tau_sigma)))
    tau = rng.uniform(0.05 * min(1.0, tau_start_hi / 0.5), max(tau_start_hi, 0.06), C) if random_effects else np.zeros(C)
    if constrained:
        mu = {id(g): np.repeat(g.mu_start[None], C, axis=0) for g in arm_list}
    else:
        mu = {id(g): g.mu_start[None] + rng.normal(0.0, 0.3, (C, g.S)) for g in arm_list}
    delta = {id(g): np.einsum("sjp,cp->csj", g.X, beta) for g in arm_list}
    cdelta = {id(g): np.einsum("sjp,cp->csj", g.X, beta) for g in con_list}

    def arm_eta(g, mu_g, dl):
        return mu_g[..., None] + np.concatenate([np.zeros(dl.shape[:-1] + (1,)), dl], axis=-1)

    # Exchangeable class effects: d_k ~ N(gamma_class, sigma_class^2) for members of a class (gamma_class ~
    # N(0, prior_sd^2), sigma_class ~ Half-Normal(0, class_sd_scale)); other d_k ~ N(0, prior_sd^2).
    if class_mode == "exchangeable":
        hier = {"gamma": np.zeros((C, len(class_names))), "sigma": np.full(C, 0.5)}
        gamma_draws = np.zeros((C, n_keep, len(class_names)))
        csigma_draws = np.zeros((C, n_keep))

        def refresh_hier() -> None:
            hier["mean"] = hier["gamma"] @ Mc.T
            hier["prec"] = np.where(member_mask[None, :], 1.0 / hier["sigma"][:, None] ** 2, prior_prec)

        refresh_hier()

    def beta_prior(bv):
        """log prior of beta (rows = chains), its gradient and Hessian."""

        if hier is None:
            return (-0.5 * prior_prec * np.sum(bv ** 2, axis=1), -prior_prec * bv,
                    np.repeat((-prior_prec * np.eye(p))[None], C, axis=0))
        resid, lam = bv - hier["mean"], hier["prec"]
        H = np.zeros((C, p, p))
        H[:, np.arange(p), np.arange(p)] = -lam
        return -0.5 * np.sum(lam * resid * resid, axis=1), -lam * resid, H

    def update_hier() -> None:
        if hier is None:
            return
        sig2 = hier["sigma"] ** 2
        precision = 1.0 / settings.prior_sd ** 2 + class_counts[None, :] / sig2[:, None]
        centre = (beta @ Mc) / sig2[:, None] / precision
        hier["gamma"] = centre + rng.standard_normal(centre.shape) / np.sqrt(precision)
        resid = (beta - hier["gamma"] @ Mc.T)[:, member_mask]
        q = np.sum(resid * resid, axis=1)
        for c in range(C):
            def log_sigma(sd, qc=float(q[c])):
                if sd <= 0:
                    return -math.inf
                return -n_member * math.log(sd) - 0.5 * qc / (sd * sd) - 0.5 * (sd / settings.class_sd_scale) ** 2

            hier["sigma"][c] = _slice(log_sigma, float(hier["sigma"][c]), rng, width=0.3, lower=0.0)
        refresh_hier()

    def feasible_mu_start(g, mean_rows: np.ndarray) -> np.ndarray:
        """Newton start for mu given the delta start `mean_rows` (rows =
        chain x study): the data-based start, pulled inside the feasible
        range of the log / identity link if needed."""

        start = np.tile(g.mu_start, C)
        if kind == "binomial-log":
            return np.minimum(start, -np.maximum(mean_rows.max(axis=1), 0.0) - 0.05)
        if kind == "binomial-identity":
            low = np.maximum(-mean_rows.min(axis=1), 0.0)
            high = np.minimum(1.0 - mean_rows.max(axis=1), 1.0)
            inside = np.clip(start, low + 0.02, high - 0.02)
            return np.where(high - low > 0.05, inside, 0.5 * (low + high))
        return start

    for it in range(warmup + n_keep):
        # 1. study blocks (rows = chain x study) ---------------------------
        for g in arm_list:
            key = id(g)
            mean = np.einsum("sjp,cp->csj", g.X, beta)
            a = np.broadcast_to(g.a, (C, g.S, g.k)).reshape(C * g.S, g.k)
            b = np.broadcast_to(g.b, (C, g.S, g.k)).reshape(C * g.S, g.k)
            if random_effects:
                P = g.sigma_inv[None] / np.maximum(tau, 1e-6)[:, None, None] ** 2  # (C, k-1, k-1)
                P_rows = np.repeat(P, g.S, axis=0)
                mean_rows = mean.reshape(C * g.S, g.k - 1)

                def fn(x, g=g, mean_rows=mean_rows, P_rows=P_rows, a=a, b=b):
                    eta = x[:, :1] + np.concatenate([np.zeros((len(x), 1)), x[:, 1:]], axis=1)
                    ll, ge, he = _arm_loglik(kind, eta, a, b)
                    r = x[:, 1:] - mean_rows
                    Pr = np.einsum("njk,nk->nj", P_rows, r)
                    lp = ll.sum(axis=1) - 0.5 * mu_prec * x[:, 0] ** 2 - 0.5 * np.einsum("nj,nj->n", r, Pr)
                    grad = np.concatenate([(ge.sum(axis=1) - mu_prec * x[:, 0])[:, None], ge[:, 1:] - Pr], axis=1)
                    H = np.zeros((len(x), g.k, g.k))
                    H[:, 0, 0] = he.sum(axis=1) - mu_prec
                    H[:, 0, 1:] = he[:, 1:]
                    H[:, 1:, 0] = he[:, 1:]
                    H[:, 1:, 1:] = -P_rows
                    idx = np.arange(1, g.k)
                    H[:, idx, idx] += he[:, 1:]
                    return lp, grad, H

                current = np.concatenate([mu[key].reshape(-1, 1), delta[key].reshape(C * g.S, g.k - 1)], axis=1)
                start = np.concatenate([feasible_mu_start(g, mean_rows)[:, None], mean_rows], axis=1)
                new, acc = _laplace_mh(fn, current, start, rng)
                mu[key] = new[:, 0].reshape(C, g.S)
                delta[key] = new[:, 1:].reshape(C, g.S, g.k - 1)
            else:
                delta[key] = mean
                eta_offset = np.concatenate([np.zeros((C, g.S, 1)), mean], axis=2).reshape(C * g.S, g.k)

                def fn(x, eta_offset=eta_offset, a=a, b=b):
                    ll, ge, he = _arm_loglik(kind, x[:, :1] + eta_offset, a, b)
                    lp = ll.sum(axis=1) - 0.5 * mu_prec * x[:, 0] ** 2
                    return lp, (ge.sum(axis=1) - mu_prec * x[:, 0])[:, None], (he.sum(axis=1) - mu_prec)[:, None, None]

                new, acc = _laplace_mh(fn, mu[key].reshape(-1, 1),
                                       feasible_mu_start(g, mean.reshape(C * g.S, g.k - 1))[:, None], rng)
                mu[key] = new[:, 0].reshape(C, g.S)
            accepted["study"][0] += int(acc.sum())
            accepted["study"][1] += acc.size
        for g in con_list:
            key = id(g)
            mean = np.einsum("sjp,cp->csj", g.X, beta)
            if random_effects:
                P = g.sigma_inv[None] / np.maximum(tau, 1e-6)[:, None, None] ** 2
                precision = g.Vinv[None] + P[:, None]  # (C, S, m, m)
                cov = np.linalg.inv(precision)
                rhs = np.einsum("sij,sj->si", g.Vinv, g.y)[None] + np.einsum("cjk,csk->csj", P, mean)
                centre = np.einsum("csij,csj->csi", cov, rhs)
                chol = np.linalg.cholesky(cov)
                cdelta[key] = centre + np.einsum("csij,csj->csi", chol, rng.standard_normal((C, g.S, g.m)))
            else:
                cdelta[key] = mean

        if random_effects:
            inv_t2 = 1.0 / np.maximum(tau, 1e-6) ** 2
            # 2. beta | delta, tau (Gibbs) -------------------------------------
            if hier is None:
                A = np.eye(p)[None] * prior_prec + G[None] * inv_t2[:, None, None]
                bvec = np.zeros((C, p))
            else:
                A = G[None] * inv_t2[:, None, None]
                A[:, np.arange(p), np.arange(p)] += hier["prec"]
                bvec = hier["prec"] * hier["mean"]
            for g, dlt in [(g, delta[id(g)]) for g in arm_list] + [(g, cdelta[id(g)]) for g in con_list]:
                bvec += np.einsum("sjp,jk,csk->cp", g.X, g.sigma_inv, dlt) * inv_t2[:, None]
            LA = np.linalg.cholesky(A)
            beta = (np.linalg.solve(A, bvec[..., None])[..., 0]
                    + np.linalg.solve(np.swapaxes(LA, -1, -2), rng.standard_normal((C, p))[..., None])[..., 0])

            # 3. shift move (rows = chains) --------------------------------------
            def shift_fn(e):
                lp, grad, H = beta_prior(beta + e)
                for g in arm_list:
                    shifted = delta[id(g)] + np.einsum("sjp,cp->csj", g.X, e)
                    ll, ge, he = _arm_loglik(kind, arm_eta(g, mu[id(g)], shifted), g.a[None], g.b[None])
                    lp = lp + ll.sum(axis=(1, 2))
                    grad = grad + np.einsum("csj,sjp->cp", ge[:, :, 1:], g.X)
                    H = H + np.einsum("sjp,csj,sjq->cpq", g.X, he[:, :, 1:], g.X)
                for g in con_list:
                    resid = g.y[None] - cdelta[id(g)] - np.einsum("sjp,cp->csj", g.X, e)
                    wres = np.einsum("sij,csj->csi", g.Vinv, resid)
                    lp = lp - 0.5 * np.einsum("csi,csi->c", resid, wres)
                    grad = grad + np.einsum("csi,sip->cp", wres, g.X)
                    H = H - np.einsum("sip,sij,sjq->pq", g.X, g.Vinv, g.X)[None]
                return lp, grad, H

            e_new, acc = _laplace_mh(shift_fn, np.zeros((C, p)), np.zeros((C, p)), rng)
            accepted["shift"][0] += int(acc.sum())
            accepted["shift"][1] += C
            beta = beta + e_new
            for g in arm_list:
                delta[id(g)] = delta[id(g)] + np.einsum("sjp,cp->csj", g.X, e_new)
            for g in con_list:
                cdelta[id(g)] = cdelta[id(g)] + np.einsum("sjp,cp->csj", g.X, e_new)

            # 4. tau: slice per chain, then a joint scale move -------------------
            # (repeated: both steps are cheap and tau mixes slowest)
            for _repeat in range(3):
                q = np.zeros(C)
                for g, dlt in [(g, delta[id(g)]) for g in arm_list] + [(g, cdelta[id(g)]) for g in con_list]:
                    r = dlt - np.einsum("sjp,cp->csj", g.X, beta)
                    q += np.einsum("csj,jk,csk->c", r, g.sigma_inv, r)
                for c in range(C):
                    qc = float(q[c])

                    def log_tau(t, qc=qc):
                        prior = log_tau_prior(t)
                        return -math.inf if prior == -math.inf else prior - m_total * math.log(t) - 0.5 * qc / (t * t)

                    tau[c] = _slice(log_tau, float(tau[c]), rng, width=0.3, lower=0.0, upper=tau_upper)

                def data_loglik(scale):
                    total = np.zeros(C)
                    for g in arm_list:
                        mean = np.einsum("sjp,cp->csj", g.X, beta)
                        dl = mean + scale[:, None, None] * (delta[id(g)] - mean)
                        total += _arm_loglik(kind, arm_eta(g, mu[id(g)], dl), g.a[None], g.b[None])[0].sum(axis=(1, 2))
                    for g in con_list:
                        mean = np.einsum("sjp,cp->csj", g.X, beta)
                        resid = g.y[None] - (mean + scale[:, None, None] * (cdelta[id(g)] - mean))
                        total -= 0.5 * np.einsum("csi,sij,csj->c", resid, g.Vinv, resid)
                    return total

                s = np.exp(0.6 * rng.standard_normal(C))
                prior_old = np.array([log_tau_prior(float(t)) for t in tau])
                prior_new = np.array([log_tau_prior(float(t)) for t in tau * s])
                log_alpha = data_loglik(s) - data_loglik(np.ones(C)) + prior_new - prior_old + np.log(s)
                accept = np.isfinite(prior_new) & (np.log(rng.random(C)) < np.nan_to_num(log_alpha, nan=-np.inf))
                accepted["scale"][0] += int(accept.sum())
                accepted["scale"][1] += C
                factor = np.where(accept, s, 1.0)
                for g in arm_list:
                    mean = np.einsum("sjp,cp->csj", g.X, beta)
                    delta[id(g)] = mean + factor[:, None, None] * (delta[id(g)] - mean)
                for g in con_list:
                    mean = np.einsum("sjp,cp->csj", g.X, beta)
                    cdelta[id(g)] = mean + factor[:, None, None] * (cdelta[id(g)] - mean)
                tau = tau * factor
        else:
            # Fixed effect: beta by a Laplace independence step (rows = chains).
            def beta_fn(bv):
                lp, grad, H = beta_prior(bv)
                for g in arm_list:
                    mean = np.einsum("sjp,cp->csj", g.X, bv)
                    ll, ge, he = _arm_loglik(kind, arm_eta(g, mu[id(g)], mean), g.a[None], g.b[None])
                    lp = lp + ll.sum(axis=(1, 2))
                    grad = grad + np.einsum("csj,sjp->cp", ge[:, :, 1:], g.X)
                    H = H + np.einsum("sjp,csj,sjq->cpq", g.X, he[:, :, 1:], g.X)
                for g in con_list:
                    resid = g.y[None] - np.einsum("sjp,cp->csj", g.X, bv)
                    wres = np.einsum("sij,csj->csi", g.Vinv, resid)
                    lp = lp - 0.5 * np.einsum("csi,csi->c", resid, wres)
                    grad = grad + np.einsum("csi,sip->cp", wres, g.X)
                    H = H - np.einsum("sip,sij,sjq->pq", g.X, g.Vinv, g.X)[None]
                return lp, grad, H

            beta, acc = _laplace_mh(beta_fn, beta, np.zeros((C, p)), rng)
            accepted["beta"][0] += int(acc.sum())
            accepted["beta"][1] += C
            for g in arm_list:
                delta[id(g)] = np.einsum("sjp,cp->csj", g.X, beta)
            for g in con_list:
                cdelta[id(g)] = np.einsum("sjp,cp->csj", g.X, beta)

        update_hier()
        if it >= warmup:
            keep = it - warmup
            beta_draws[:, keep] = beta
            tau_draws[:, keep] = tau
            if hier is not None:
                gamma_draws[:, keep] = hier["gamma"]
                csigma_draws[:, keep] = hier["sigma"]
            for g in arm_list:
                fitted = _fitted(kind, arm_eta(g, mu[id(g)], delta[id(g)]))  # (C, S, k)
                dev_sum[id(g)] += _arm_deviance(kind, fitted, g.a[None], g.b[None]).sum(axis=0)
                fit_sum[id(g)] += fitted.sum(axis=0)
            for g in con_list:
                resid = g.y[None] - cdelta[id(g)]
                cdev_sum[id(g)] += np.einsum("csi,sij,csj->s", resid, g.Vinv, resid)
                cfit_sum[id(g)] += cdelta[id(g)].sum(axis=0)

    # ---- summaries ----------------------------------------------------------
    total = n_chains * n_keep
    result.acceptance = {name: (a / n if n else float("nan")) for name, (a, n) in accepted.items() if n}
    sampled_beta = beta_draws
    if class_mode == "common":  # report one effect per treatment: members of a class share the class effect
        beta_draws = np.zeros((n_chains, n_keep, len(report_parameters)))
        for j, col in enumerate(expand_index):
            if col >= 0:
                beta_draws[:, :, j] = sampled_beta[:, :, col]
        parameters, p = list(report_parameters), len(report_parameters)
    flat_beta = beta_draws.reshape(total, p)
    flat_tau = tau_draws.reshape(total)
    result.parameter_names = [f"d[{name}]" for name in parameters]
    result.chain_draws = beta_draws
    result.chain_tau = tau_draws if random_effects else None
    level = model.spec.ci_level
    for column, name in enumerate(parameters):
        result.diagnostics.append(_summary(f"d[{name}]", beta_draws[:, :, column], level))
    if random_effects:
        result.tau = _summary("tau", tau_draws, level)
        result.diagnostics.append(result.tau)
    result.tau_draws = flat_tau
    ratio = is_ratio_measure(model.spec.measure)
    if class_mode != "none":
        alpha = (1.0 - level) / 2.0
        transform = math.exp if ratio else (lambda v: v)
        class_draws: dict[str, np.ndarray] = {}
        if class_mode == "common":
            for c in class_groups:
                if c != ref_class:
                    class_draws[c] = sampled_beta[:, :, sampled_names.index(f"class[{c}]")].reshape(total)
        else:
            for i, c in enumerate(class_names):
                class_draws[c] = gamma_draws[:, :, i].reshape(total)
                result.diagnostics.append(_summary(f"class[{c}]", gamma_draws[:, :, i], level))
            result.class_sd = _summary("sigma_class", csigma_draws, level)
            result.diagnostics.append(result.class_sd)
        for c, values in class_draws.items():
            median = float(np.median(values))
            lower, upper = (float(q) for q in np.quantile(values, [alpha, 1.0 - alpha]))
            result.class_effects[c] = RelativeEffect(
                treatment=c, versus=reference, estimate=median, se=float(values.std(ddof=1)), lower=lower, upper=upper,
                natural=transform(median), natural_lower=transform(lower), natural_upper=transform(upper))
        ranked = dict(class_draws)
        if ref_class is not None:
            ranked[ref_class] = np.zeros(total)  # constrained equal to the reference
        if len(ranked) >= 1:
            names_c = list(ranked)
            matrix = np.column_stack([ranked[c] for c in names_c])
            order_c = np.argsort(matrix if model.spec.small_values == "good" else -matrix, axis=1)
            ranks_c = np.empty_like(order_c)
            ranks_c[np.arange(total)[:, None], order_c] = np.arange(len(names_c))[None, :]
            for i, c in enumerate(names_c):
                counts = np.bincount(ranks_c[:, i], minlength=len(names_c)) / total
                result.class_sucra[c] = (float(np.cumsum(counts)[:-1].sum() / (len(names_c) - 1))
                                         if len(names_c) > 1 else 1.0)
    if model_type == "consistency":
        result.draws = flat_beta
        result.fit = _posterior_fit(flat_beta, flat_tau, treatments, reference, parameters,
                                    model.spec.measure, level, random_effects)
        _rankings(result, flat_beta, treatments, reference, parameters, model.spec.small_values)
        result.fit.p_scores = dict(result.sucra)
    elif model_type == "nodesplit":
        transform = math.exp if ratio else (lambda v: v)
        alpha = (1.0 - level) / 2.0
        index = {name: i for i, name in enumerate(parameters)}
        direct = flat_beta[:, -1]
        indirect = np.zeros(total)
        if split_pair[0] in index:
            indirect += flat_beta[:, index[split_pair[0]]]
        if split_pair[1] in index:
            indirect -= flat_beta[:, index[split_pair[1]]]
        difference = direct - indirect

        def effect(values):
            median = float(np.median(values))
            lower, upper = (float(q) for q in np.quantile(values, [alpha, 1.0 - alpha]))
            return RelativeEffect(treatment=split_pair[0], versus=split_pair[1], estimate=median,
                                  se=float(values.std(ddof=1)), lower=lower, upper=upper, natural=transform(median),
                                  natural_lower=transform(lower), natural_upper=transform(upper))

        below = float(np.mean(difference < 0))
        n_direct = sum(1 for g in arm_list for arms in g.arms if set(split_pair) <= set(arms))
        n_direct += sum(1 for g in con_list for label in g.labels
                        if set(split_pair) <= set(label.replace(" vs ", ", ").split(", ")))
        result.split = BayesNodeSplit(
            treatment=split_pair[0], versus=split_pair[1], n_direct_studies=n_direct,
            direct=effect(direct), indirect=effect(indirect), difference=float(np.median(difference)),
            difference_lower=float(np.quantile(difference, alpha)), difference_upper=float(np.quantile(difference, 1 - alpha)),
            p_value=2.0 * min(below, 1.0 - below))
    else:
        transform = math.exp if ratio else (lambda v: v)
        alpha = (1.0 - level) / 2.0
        for column, (arm, base) in enumerate(pairs):
            values = flat_beta[:, column]
            median = float(np.median(values))
            lower, upper = (float(q) for q in np.quantile(values, [alpha, 1.0 - alpha]))
            result.ume_effects[(arm, base)] = RelativeEffect(
                treatment=arm, versus=base, estimate=median, se=float(values.std(ddof=1)), lower=lower, upper=upper,
                natural=transform(median), natural_lower=transform(lower), natural_upper=transform(upper))

    dres = dplug = 0.0
    n_points = 0
    for g in arm_list:
        dev_bar = dev_sum[id(g)] / total
        dev_hat = _arm_deviance(kind, fit_sum[id(g)] / total, g.a, g.b)
        observed = g.a if kind != "poisson" else g.a / g.b
        fitted_mean = fit_sum[id(g)] / total
        sign = np.sign(observed - fitted_mean * (g.b if kind in BINOMIAL else 1.0))
        for s_index, study_id in enumerate(g.ids):
            for j, name in enumerate(g.arms[s_index]):
                result.data_points.append(DataPoint(study_id, name, 1, float(dev_bar[s_index, j]),
                                                    float(dev_hat[s_index, j]),
                                                    float(dev_bar[s_index, j] - dev_hat[s_index, j]),
                                                    float(sign[s_index, j])))
        dres += float(dev_bar.sum())
        dplug += float(dev_hat.sum())
        n_points += g.S * g.k
    for g in con_list:
        dev_bar = cdev_sum[id(g)] / total
        theta = cfit_sum[id(g)] / total
        resid = g.y - theta
        dev_hat = np.einsum("si,sij,sj->s", resid, g.Vinv, resid)
        for s_index, study_id in enumerate(g.ids):
            result.data_points.append(DataPoint(study_id, g.labels[s_index], g.m, float(dev_bar[s_index]),
                                                float(dev_hat[s_index]), float(dev_bar[s_index] - dev_hat[s_index]),
                                                float(np.sign(resid[s_index].sum()))))
        dres += float(dev_bar.sum())
        dplug += float(dev_hat.sum())
        n_points += g.S * g.m
    result.residual_deviance = dres
    result.p_d = dres - dplug
    result.dic = dres + result.p_d
    result.n_data_points = n_points
    if result.split is not None:
        result.split.dic = result.dic
    worst = [name for name, value in result.acceptance.items() if value < 0.2]
    if worst:
        result.warnings.append("Low acceptance in " + ", ".join(worst) + " - check R-hat / ESS; consider more draws.")
    return result
