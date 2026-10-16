"""Probabilistic sensitivity analysis (PSA): Monte Carlo simulation and its
standard HEOR outputs -- CE plane, CEAC, CEAF, summary statistics,
convergence, EVPI and EVPPI.

`run_psa` resamples every configured PSA parameter each iteration and
re-evaluates every strategy through the same `evaluate_markov_scenario` /
`evaluate_tree_scenario` used by the deterministic sensitivity-analysis
tools, so the exact same document semantics (overrides, distributions,
discounting) apply. Everything downstream of `run_psa` is pure
post-processing of the resulting cost/effect sample: no further model
evaluation is needed, so CEAC/CEAF/EVPI can be recomputed cheaply across a
whole willingness-to-pay grid.
"""

from __future__ import annotations

import copy
import dataclasses
import inspect
import os
import re
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from itertools import combinations_with_replacement
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .sensitivity import SensitivityResult

PSA_DISTRIBUTION_TYPES = ["Beta", "Gamma", "Normal", "Lognormal", "Uniform", "Triangular"]


class PSAInputError(ValueError):
    """Raised when a PSA parameter row can't be turned into a sampler, or
    when `run_psa` is called with an invalid setup. Raised before any
    Monte Carlo iteration runs, so setup mistakes fail fast."""


def _as_float(values: Mapping[str, Any], key: str, label: str) -> float:
    raw = values.get(key)
    try:
        return float(raw)
    except (TypeError, ValueError):
        raise PSAInputError(f"{label} PSA input: enter a numeric value for '{key}'") from None


def _beta_from_mean_se(mean: float, se: float) -> tuple[float, float]:
    if not (0.0 < mean < 1.0):
        raise PSAInputError("Beta PSA input needs a mean strictly between 0 and 1")
    if se <= 0:
        raise PSAInputError("Beta PSA input needs a positive SE")
    variance = se * se
    max_variance = mean * (1.0 - mean)
    if not (variance < max_variance):
        raise PSAInputError(
            "Beta PSA input's SE is too large for its mean (variance must be below mean*(1-mean))"
        )
    common = mean * (1.0 - mean) / variance - 1.0
    return mean * common, (1.0 - mean) * common


def _gamma_from_mean_se(mean: float, se: float) -> tuple[float, float]:
    if mean <= 0:
        raise PSAInputError("Gamma PSA input needs a positive mean")
    if se <= 0:
        raise PSAInputError("Gamma PSA input needs a positive SE")
    variance = se * se
    shape = (mean * mean) / variance
    scale = variance / mean
    return shape, scale


def _lognormal_from_mean_se(mean: float, se: float) -> tuple[float, float]:
    if mean <= 0:
        raise PSAInputError("Lognormal PSA input needs a positive mean")
    if se <= 0:
        raise PSAInputError("Lognormal PSA input needs a positive SE")
    variance = se * se
    sigma2 = float(np.log(1.0 + variance / (mean * mean)))
    mu = float(np.log(mean) - sigma2 / 2.0)
    return mu, float(np.sqrt(sigma2))


def make_sampler(psa_type: str, values: Mapping[str, Any]) -> Callable[[np.random.Generator], float]:
    """Build a `sampler(rng) -> float` function for one PSA setup row.

    Beta/Gamma/Normal/Lognormal take Mean+SE (converted to each
    distribution's native shape parameters via method-of-moments); Uniform
    takes Min/Max; Triangular takes Min/Mode/Max. All numeric parsing and
    validation happens here, at setup time -- not inside the Monte Carlo
    loop -- so a bad input is reported once, immediately.
    """

    if psa_type == "Beta":
        alpha, beta = _beta_from_mean_se(_as_float(values, "mean", "Beta"), _as_float(values, "se", "Beta"))
        return lambda rng: float(rng.beta(alpha, beta))
    if psa_type == "Gamma":
        shape, scale = _gamma_from_mean_se(_as_float(values, "mean", "Gamma"), _as_float(values, "se", "Gamma"))
        return lambda rng: float(rng.gamma(shape, scale))
    if psa_type == "Normal":
        mean = _as_float(values, "mean", "Normal")
        se = _as_float(values, "se", "Normal")
        if se < 0:
            raise PSAInputError("Normal PSA input needs a non-negative SE")
        return lambda rng: float(rng.normal(mean, se))
    if psa_type == "Lognormal":
        mu, sigma = _lognormal_from_mean_se(_as_float(values, "mean", "Lognormal"), _as_float(values, "se", "Lognormal"))
        return lambda rng: float(rng.lognormal(mu, sigma))
    if psa_type == "Uniform":
        low = _as_float(values, "min", "Uniform")
        high = _as_float(values, "max", "Uniform")
        if not (low < high):
            raise PSAInputError("Uniform PSA input needs Min < Max")
        return lambda rng: float(rng.uniform(low, high))
    if psa_type == "Triangular":
        low = _as_float(values, "min", "Triangular")
        mode = _as_float(values, "mode", "Triangular")
        high = _as_float(values, "max", "Triangular")
        if not (low <= mode <= high) or low == high:
            raise PSAInputError("Triangular PSA input needs Min <= Mode <= Max (and Min different from Max)")
        return lambda rng: float(rng.triangular(low, mode, high))
    raise PSAInputError(f"Unsupported PSA distribution '{psa_type}'")


def make_ppf(psa_type: str, values: Mapping[str, Any]) -> Callable[[float], float]:
    """Inverse-CDF twin of `make_sampler` (same parameterisation), used by
    Latin Hypercube sampling: `ppf(u)` for u in (0, 1)."""

    from scipy import stats

    if psa_type == "Beta":
        alpha, beta = _beta_from_mean_se(_as_float(values, "mean", "Beta"), _as_float(values, "se", "Beta"))
        return lambda u: float(stats.beta.ppf(u, alpha, beta))
    if psa_type == "Gamma":
        shape, scale = _gamma_from_mean_se(_as_float(values, "mean", "Gamma"), _as_float(values, "se", "Gamma"))
        return lambda u: float(stats.gamma.ppf(u, shape, scale=scale))
    if psa_type == "Normal":
        mean = _as_float(values, "mean", "Normal")
        se = _as_float(values, "se", "Normal")
        if se < 0:
            raise PSAInputError("Normal PSA input needs a non-negative SE")
        return lambda u: float(stats.norm.ppf(u, mean, se))
    if psa_type == "Lognormal":
        mu, sigma = _lognormal_from_mean_se(_as_float(values, "mean", "Lognormal"), _as_float(values, "se", "Lognormal"))
        return lambda u: float(stats.lognorm.ppf(u, sigma, scale=float(np.exp(mu))))
    if psa_type == "Uniform":
        low = _as_float(values, "min", "Uniform")
        high = _as_float(values, "max", "Uniform")
        if not (low < high):
            raise PSAInputError("Uniform PSA input needs Min < Max")
        return lambda u: float(low + u * (high - low))
    if psa_type == "Triangular":
        low = _as_float(values, "min", "Triangular")
        mode = _as_float(values, "mode", "Triangular")
        high = _as_float(values, "max", "Triangular")
        if not (low <= mode <= high) or low == high:
            raise PSAInputError("Triangular PSA input needs Min <= Mode <= Max (and Min different from Max)")
        return lambda u: float(stats.triang.ppf(u, (mode - low) / (high - low), loc=low, scale=high - low))
    raise PSAInputError(f"Unsupported PSA distribution '{psa_type}'")


def latin_hypercube_unit(iterations: int, columns: int, rng: np.random.Generator) -> np.ndarray:
    """(iterations x columns) uniforms in (0, 1): each column has exactly one
    point in each of the `iterations` equal-probability strata, in an
    independent random order."""

    out = np.empty((iterations, columns))
    for column in range(columns):
        out[:, column] = (rng.permutation(iterations) + rng.random(iterations)) / iterations
    return out


def parse_target_id(target_id: str) -> tuple[str, str, str | None]:
    """Split a PSA target id into (kind, name_or_index, parameter_key).

    Mirrors `_psa_target_options` in app.py: "param:<name>" targets a
    named Parameter or Variable; "dist:<index>:<parameter_key>" targets one
    shape/scale slot of a Distribution row.
    """

    if target_id.startswith("param:"):
        return "param", target_id[len("param:"):], None
    if target_id.startswith("dist:"):
        parts = target_id.split(":", 2)
        if len(parts) != 3:
            raise PSAInputError(f"Malformed PSA target id '{target_id}'")
        return "dist", parts[1], parts[2]
    raise PSAInputError(f"Unrecognised PSA target id '{target_id}'")


@dataclass(frozen=True)
class _PSATarget:
    target_id: str
    kind: str
    name_or_index: str
    parameter_key: str | None
    sampler: Callable[[np.random.Generator], float]
    # Empirical draws (e.g. NMA posterior samples): when set, every iteration
    # takes samples[index] with one shared index per `joint` group, so the
    # joint distribution (correlations, skew) of the group is kept exactly.
    samples: np.ndarray | None = None
    joint: str | None = None


def empirical_overrides(document: Mapping[str, Any], psa_parameters: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """`document["psa_empirical"]` = {target_id: {"samples": [...], "joint": key,
    "signature": {psa_type, values}}}. An entry applies only while the
    PSA row still carries the signature it was written with, so editing
    the row's distribution by hand switches back to that distribution."""

    entries = document.get("psa_empirical") or {}
    if not isinstance(entries, Mapping):
        return {}
    active: dict[str, dict[str, Any]] = {}
    for row in psa_parameters:
        target_id = str(row.get("target_id", ""))
        entry = entries.get(target_id)
        if not isinstance(entry, Mapping) or not entry.get("samples"):
            continue
        signature = entry.get("signature") or {}
        if signature:
            values = {k: _as_float({k: v}, k, "signature") for k, v in (row.get("values") or {}).items()
                      if v not in (None, "")}
            expected = {k: float(v) for k, v in (signature.get("values") or {}).items()}
            same = str(row.get("psa_type")) == str(signature.get("psa_type")) and all(
                k in values and abs(values[k] - expected[k]) <= 1e-9 * max(1.0, abs(expected[k])) for k in expected)
            if not same:
                continue
        active[target_id] = dict(entry)
    return active


def _nan_if_none(value: float | int | None) -> float:
    """NaN placeholder for a per-strategy statistic a non-microsim
    evaluate_scenario never populates (see StrategyOutcome's
    patient_cost_variance/patient_effect_variance/... fields)."""

    return float("nan") if value is None else float(value)


def _psa_worker(
    scenario: Mapping[str, Any],
    param_overrides: Mapping[str, float],
    child_seed: int | None,
    evaluate_scenario: Callable[..., SensitivityResult],
) -> SensitivityResult:
    """Evaluate ONE PSA trial's scenario in a worker process.

    Module-level (picklable) so ProcessPoolExecutor can ship it to a
    spawned worker. `child_seed` reconstructs the per-trial RNG derived
    from the parent run's master seed via SeedSequence.spawn() (see
    run_psa): the per-trial "per_psa" microsim distribution tier then
    draws from the same child generator it would have used in a
    sequential run. `None` when evaluate_scenario has no `psa_rng`
    parameter (Markov/Tree), so those engines run exactly as before.
    """

    _ensure_worker_import_path()
    extra: dict[str, Any] = {}
    if child_seed is not None:
        extra["psa_rng"] = np.random.default_rng(np.random.SeedSequence(child_seed))
    return evaluate_scenario(scenario, overrides=param_overrides, **extra)


def _ensure_worker_import_path() -> None:
    """Put the app's source tree back on sys.path inside a spawned worker.

    In the frozen desktop app a spawned ProcessPoolExecutor worker
    re-executes the bundled binary, so it starts with a clean sys.path
    that doesn't include the copied app payload. The launcher exports
    DECIDE_APP_DIR (inherited by every child) so the worker can restore
    the path before unpickling the scenario's evaluate_scenario. A no-op
    when running from source (the path is already importable) or when the
    variable isn't set.
    """

    import os as _os
    import sys as _sys

    app_dir = _os.environ.get("DECIDE_APP_DIR")
    if not app_dir:
        return
    for path in (app_dir, _os.path.join(app_dir, "src")):
        if path not in _sys.path:
            _sys.path.insert(0, path)


def _psa_sequential(
    scenario: Mapping[str, Any],
    param_overrides: Mapping[str, float],
    rng: np.random.Generator,
    evaluate_scenario: Callable[..., SensitivityResult],
    supports_psa_rng: bool,
) -> SensitivityResult:
    """The sequential per-iteration call, kept byte-for-byte identical to
    the pre-parallel implementation: the SAME shared `rng` object is passed
    straight through as psa_rng (not a spawned child), so a fixed seed
    reproduces the historical sequential run exactly."""

    extra: dict[str, Any] = {}
    if supports_psa_rng:
        extra["psa_rng"] = rng
    return evaluate_scenario(scenario, overrides=param_overrides, **extra)


@dataclass
class PSAResult:
    """The raw Monte Carlo sample a PSA run produces. Every CE plane /
    CEAC / CEAF / summary / convergence / EVPI / EVPPI helper below is
    pure post-processing of this, so `run_psa` only needs to run once."""

    strategy_ids: list[str]
    strategy_names: dict[str, str]
    parameter_labels: list[str]
    costs: np.ndarray  # shape (n_iterations, n_strategies)
    effects: np.ndarray  # shape (n_iterations, n_strategies)
    samples: np.ndarray  # shape (n_iterations, n_parameters)
    requested_iterations: int
    errors: list[str] = field(default_factory=list)
    # Microsimulation-only per-trial patient-level statistics (shape
    # (n_iterations, n_strategies)), NaN wherever unavailable -- Markov/Tree
    # PSA, or a strategy with no patient-level tracking. See
    # decompose_psa_noise().
    patient_cost_variance: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    patient_effect_variance: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    patient_cost_effect_covariance: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    patient_counts: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))

    @property
    def n_iterations(self) -> int:
        return self.costs.shape[0]

    def strategy_index(self, strategy_id: str) -> int:
        return self.strategy_ids.index(strategy_id)


def _nearest_positive_semidefinite_correlation(matrix: np.ndarray) -> np.ndarray:
    """Clip a user-supplied correlation matrix's eigenvalues to strictly
    positive before Cholesky-decomposing it: pairwise correlations typed
    in one at a time (target A-B = 0.6, B-C = 0.6, A-C = -0.6, say) are
    not guaranteed to form a mathematically consistent (positive
    semi-definite) correlation matrix, and this repairs that with the
    smallest change (eigenvalue clipping), renormalized back to a unit
    diagonal, rather than rejecting the setup outright."""

    symmetric = (matrix + matrix.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    eigenvalues = np.clip(eigenvalues, 1e-8, None)
    reconstructed = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    scale = np.sqrt(np.diag(reconstructed))
    scale[scale == 0] = 1.0
    return reconstructed / np.outer(scale, scale)


def _iman_conover_reorder(
    sample_columns: np.ndarray, correlation_matrix: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Reorder each column of `sample_columns` (n_iterations x k, one
    already-drawn column per correlated PSA target) so their rank
    correlation matches `correlation_matrix`, without changing any
    column's own set of realized values -- Iman & Conover's (1982)
    distribution-free rank correlation method: every PSA target keeps
    its own exact configured marginal distribution, and only *which PSA
    iteration* each already-drawn value lands in changes.

    Uses iid standard-normal auxiliary scores rather than the original
    paper's van der Waerden scores (a standard simplification that avoids
    a scipy dependency for the normal quantile function); the induced
    correlation is still asymptotically exact and close in finite
    samples.
    """

    n_iterations, k = sample_columns.shape
    if k < 2 or n_iterations < 2:
        return sample_columns
    psd_correlation = _nearest_positive_semidefinite_correlation(correlation_matrix)
    cholesky_factor = np.linalg.cholesky(psd_correlation)
    auxiliary_normals = rng.standard_normal((n_iterations, k))
    correlated_scores = auxiliary_normals @ cholesky_factor.T
    reordered = np.empty_like(sample_columns)
    for column in range(k):
        sorted_values = np.sort(sample_columns[:, column])
        rank_order = np.argsort(np.argsort(correlated_scores[:, column]))
        reordered[:, column] = sorted_values[rank_order]
    return reordered


JOINT_DISTRIBUTION_KINDS = ["Dirichlet", "Multivariate Normal"]

_BARE_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def generate_joint_samples(
    kind: str, params: Mapping[str, Any], n: int, seed: int | None = None
) -> np.ndarray:
    """Pre-generate `n` joint draws for a "Correlated Distribution" PSA
    group (see the group's own doc), returned as an (n, k) array -- one
    column per group member, in the same order as the member list the
    caller used to build `params`.

    - Dirichlet: `params = {"alpha": [a_1, ..., a_k]}` (concentration per
      member; every row of the result sums to exactly 1, so it needs no
      further correction).
    - Multivariate Normal: `params = {"mean": [...], "sd": [...],
      "correlation": [[...]]}` -- built into a covariance matrix via
      `_nearest_positive_semidefinite_correlation` (the same repair the
      pairwise Correlation table already applies), so an inconsistent
      pairwise correlation matrix is corrected rather than rejected.

    Feed each column into `document["psa_empirical"][target_id] =
    {"samples": column.tolist(), "joint": <shared key>, "signature":
    {...}}` -- `run_psa`'s existing empirical/joint mechanism (see its own
    docstring) then reproduces this exact joint distribution with zero
    changes to the PSA loop itself.
    """

    rng = np.random.default_rng(seed)
    if kind == "Dirichlet":
        alpha = np.asarray(params["alpha"], dtype=float)
        if alpha.ndim != 1 or alpha.size < 2:
            raise PSAInputError("Dirichlet needs at least 2 members")
        if np.any(alpha <= 0):
            raise PSAInputError("Dirichlet's alpha (concentration) must be strictly positive for every member")
        return rng.dirichlet(alpha, size=n)
    if kind == "Multivariate Normal":
        mean = np.asarray(params["mean"], dtype=float)
        sd = np.asarray(params["sd"], dtype=float)
        correlation = np.asarray(params["correlation"], dtype=float)
        k = mean.size
        if sd.size != k or correlation.shape != (k, k):
            raise PSAInputError("Multivariate Normal: mean/sd/correlation sizes don't match")
        if np.any(sd <= 0):
            raise PSAInputError("Multivariate Normal's SD must be strictly positive for every member")
        psd_correlation = _nearest_positive_semidefinite_correlation(correlation)
        covariance = psd_correlation * np.outer(sd, sd)
        return rng.multivariate_normal(mean, covariance, size=n)
    raise PSAInputError(f"Unsupported joint distribution kind '{kind}'")


def joint_group_empirical_entries(
    kind: str,
    member_target_ids: Sequence[str],
    params: Mapping[str, Any],
    n: int,
    joint_key: str,
    seed: int | None = None,
) -> dict[str, dict[str, Any]]:
    """`generate_joint_samples` plus the bookkeeping (`joint` key) needed
    to write the result straight into `document["psa_empirical"]`, one
    entry per member -- the UI's "Correlated Distribution" Apply button's
    whole job.

    `signature` is deliberately left empty here (not the NMA-empirical
    convention's `{psa_type, values}`, see `empirical_overrides`): that
    convention re-validates against a PSA row's OWN marginal-distribution
    fields, which a joint-group member doesn't have in the same shape
    (its group's `alpha`/`mean`/`sd`/`correlation` live in
    `document["psa_joint_groups"]`, not the row itself), and an empty
    signature makes an entry apply unconditionally once written.
    Staleness (the group's own parameters edited since the pool was last
    generated) is instead the UI's job: compare the live group definition
    in `document["psa_joint_groups"]` against what was Applied last, and
    prompt to Apply again when they differ.
    """

    samples = generate_joint_samples(kind, params, n, seed=seed)
    return {
        target_id: {
            "samples": samples[:, index].tolist(),
            "joint": joint_key,
            "signature": {},
        }
        for index, target_id in enumerate(member_target_ids)
    }


def dirichlet_group_targets_for_state(transitions: Sequence[Mapping[str, Any]], state_id: str) -> list[str]:
    """The exact PSA target ids (`param:<name>`) a Dirichlet "Correlated
    Distribution" group must cover to represent `state_id`'s full set of
    outgoing transition probabilities.

    Every outgoing transition from `state_id` must have a probability
    that is a bare Parameter/Variable reference (just a name, e.g.
    "p_ab" -- not a formula, a literal number, or the complement marker).
    A Dirichlet group's draws already sum to exactly 1 on their own, so a
    state can't also carry a complement transition (its own "remaining
    probability" marker) alongside a Dirichlet group -- that would make
    the complement always resolve to 0, silently, which is rejected here
    instead.
    """

    from .models import is_complement

    outgoing = [t for t in transitions if str(t.get("from_state", "")) == state_id]
    if not outgoing:
        raise PSAInputError(f"State '{state_id}' has no outgoing transitions")

    targets: list[str] = []
    bad: list[str] = []
    for transition in outgoing:
        to_state = str(transition.get("to_state", "?"))
        probability = transition.get("probability")
        if is_complement(probability):
            raise PSAInputError(
                f"State '{state_id}' has a complement transition (to '{to_state}'): a Dirichlet "
                "group needs every outgoing probability to be an explicit Parameter, since the "
                "group's draws already sum to 1 on their own -- remove the complement transition "
                "or give it its own explicit Parameter first."
            )
        if not (isinstance(probability, str) and _BARE_IDENTIFIER_RE.fullmatch(probability.strip())):
            bad.append(f"'{state_id}' -> '{to_state}'")
            continue
        targets.append(f"param:{probability.strip()}")

    if bad:
        raise PSAInputError(
            "These outgoing transitions aren't a plain Parameter/Variable reference (a formula or "
            "literal number can't be part of a Correlated Distribution group): " + ", ".join(bad)
        )
    if len(set(targets)) != len(targets):
        raise PSAInputError(f"State '{state_id}' has the same Parameter used on more than one outgoing transition")
    return targets


def validate_dirichlet_group(
    transitions: Sequence[Mapping[str, Any]], state_id: str, target_ids: Sequence[str]
) -> None:
    """Raise PSAInputError unless `target_ids` is EXACTLY the full set
    `dirichlet_group_targets_for_state` requires for `state_id` -- no
    member missing, none extra. Called before Apply generates any
    samples, so a mismatched group is caught immediately with the
    specific Parameters at fault, not after a pool has already been
    written."""

    expected = set(dirichlet_group_targets_for_state(transitions, state_id))
    got = set(target_ids)
    missing = expected - got
    extra = got - expected
    if missing or extra:
        parts = []
        if missing:
            parts.append("missing: " + ", ".join(sorted(m.removeprefix("param:") for m in missing)))
        if extra:
            parts.append(
                "not part of this state's outgoing transitions: "
                + ", ".join(sorted(e.removeprefix("param:") for e in extra))
            )
        raise PSAInputError(
            f"Dirichlet group for state '{state_id}' doesn't match its full set of outgoing "
            "probabilities (" + "; ".join(parts) + ")"
        )


def run_psa(
    document: Mapping[str, Any],
    psa_parameters: Sequence[Mapping[str, Any]],
    *,
    iterations: int,
    evaluate_scenario: Callable[..., SensitivityResult],
    seed: int | None = None,
    max_errors: int = 20,
    progress_callback: Callable[[int, int], None] | None = None,
    correlations: Sequence[Mapping[str, Any]] | None = None,
    workers: int | None = None,
    sampling: str = "random",
) -> PSAResult:
    """Run the Monte Carlo simulation.

    `sampling` is "random" (plain Monte Carlo, the historical behaviour,
    bit-identical for a fixed seed) or "lhs" (Latin Hypercube: every
    standard-distribution target is stratified into `iterations` equal-
    probability slices, one draw per slice, shuffled independently per
    target; empirical/joint draws stay random, and a pairwise correlation
    still reorders the columns afterwards, which keeps each marginal).

    Every configured PSA parameter is resampled each iteration, applied to
    a document copy (named Parameters/Variables via `evaluate_scenario`'s
    own `overrides`; Distribution shape/scale slots by mutating a
    per-iteration copy of `document["distributions"]`), then every
    strategy is re-evaluated and its total cost/effect recorded.

    `correlations` is an optional list of `{"target_a", "target_b", "rho"}`
    rows (target ids as in `psa_parameters`' own `target_id`, rho in
    [-1, 1]) declaring a pairwise rank correlation between two PSA
    targets' independent draws -- e.g. two costs that plausibly rise and
    fall together, so PSA doesn't spend probability mass on unrealistic
    combinations (a cheap drug with an implausibly high utility, say)
    that fully independent sampling would otherwise draw. Every target
    keeps its OWN configured marginal distribution exactly -- only the
    pairing across targets changes, via the distribution-free
    Iman-Conover rank correlation method (see `_iman_conover_reorder`).
    Any target not named in `correlations` is sampled exactly as before
    (fully independent), and when `correlations` is empty/None the whole
    run is byte-for-byte identical to before this feature existed.

    `workers` controls parallelism: `None` or `1` runs the historical
    sequential loop (bit-identical to the pre-parallel implementation for
    a fixed seed); an integer > 1 evaluates trials across that many worker
    PROCESSES (ProcessPoolExecutor). In parallel mode every trial's
    parameter sample matrix is pre-drawn from the parent RNG (same values
    as the sequential run would draw, in the same order) so parameter
    uncertainty is bit-identical to sequential; only each trial's
    per-trial "per_psa" microsim distribution draw is re-keyed via
    SeedSequence.spawn() -- statistically equivalent, deterministic given
    the seed, and reproducible, just not bit-identical to the sequential
    microsim run (which is unavoidable once iterations no longer share one
    in-order RNG). Results are assembled in iteration order regardless of
    completion order.
    """

    if iterations < 1:
        raise PSAInputError("PSA needs at least one iteration")
    if sampling not in ("random", "lhs"):
        raise PSAInputError(f"Unknown PSA sampling method '{sampling}' (use 'random' or 'lhs')")
    if not psa_parameters:
        raise PSAInputError("Add at least one PSA parameter before running PSA")

    distributions_list = document.get("distributions", [])
    targets: list[_PSATarget] = []
    for row in psa_parameters:
        target_id = str(row.get("target_id", ""))
        kind, name_or_index, parameter_key = parse_target_id(target_id)
        if kind == "dist":
            index = int(name_or_index)
            if index < 0 or index >= len(distributions_list):
                raise PSAInputError(f"'{target_id}': that distribution no longer exists")
        sampler = make_sampler(str(row.get("psa_type", "Beta")), row.get("values", {}) or {})
        targets.append(_PSATarget(target_id, kind, name_or_index, parameter_key, sampler))
    empirical = empirical_overrides(document, psa_parameters)
    for position, target in enumerate(targets):
        entry = empirical.get(target.target_id)
        if entry:
            samples = np.asarray(entry["samples"], dtype=float)
            if samples.ndim == 1 and samples.size > 0 and np.all(np.isfinite(samples)):
                targets[position] = dataclasses.replace(target, samples=samples,
                                                        joint=str(entry.get("joint") or target.target_id))
    joint_groups: dict[str, list[int]] = {}
    for index, target in enumerate(targets):
        if target.samples is not None:
            joint_groups.setdefault(target.joint, []).append(index)

    def draw_empirical(row_values: list[float] | np.ndarray) -> None:
        """Overwrite the empirical targets' values with one joint draw per group."""

        for members in joint_groups.values():
            size = min(len(targets[i].samples) for i in members)
            index = int(rng.integers(size))
            for i in members:
                row_values[i] = float(targets[i].samples[index])

    target_index_by_id = {target.target_id: index for index, target in enumerate(targets)}
    correlation_pairs: list[tuple[int, int, float]] = []
    for row in correlations or []:
        target_a_id = str(row.get("target_a", ""))
        target_b_id = str(row.get("target_b", ""))
        if target_a_id not in target_index_by_id or target_b_id not in target_index_by_id:
            raise PSAInputError(
                f"Correlation references a PSA target that isn't in this PSA's parameter list "
                f"('{target_a_id}' / '{target_b_id}')"
            )
        if target_a_id == target_b_id:
            raise PSAInputError(f"Correlation can't pair a PSA target with itself ('{target_a_id}')")
        rho = _as_float(row, "rho", "Correlation")
        if not (-1.0 <= rho <= 1.0):
            raise PSAInputError(f"Correlation coefficient must be between -1 and 1 (got {rho})")
        a_index, b_index = target_index_by_id[target_a_id], target_index_by_id[target_b_id]
        if targets[a_index].samples is not None and targets[b_index].samples is not None \
                and targets[a_index].joint == targets[b_index].joint:
            continue  # a joint empirical draw already carries this correlation exactly
        correlation_pairs.append((a_index, b_index, rho))

    correlated_target_indices = sorted({index for pair in correlation_pairs for index in pair[:2]})
    correlation_matrix: np.ndarray | None = None
    if correlated_target_indices:
        k = len(correlated_target_indices)
        local_index_of = {global_index: local for local, global_index in enumerate(correlated_target_indices)}
        correlation_matrix = np.eye(k)
        for target_a_index, target_b_index, rho in correlation_pairs:
            local_a, local_b = local_index_of[target_a_index], local_index_of[target_b_index]
            correlation_matrix[local_a, local_b] = rho
            correlation_matrix[local_b, local_a] = rho

    has_dist_targets = any(target.kind == "dist" for target in targets)
    base_document = dict(document)  # shallow: cheap per-iteration starting point

    rng = np.random.default_rng(seed)
    # Microsim's evaluate_microsim_scenario() accepts an optional psa_rng
    # to draw its "per_psa" tier once per PSA trial, shared across every
    # strategy compared in that trial (see its docstring for the full
    # two-level Monte-Carlo picture). Markov's/Tree's evaluate_scenario
    # functions have no such parameter and are completely unaffected by
    # this branch -- run_psa() stays fully model-agnostic either way.
    supports_psa_rng = "psa_rng" in inspect.signature(evaluate_scenario).parameters
    strategy_ids: list[str] = []
    strategy_names: dict[str, str] = {}
    costs_rows: list[list[float]] = []
    effects_rows: list[list[float]] = []
    sample_rows: list[list[float]] = []
    patient_cost_var_rows: list[list[float]] = []
    patient_effect_var_rows: list[list[float]] = []
    patient_cov_rows: list[list[float]] = []
    patient_count_rows: list[list[float]] = []
    errors: list[str] = []

    # When no correlation is configured, every target is drawn fresh each
    # iteration exactly as before this feature existed (so a fixed seed's
    # output is completely unaffected). A correlation needs every target's
    # full column of draws generated up front before Iman-Conover can
    # reorder it, so that case pre-draws the whole (iterations x targets)
    # sample matrix first instead of drawing inline per iteration below.
    raw_sample_matrix: np.ndarray | None = None
    if correlation_matrix is not None or sampling == "lhs":
        raw_sample_matrix = np.empty((iterations, len(targets)))
        if sampling == "lhs":
            unit = latin_hypercube_unit(iterations, len(targets), rng)
            for target_index, (row, target) in enumerate(zip(psa_parameters, targets)):
                if target.samples is not None:
                    raw_sample_matrix[:, target_index] = 0.0  # overwritten by the joint empirical draw below
                    continue
                ppf = make_ppf(str(row.get("psa_type", "Beta")), row.get("values", {}) or {})
                raw_sample_matrix[:, target_index] = [ppf(float(u)) for u in unit[:, target_index]]
            for iteration in range(iterations):
                draw_empirical(raw_sample_matrix[iteration])
        else:
            for iteration in range(iterations):
                for target_index, target in enumerate(targets):
                    raw_sample_matrix[iteration, target_index] = target.sampler(rng)
                draw_empirical(raw_sample_matrix[iteration])
        if correlation_matrix is not None:
            correlated_columns = raw_sample_matrix[:, correlated_target_indices]
            raw_sample_matrix[:, correlated_target_indices] = _iman_conover_reorder(
                correlated_columns, correlation_matrix, rng
            )

    # ------------------------------------------------------------------
    # Parallel execution
    # ------------------------------------------------------------------
    # Every trial's scenario (document copy + overrides + per-trial seed)
    # is built up front from the PARENT rng, in iteration order -- so the
    # parameter sample matrix is identical whether we then evaluate the
    # trials sequentially or farm them out to worker processes. The parent
    # rng is then discarded; workers each get their own spawned seed.
    n_workers = 1 if workers is None else max(1, int(workers))
    if n_workers > 1 and iterations > 1:
        # Pre-draw every iteration's sample row in order (same numbers the
        # sequential loop would draw).
        if raw_sample_matrix is None:
            pre_samples = np.empty((iterations, len(targets)))
            for iteration in range(iterations):
                for target_index, target in enumerate(targets):
                    pre_samples[iteration, target_index] = target.sampler(rng)
                draw_empirical(pre_samples[iteration])
        else:
            pre_samples = raw_sample_matrix

        # Spawn one child seed per iteration from the parent's SeedSequence
        # so every trial's per-trial RNG is deterministic given `seed` and
        # independent across workers (results identical regardless of which
        # process happens to run which trial).
        parent_seq = np.random.SeedSequence(seed)
        child_seeds = [int(seq.generate_state(1)[0]) for seq in parent_seq.spawn(iterations)]

        def build_scenario(iteration: int) -> tuple[dict, dict]:
            scenario = dict(base_document)
            if has_dist_targets:
                scenario["distributions"] = copy.deepcopy(distributions_list)
            param_overrides: dict[str, float] = {}
            for target_index, target in enumerate(targets):
                value = float(pre_samples[iteration, target_index])
                if target.kind == "param":
                    param_overrides[target.name_or_index] = value
                else:
                    distribution = scenario["distributions"][int(target.name_or_index)]
                    distribution.setdefault("parameters", {})[target.parameter_key] = value
            return scenario, param_overrides

        futures = []
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            for iteration in range(iterations):
                scenario, param_overrides = build_scenario(iteration)
                child_seed = child_seeds[iteration] if supports_psa_rng else None
                futures.append(
                    pool.submit(_psa_worker, scenario, param_overrides, child_seed, evaluate_scenario)
                )
            for iteration, future in enumerate(futures):
                result = future.result()
                if progress_callback is not None:
                    progress_callback(iteration + 1, iterations)
                sampled_row = [float(v) for v in pre_samples[iteration]]
                if result.errors:
                    if len(errors) < max_errors:
                        errors.append(f"Iteration {iteration + 1}: " + "; ".join(result.errors))
                    continue

                if not strategy_ids:
                    strategy_ids = [outcome.strategy_id for outcome in result.outcomes]
                    strategy_names = {outcome.strategy_id: outcome.strategy_name for outcome in result.outcomes}

                outcome_by_id = {outcome.strategy_id: outcome for outcome in result.outcomes}
                if any(sid not in outcome_by_id for sid in strategy_ids):
                    if len(errors) < max_errors:
                        errors.append(f"Iteration {iteration + 1}: strategy set changed mid-run, skipped")
                    continue

                costs_rows.append([outcome_by_id[sid].total_cost for sid in strategy_ids])
                effects_rows.append([outcome_by_id[sid].total_effect for sid in strategy_ids])
                sample_rows.append(sampled_row)
                patient_cost_var_rows.append([_nan_if_none(outcome_by_id[sid].patient_cost_variance) for sid in strategy_ids])
                patient_effect_var_rows.append([_nan_if_none(outcome_by_id[sid].patient_effect_variance) for sid in strategy_ids])
                patient_cov_rows.append([_nan_if_none(outcome_by_id[sid].patient_cost_effect_covariance) for sid in strategy_ids])
                patient_count_rows.append([_nan_if_none(outcome_by_id[sid].n_patients) for sid in strategy_ids])
    else:
        for iteration in range(iterations):
            scenario = dict(base_document)
            if has_dist_targets:
                scenario["distributions"] = copy.deepcopy(distributions_list)
            param_overrides: dict[str, float] = {}
            sampled_row: list[float] = []
            if raw_sample_matrix is None:
                sampled_row = [target.sampler(rng) for target in targets]
                draw_empirical(sampled_row)
            for target_index, target in enumerate(targets):
                value = (
                    float(raw_sample_matrix[iteration, target_index])
                    if raw_sample_matrix is not None
                    else sampled_row[target_index]
                )
                if raw_sample_matrix is not None:
                    sampled_row.append(value)
                if target.kind == "param":
                    param_overrides[target.name_or_index] = value
                else:
                    distribution = scenario["distributions"][int(target.name_or_index)]
                    distribution.setdefault("parameters", {})[target.parameter_key] = value

            result = _psa_sequential(
                scenario, param_overrides, rng, evaluate_scenario, supports_psa_rng
            )
            if progress_callback is not None:
                progress_callback(iteration + 1, iterations)
            if result.errors:
                if len(errors) < max_errors:
                    errors.append(f"Iteration {iteration + 1}: " + "; ".join(result.errors))
                continue

            if not strategy_ids:
                strategy_ids = [outcome.strategy_id for outcome in result.outcomes]
                strategy_names = {outcome.strategy_id: outcome.strategy_name for outcome in result.outcomes}

            outcome_by_id = {outcome.strategy_id: outcome for outcome in result.outcomes}
            if any(sid not in outcome_by_id for sid in strategy_ids):
                if len(errors) < max_errors:
                    errors.append(f"Iteration {iteration + 1}: strategy set changed mid-run, skipped")
                continue

            costs_rows.append([outcome_by_id[sid].total_cost for sid in strategy_ids])
            effects_rows.append([outcome_by_id[sid].total_effect for sid in strategy_ids])
            sample_rows.append(sampled_row)
            patient_cost_var_rows.append([_nan_if_none(outcome_by_id[sid].patient_cost_variance) for sid in strategy_ids])
            patient_effect_var_rows.append([_nan_if_none(outcome_by_id[sid].patient_effect_variance) for sid in strategy_ids])
            patient_cov_rows.append([_nan_if_none(outcome_by_id[sid].patient_cost_effect_covariance) for sid in strategy_ids])
            patient_count_rows.append([_nan_if_none(outcome_by_id[sid].n_patients) for sid in strategy_ids])

    if not costs_rows:
        errors.append("Every PSA iteration failed; check the PSA parameter setup and the model itself")

    n_strategies = len(strategy_ids)
    return PSAResult(
        strategy_ids=strategy_ids,
        strategy_names=strategy_names,
        parameter_labels=[target.target_id for target in targets],
        costs=np.array(costs_rows, dtype=float) if costs_rows else np.zeros((0, n_strategies)),
        effects=np.array(effects_rows, dtype=float) if effects_rows else np.zeros((0, n_strategies)),
        samples=np.array(sample_rows, dtype=float) if sample_rows else np.zeros((0, len(targets))),
        requested_iterations=iterations,
        errors=errors,
        patient_cost_variance=np.array(patient_cost_var_rows, dtype=float) if patient_cost_var_rows else np.zeros((0, n_strategies)),
        patient_effect_variance=np.array(patient_effect_var_rows, dtype=float) if patient_effect_var_rows else np.zeros((0, n_strategies)),
        patient_cost_effect_covariance=np.array(patient_cov_rows, dtype=float) if patient_cov_rows else np.zeros((0, n_strategies)),
        patient_counts=np.array(patient_count_rows, dtype=float) if patient_count_rows else np.zeros((0, n_strategies)),
    )


def nmb_matrix(result: PSAResult, willingness_to_pay: float) -> np.ndarray:
    """Net monetary benefit per iteration per strategy: shape (n_iterations, n_strategies)."""

    return willingness_to_pay * result.effects - result.costs


@dataclass(frozen=True)
class CEPlaneSeries:
    strategy_id: str
    strategy_name: str
    incremental_cost: np.ndarray
    incremental_effect: np.ndarray
    mean_incremental_cost: float
    mean_incremental_effect: float


def ce_plane(result: PSAResult, reference_strategy_id: str | None = None) -> list[CEPlaneSeries]:
    """Incremental cost/effect vs a reference strategy, per PSA iteration,
    for every other strategy -- the raw points of a cost-effectiveness
    plane scatter."""

    if result.n_iterations == 0 or not result.strategy_ids:
        return []
    reference_id = reference_strategy_id or result.strategy_ids[0]
    ref_index = result.strategy_index(reference_id)
    series: list[CEPlaneSeries] = []
    for index, strategy_id in enumerate(result.strategy_ids):
        if strategy_id == reference_id:
            continue
        inc_cost = result.costs[:, index] - result.costs[:, ref_index]
        inc_effect = result.effects[:, index] - result.effects[:, ref_index]
        series.append(
            CEPlaneSeries(
                strategy_id=strategy_id,
                strategy_name=result.strategy_names.get(strategy_id, strategy_id),
                incremental_cost=inc_cost,
                incremental_effect=inc_effect,
                mean_incremental_cost=float(np.mean(inc_cost)),
                mean_incremental_effect=float(np.mean(inc_effect)),
            )
        )
    return series


def ceac_curve_nmb(result: PSAResult, wtp_grid: Sequence[float]) -> dict[str, np.ndarray]:
    """Probability each strategy has the single highest NMB, per WTP value.

    Generalises to any number of strategies: at every iteration, exactly
    one strategy "wins", so these curves always sum to 1 across strategies
    at each willingness-to-pay value.
    """

    probs = {sid: np.zeros(len(wtp_grid)) for sid in result.strategy_ids}
    n_iter = result.n_iterations
    if n_iter == 0:
        return probs
    for wtp_index, wtp in enumerate(wtp_grid):
        nmb = wtp * result.effects - result.costs
        winners = np.argmax(nmb, axis=1)
        counts = np.bincount(winners, minlength=len(result.strategy_ids))
        for strategy_index, sid in enumerate(result.strategy_ids):
            probs[sid][wtp_index] = counts[strategy_index] / n_iter
    return probs


def ceac_curve_icer(
    result: PSAResult, wtp_grid: Sequence[float], reference_strategy_id: str | None = None
) -> dict[str, np.ndarray]:
    """Probability each non-reference strategy is cost-effective against a
    fixed reference (the classic pairwise incremental decision rule).

    Unlike the NMB method, these curves are computed independently per
    strategy against the same reference, so with more than two strategies
    they need not sum to 1 -- each answers "is this one option worth it
    compared to the reference", not "which option wins overall".
    """

    if not result.strategy_ids:
        return {}
    reference_id = reference_strategy_id or result.strategy_ids[0]
    probs: dict[str, np.ndarray] = {}
    n_iter = result.n_iterations
    if n_iter == 0:
        return {sid: np.zeros(len(wtp_grid)) for sid in result.strategy_ids if sid != reference_id}
    ref_index = result.strategy_index(reference_id)
    for index, sid in enumerate(result.strategy_ids):
        if sid == reference_id:
            continue
        inc_cost = result.costs[:, index] - result.costs[:, ref_index]
        inc_effect = result.effects[:, index] - result.effects[:, ref_index]
        curve = np.empty(len(wtp_grid))
        for wtp_index, wtp in enumerate(wtp_grid):
            curve[wtp_index] = float(np.mean(wtp * inc_effect - inc_cost > 0))
        probs[sid] = curve
    return probs


@dataclass(frozen=True)
class CEAFPoint:
    willingness_to_pay: float
    probability: float
    frontier_strategy_id: str


def ceaf_curve(result: PSAResult, wtp_grid: Sequence[float]) -> list[CEAFPoint]:
    """Cost-effectiveness acceptability frontier: at each WTP, the
    strategy with the highest EXPECTED NMB, and that strategy's own CEAC
    probability. Only meaningful with more than two strategies (with
    exactly two, the frontier is just the higher of the two CEAC curves)."""

    if result.n_iterations == 0 or len(result.strategy_ids) < 2:
        return []
    nmb_curves = ceac_curve_nmb(result, wtp_grid)
    mean_effect = result.effects.mean(axis=0)
    mean_cost = result.costs.mean(axis=0)
    points: list[CEAFPoint] = []
    for wtp_index, wtp in enumerate(wtp_grid):
        expected_nmb = wtp * mean_effect - mean_cost
        best_index = int(np.argmax(expected_nmb))
        best_id = result.strategy_ids[best_index]
        points.append(
            CEAFPoint(
                willingness_to_pay=wtp,
                probability=float(nmb_curves[best_id][wtp_index]),
                frontier_strategy_id=best_id,
            )
        )
    return points


@dataclass(frozen=True)
class PSASummaryRow:
    strategy_id: str
    strategy_name: str
    mean_cost: float
    cost_ci_low: float
    cost_ci_high: float
    mean_effect: float
    effect_ci_low: float
    effect_ci_high: float
    incremental_cost: float
    incremental_effect: float
    icer: float | None
    probability_cost_effective: float


def summary_stats(
    result: PSAResult, willingness_to_pay: float, reference_strategy_id: str | None = None
) -> list[PSASummaryRow]:
    """One row per strategy: mean cost/effect with a 95% credible
    interval, incremental cost/effect and ICER vs the reference strategy,
    and the probability (share of PSA iterations) each strategy is the
    most cost-effective option at the given willingness-to-pay."""

    if result.n_iterations == 0 or not result.strategy_ids:
        return []
    reference_id = reference_strategy_id or result.strategy_ids[0]
    ref_index = result.strategy_index(reference_id)
    nmb = willingness_to_pay * result.effects - result.costs
    winners = np.argmax(nmb, axis=1)
    mean_ref_cost = float(np.mean(result.costs[:, ref_index]))
    mean_ref_effect = float(np.mean(result.effects[:, ref_index]))
    rows: list[PSASummaryRow] = []
    for index, sid in enumerate(result.strategy_ids):
        cost_samples = result.costs[:, index]
        effect_samples = result.effects[:, index]
        mean_cost = float(np.mean(cost_samples))
        mean_effect = float(np.mean(effect_samples))
        incremental_cost = mean_cost - mean_ref_cost
        incremental_effect = mean_effect - mean_ref_effect
        rows.append(
            PSASummaryRow(
                strategy_id=sid,
                strategy_name=result.strategy_names.get(sid, sid),
                mean_cost=mean_cost,
                cost_ci_low=float(np.percentile(cost_samples, 2.5)),
                cost_ci_high=float(np.percentile(cost_samples, 97.5)),
                mean_effect=mean_effect,
                effect_ci_low=float(np.percentile(effect_samples, 2.5)),
                effect_ci_high=float(np.percentile(effect_samples, 97.5)),
                incremental_cost=incremental_cost,
                incremental_effect=incremental_effect,
                icer=(incremental_cost / incremental_effect) if incremental_effect else None,
                probability_cost_effective=float(np.mean(winners == index)),
            )
        )
    return rows


@dataclass(frozen=True)
class ConvergenceSeries:
    iteration_numbers: np.ndarray
    running_mean_nmb: dict[str, np.ndarray]


def convergence_series(result: PSAResult, willingness_to_pay: float, step: int | None = None) -> ConvergenceSeries:
    """Cumulative running mean of each strategy's NMB (at a fixed WTP),
    sampled every `step` iterations, so a user can see whether the PSA
    estimate has stabilised or needs more iterations."""

    n_iter = result.n_iterations
    if n_iter == 0:
        return ConvergenceSeries(iteration_numbers=np.array([], dtype=int), running_mean_nmb={})
    nmb = willingness_to_pay * result.effects - result.costs
    cumulative_mean = np.cumsum(nmb, axis=0) / np.arange(1, n_iter + 1)[:, None]
    if step is None:
        step = max(1, n_iter // 200)
    idx = np.arange(step - 1, n_iter, step)
    if len(idx) == 0 or idx[-1] != n_iter - 1:
        idx = np.append(idx, n_iter - 1)
    return ConvergenceSeries(
        iteration_numbers=idx + 1,
        running_mean_nmb={sid: cumulative_mean[idx, i] for i, sid in enumerate(result.strategy_ids)},
    )


@dataclass(frozen=True)
class NoiseDecomposition:
    """Splits one strategy's between-PSA-trial NMB variance (at a given
    willingness-to-pay) into the part driven by resampled parameter
    uncertainty -- what a PSA is meant to measure -- vs. the part left
    over purely from simulating a finite `n_patients` per trial (Monte
    Carlo sampling noise). By the law of total variance:

        Var(trial NMB) = Var_params(E[NMB | params])
                        + E_params[Var(NMB | params)]

    the second term is estimated per trial as
    (wtp^2 * patient_effect_variance + patient_cost_variance
     - 2 * wtp * patient_cost_effect_covariance) / n_patients,
    using that trial's own patient-level sample statistics, then averaged
    across trials. Only meaningful for microsimulation PSA -- a Markov
    cohort trial has no patient-level sampling, so PSAResult's patient_*
    arrays are all NaN for it and no NoiseDecomposition rows come out."""

    strategy_id: str
    strategy_name: str
    total_variance: float
    parameter_variance: float
    sampling_noise_variance: float
    noise_fraction: float | None
    mean_n_patients: float
    recommended_n_patients: int | None


def decompose_psa_noise(
    result: PSAResult, willingness_to_pay: float, target_noise_fraction: float = 0.05
) -> list[NoiseDecomposition]:
    """Per strategy, decompose PSA's between-trial NMB variance into
    parameter uncertainty vs. finite-patient-sample noise (see
    NoiseDecomposition). A strategy with no patient-level data (non-microsim
    PSA, or every trial missing it) is omitted, not zero-filled -- an empty
    return means "not applicable", not "no noise". `recommended_n_patients`
    is set only when `noise_fraction` exceeds `target_noise_fraction`
    (default 5%), extrapolating from sampling-noise-variance's known 1/N
    scaling -- a suggestion to raise Model properties' n_patients, not an
    automatic change."""

    rows: list[NoiseDecomposition] = []
    n_iter = result.n_iterations
    if n_iter < 2 or not result.strategy_ids:
        return rows
    nmb = willingness_to_pay * result.effects - result.costs
    for index, sid in enumerate(result.strategy_ids):
        if result.patient_cost_variance.size == 0:
            break
        var_c = result.patient_cost_variance[:, index]
        var_e = result.patient_effect_variance[:, index]
        cov_ce = result.patient_cost_effect_covariance[:, index]
        n_patients = result.patient_counts[:, index]
        valid = ~np.isnan(var_c) & ~np.isnan(var_e) & ~np.isnan(cov_ce) & ~np.isnan(n_patients) & (n_patients > 0)
        if not np.any(valid):
            continue
        per_trial_noise = (
            willingness_to_pay**2 * var_e[valid] + var_c[valid] - 2 * willingness_to_pay * cov_ce[valid]
        ) / n_patients[valid]
        sampling_noise = float(np.mean(per_trial_noise))
        total_variance = float(np.var(nmb[:, index], ddof=1))
        parameter_variance = max(total_variance - sampling_noise, 0.0)
        noise_fraction = (sampling_noise / total_variance) if total_variance > 0 else None
        mean_n = float(np.mean(n_patients[valid]))
        recommended_n: int | None = None
        if noise_fraction is not None and noise_fraction > target_noise_fraction and mean_n > 0:
            recommended_n = int(np.ceil(mean_n * (noise_fraction / target_noise_fraction)))
        rows.append(
            NoiseDecomposition(
                strategy_id=sid,
                strategy_name=result.strategy_names.get(sid, sid),
                total_variance=total_variance,
                parameter_variance=parameter_variance,
                sampling_noise_variance=sampling_noise,
                noise_fraction=noise_fraction,
                mean_n_patients=mean_n,
                recommended_n_patients=recommended_n,
            )
        )
    return rows


def evpi_curve(result: PSAResult, wtp_grid: Sequence[float]) -> np.ndarray:
    """Per-patient EVPI at each WTP: E[max strategy NMB] - max strategy E[NMB]."""

    if result.n_iterations == 0:
        return np.zeros(len(wtp_grid))
    evpi = np.empty(len(wtp_grid))
    for wtp_index, wtp in enumerate(wtp_grid):
        nmb = wtp * result.effects - result.costs
        expected_max_nmb = float(np.mean(np.max(nmb, axis=1)))
        max_expected_nmb = float(np.max(np.mean(nmb, axis=0)))
        evpi[wtp_index] = max(0.0, expected_max_nmb - max_expected_nmb)
    return evpi


def population_effective_size(
    annual_population: float, time_horizon_years: int, discount_rate: float
) -> float:
    """Discounted patient-years a decision will affect -- the "effective
    population" in the Claxton/Briggs population-EVPI formula (Briggs,
    Sculpher & Claxton, "Decision Modelling for Health Economic
    Evaluation"): sum over t=0..T-1 of
    annual_population / (1+discount_rate)**t. Multiplying a per-patient
    EVPI/EVPPI value (evpi_curve()/evppi_by_parameter()/evppi_for_group())
    by this converts it to the total expected value of eliminating
    uncertainty across everyone the decision will affect over its time
    horizon -- the number an HTA body compares against the cost of a
    proposed further study when deciding whether it is worth funding.

    Year 0 (the first year) is undiscounted, matching the (1+rate)**-t
    discount-factor convention markov_engine.py's own per-cycle
    discounting uses, so a population discount rate equal to the model's
    own cost discount rate gives an internally consistent number.
    """

    if annual_population <= 0 or time_horizon_years <= 0:
        return 0.0
    years = np.arange(int(time_horizon_years))
    if discount_rate <= 0:
        return float(annual_population) * len(years)
    return float(annual_population) * float(np.sum((1.0 + discount_rate) ** -years))


def _polynomial_design_matrix(theta_matrix: np.ndarray, degree: int) -> np.ndarray:
    """Full polynomial feature matrix (every monomial up to total degree
    `degree`, including cross terms between parameters) for one or more
    standardized parameter columns -- e.g. for 2 parameters and degree=2:
    [1, a, b, a^2, a*b, b^2]. This is the natural multi-parameter
    generalisation of np.polyfit's single-variable Vandermonde matrix
    (numpy has no multivariate polyfit), used by both the single-parameter
    EVPPI (n_params=1, equivalent to np.polyfit/np.polyval) and
    evppi_for_group() (n_params>1) so the two share one fitting path.
    """

    n_samples, n_params = theta_matrix.shape
    columns = [np.ones(n_samples)]
    for d in range(1, degree + 1):
        for combo in combinations_with_replacement(range(n_params), d):
            column = np.ones(n_samples)
            for index in combo:
                column = column * theta_matrix[:, index]
            columns.append(column)
    return np.column_stack(columns)


def _standardize(theta_matrix: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-variance each parameter column before building
    polynomial features -- keeps higher-degree terms well-conditioned for
    np.linalg.lstsq regardless of the parameter's raw scale (a probability
    in [0,1] vs. a cost in the thousands), the same numerical-stability
    concern np.polyfit's internal scaling addresses for the single-variable
    case this replaces."""

    mean = theta_matrix.mean(axis=0)
    std = theta_matrix.std(axis=0)
    std = np.where(std == 0, 1.0, std)
    return (theta_matrix - mean) / std


def _select_degree_by_cv(
    theta_matrix: np.ndarray,
    y_matrix: np.ndarray,
    candidate_degrees: Sequence[int],
    k_folds: int = 5,
    seed: int = 0,
) -> int:
    """Picks the polynomial degree (from `candidate_degrees`) that best
    predicts `y_matrix` (one column per strategy's NMB) out-of-sample via
    k-fold cross-validation, instead of a fixed degree -- reduces the risk
    of under- or over-fitting the NMB-vs-parameter relationship that a
    fixed degree=2 risked. Falls back to the lowest candidate degree when
    there are too few PSA iterations to cross-validate meaningfully (each
    fold's training set must outnumber that degree's coefficient count)."""

    n_samples = theta_matrix.shape[0]
    if n_samples < k_folds * 4:
        return candidate_degrees[0]
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_samples)
    folds = np.array_split(indices, k_folds)
    best_degree = candidate_degrees[0]
    best_mse = np.inf
    for degree in candidate_degrees:
        n_coefficients = _polynomial_design_matrix(theta_matrix[:1], degree).shape[1]
        mse_total = 0.0
        n_scored = 0
        for fold in folds:
            test_idx = fold
            train_idx = np.setdiff1d(indices, fold, assume_unique=True)
            if len(train_idx) <= n_coefficients:
                continue
            design_train = _polynomial_design_matrix(theta_matrix[train_idx], degree)
            design_test = _polynomial_design_matrix(theta_matrix[test_idx], degree)
            coefficients, *_ = np.linalg.lstsq(design_train, y_matrix[train_idx], rcond=None)
            predicted = design_test @ coefficients
            mse_total += float(np.mean((predicted - y_matrix[test_idx]) ** 2))
            n_scored += 1
        if n_scored == 0:
            continue
        mean_mse = mse_total / n_scored
        if mean_mse < best_mse:
            best_mse = mean_mse
            best_degree = degree
    return best_degree


@dataclass(frozen=True)
class EVPPIRow:
    target_id: str
    evppi: float
    percent_of_evpi: float | None
    degree_used: int


def evppi_by_parameter(
    result: PSAResult,
    willingness_to_pay: float,
    degree: int | None = None,
    candidate_degrees: Sequence[int] = (1, 2, 3),
) -> list[EVPPIRow]:
    """Approximate single-parameter EVPPI via polynomial regression (the
    Strong/Oakley "regression method"): for each PSA parameter, fit a
    polynomial of each strategy's NMB against that parameter's sampled
    values, then measure how much of the overall EVPI that regression
    surface alone would resolve. A lightweight, dependency-free stand-in
    for a full GAM fit -- good for ranking parameters and, since the
    polynomial degree is now chosen per parameter via cross-validation
    (see _select_degree_by_cv()) instead of a fixed degree=2, robust
    enough for a submission-grade EVPPI rather than ranking alone.

    Pass an explicit `degree` to force a fixed degree instead of
    cross-validating (e.g. to reproduce an older run, or when
    n_iterations is too small for CV to be meaningful); leave it None
    (the default) to auto-select from `candidate_degrees`.
    """

    n_iter = result.n_iterations
    n_params = result.samples.shape[1] if result.samples.ndim == 2 else 0
    if n_iter == 0 or n_params == 0:
        return []
    nmb = willingness_to_pay * result.effects - result.costs
    max_expected_nmb = float(np.max(np.mean(nmb, axis=0)))
    overall_evpi = float(np.mean(np.max(nmb, axis=1))) - max_expected_nmb
    rows: list[EVPPIRow] = []
    for param_index, target_id in enumerate(result.parameter_labels):
        theta = result.samples[:, param_index]
        if np.allclose(theta, theta[0]):
            rows.append(EVPPIRow(target_id=target_id, evppi=0.0, percent_of_evpi=0.0, degree_used=0))
            continue
        theta_matrix = _standardize(theta.reshape(-1, 1))
        degree_used = degree if degree is not None else _select_degree_by_cv(
            theta_matrix, nmb, candidate_degrees
        )
        design = _polynomial_design_matrix(theta_matrix, degree_used)
        coefficients, *_ = np.linalg.lstsq(design, nmb, rcond=None)
        fitted = design @ coefficients
        expected_max_given_theta = float(np.mean(np.max(fitted, axis=1)))
        evppi = max(0.0, expected_max_given_theta - max_expected_nmb)
        rows.append(
            EVPPIRow(
                target_id=target_id,
                evppi=evppi,
                percent_of_evpi=(evppi / overall_evpi) if overall_evpi > 0 else None,
                degree_used=degree_used,
            )
        )
    rows.sort(key=lambda row: row.evppi, reverse=True)
    return rows


@dataclass(frozen=True)
class EVPPIGroupResult:
    target_ids: tuple[str, ...]
    evppi: float
    percent_of_evpi: float | None
    degree_used: int


def evppi_for_group(
    result: PSAResult,
    willingness_to_pay: float,
    target_ids: Sequence[str],
    degree: int | None = None,
    candidate_degrees: Sequence[int] = (1, 2, 3),
) -> EVPPIGroupResult | None:
    """Joint EVPPI for learning several parameters TOGETHER -- the value
    of a study that would resolve uncertainty in every parameter in
    `target_ids` at once, which is not generally the sum of their
    individual EVPPIs (parameters can be correlated in their effect on
    NMB, or their combined effect can be more or less than additive).
    This is what TreeAge's "EVPPI Simulation" reports when several
    parameters are grouped, and the natural multi-parameter counterpart
    to evppi_by_parameter()'s one-at-a-time ranking -- both share the
    same cross-validated multivariate-polynomial-regression fitting path
    (see _polynomial_design_matrix()/_select_degree_by_cv()).

    Returns None if fewer than 1 valid (non-constant) parameter from
    `target_ids` is found in the PSA result, or the result has no
    iterations. A group larger than 3-4 parameters needs proportionally
    more PSA iterations for the polynomial regression to be well
    identified (the number of coefficients grows combinatorially with
    both group size and degree) -- interpret a group EVPPI cautiously
    when the PSA run has few iterations relative to the group size.
    """

    n_iter = result.n_iterations
    if n_iter == 0 or not target_ids:
        return None
    label_to_index = {label: index for index, label in enumerate(result.parameter_labels)}
    indices = [label_to_index[t] for t in target_ids if t in label_to_index]
    if not indices:
        return None
    nmb = willingness_to_pay * result.effects - result.costs
    max_expected_nmb = float(np.max(np.mean(nmb, axis=0)))
    overall_evpi = float(np.mean(np.max(nmb, axis=1))) - max_expected_nmb
    theta_matrix = result.samples[:, indices]
    non_constant = [
        i for i in range(theta_matrix.shape[1]) if not np.allclose(theta_matrix[:, i], theta_matrix[0, i])
    ]
    if not non_constant:
        return EVPPIGroupResult(target_ids=tuple(target_ids), evppi=0.0, percent_of_evpi=0.0, degree_used=0)
    theta_matrix = _standardize(theta_matrix[:, non_constant])
    degree_used = degree if degree is not None else _select_degree_by_cv(
        theta_matrix, nmb, candidate_degrees
    )
    design = _polynomial_design_matrix(theta_matrix, degree_used)
    coefficients, *_ = np.linalg.lstsq(design, nmb, rcond=None)
    fitted = design @ coefficients
    expected_max_given_theta = float(np.mean(np.max(fitted, axis=1)))
    evppi = max(0.0, expected_max_given_theta - max_expected_nmb)
    return EVPPIGroupResult(
        target_ids=tuple(target_ids),
        evppi=evppi,
        percent_of_evpi=(evppi / overall_evpi) if overall_evpi > 0 else None,
        degree_used=degree_used,
    )
