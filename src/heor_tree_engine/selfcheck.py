"""Shared plumbing for the validation self-checks (AdViSHE internal validity).  A check runs when a report is built; a failure raises
`SelfCheckFailure` instead of returning a report, because an internal-consistency failure means an engine bug,
not a bad model.  Tolerances: relative 1e-6 / absolute 1e-9."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

DEFAULT_REL_TOL = 1e-6
DEFAULT_ABS_TOL = 1e-9


@dataclass
class SelfCheckResult:
    name: str
    passed: bool
    detail: str
    applicable: bool = True  # False = "Not applicable to this model" (never a failure)
    # "error": an internal-consistency failure (engine bug) that blocks the report.
    # "warning": a model-quality finding (e.g. crossing survival curves) that is reported but never blocks.
    severity: str = "error"


class SelfCheckFailure(Exception):
    """An internal-validity self-check failed: an engine bug, not a bad model.  No report is returned."""

    def __init__(self, failures: Sequence[SelfCheckResult]) -> None:
        self.failures = list(failures)
        super().__init__("Validation self-check failed: " + "; ".join(f"{c.name} ({c.detail})" for c in failures))


def values_close(a: float, b: float, rel: float = DEFAULT_REL_TOL, abs_: float = DEFAULT_ABS_TOL) -> bool:
    return abs(a - b) <= max(abs_, rel * max(abs(a), abs(b)))


def raise_on_failure(checks: Sequence[SelfCheckResult]) -> None:
    failed = [c for c in checks if c.applicable and not c.passed and c.severity == "error"]
    if failed:
        raise SelfCheckFailure(failed)
