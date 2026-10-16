"""Build a PartitionedSurvivalModel from its flat JSON representation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .psm_models import PartitionedSurvivalModel, PSMState, SurvivalCurve


def _numeric_ref(value: Any) -> float | str:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError("Numeric references must be numbers or parameter names")
    return float(value) if isinstance(value, (int, float)) else value


def _parse_curve(raw: Mapping[str, Any]) -> SurvivalCurve:
    if "distribution" not in raw:
        raise ValueError("A survival curve requires a 'distribution'")
    return SurvivalCurve(
        distribution=str(raw["distribution"]),
        params={str(key): _numeric_ref(value) for key, value in raw.get("params", {}).items()},
    )


def _parse_state(raw: Mapping[str, Any]) -> PSMState:
    if "role" not in raw:
        raise ValueError("A PSM state requires a 'role'")
    return PSMState(
        role=str(raw["role"]),
        name=str(raw.get("name", raw["role"])),
        cost=_numeric_ref(raw.get("cost", 0.0)),
        effect=_numeric_ref(raw.get("effect", 0.0)),
    )


def resolve_psm(document: Mapping[str, Any]) -> PartitionedSurvivalModel:
    """Parse a flat PSM JSON document into a PartitionedSurvivalModel."""

    if "pfs_curve" not in document or "os_curve" not in document:
        raise ValueError("PSM document requires 'pfs_curve' and 'os_curve'")
    return PartitionedSurvivalModel(
        id=str(document.get("id", "psm")),
        name=str(document.get("name", "Partitioned survival model")),
        pfs_curve=_parse_curve(document["pfs_curve"]),
        os_curve=_parse_curve(document["os_curve"]),
        states=[_parse_state(raw) for raw in document.get("states", [])],
        cycles=int(document.get("cycles", 10)),
        half_cycle_correction=bool(document.get("half_cycle_correction", True)),
    )
