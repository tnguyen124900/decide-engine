import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import nma_editor_support as sup  # noqa: E402
from heor_tree_engine.nma_absolute import (absolute_effects, observed_baseline_risk,  # noqa: E402
                                           risk_with_treatment)
from heor_tree_engine.nma_freq import run_nma_frequentist  # noqa: E402
from heor_tree_engine.nma_models import RelativeEffect  # noqa: E402
from heor_tree_engine.nma_parser import resolve_nma  # noqa: E402


class _Fit:
    def __init__(self, *effects):
        self.effects = list(effects)


def _effect(value, low, high):
    return RelativeEffect("A", "P", math.log(value), 0.1, math.log(low), math.log(high), value, low, high)


def test_risk_conversions() -> None:
    assert risk_with_treatment(0.2, 0.5, "OR") == pytest.approx(0.1 / 0.9)
    assert risk_with_treatment(0.2, 0.5, "RR") == pytest.approx(0.1)
    assert risk_with_treatment(0.6, 2.0, "RR") == 1.0
    assert risk_with_treatment(0.2, 0.5, "HR") == pytest.approx(1 - 0.8 ** 0.5)
    assert risk_with_treatment(0.2, -0.05, "RD") == pytest.approx(0.15)


def test_absolute_effect_and_nnt_with_ci() -> None:
    # RR 0.5 (0.4 to 0.8) with 20% baseline risk: 10% vs 20%, NNT 10 (8 to 25).
    row = absolute_effects(_Fit(_effect(0.5, 0.4, 0.8)), "RR", 0.2, "good")[0]
    assert row.risk == pytest.approx(0.1) and row.difference == pytest.approx(-0.1)
    assert (row.difference_lower, row.difference_upper) == pytest.approx((-0.12, -0.04))
    assert row.nnt == pytest.approx(10) and row.nnt_kind == "NNTB"
    assert row.nnt_text == "NNTB 10 (9 to 25)"

    # Higher is better (e.g. response): a lower risk is harm.
    assert absolute_effects(_Fit(_effect(0.5, 0.4, 0.8)), "RR", 0.2, "bad")[0].nnt_kind == "NNTH"


def test_nnt_interval_through_infinity_when_ci_includes_no_effect() -> None:
    row = absolute_effects(_Fit(_effect(0.8, 0.5, 1.25)), "RR", 0.2, "good")[0]
    assert row.difference_lower < 0 < row.difference_upper
    assert row.nnt_text == "NNTB 25 (NNTB 10 to ∞ to NNTH 20)"


def test_observed_baseline_risk_and_network_rows() -> None:
    document = sup.default_document()
    model = resolve_nma(document)
    p0 = observed_baseline_risk(model)
    placebo = [arm for study in document["studies"] for arm in study["arms"] if arm["treatment"] == "Placebo"]
    assert p0 == pytest.approx(sum(a["events"] for a in placebo) / sum(a["n"] for a in placebo))

    rows = absolute_effects(run_nma_frequentist(model).random, "OR", p0, "good")
    assert [row.treatment for row in rows] == ["Drug A", "Drug B", "Drug C", "Drug D"]
    assert all(row.risk_lower <= row.risk <= row.risk_upper for row in rows)
    assert absolute_effects(run_nma_frequentist(model).random, "MD", 0.2) == []
    with pytest.raises(ValueError):
        absolute_effects(run_nma_frequentist(model).random, "OR", 1.2)
