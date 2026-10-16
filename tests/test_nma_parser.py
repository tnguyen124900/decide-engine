import math

import pytest

from heor_tree_engine.nma_models import NMASpec
from heor_tree_engine.nma_parser import (
    normalize_name,
    resolve_nma,
    se_from_ci,
    studies_from_arm_rows,
    studies_from_contrast_rows,
)


def test_aliases_and_spelling_variants_map_to_one_treatment() -> None:
    model = resolve_nma({
        "spec": {"reference": "placebo"},
        "treatments": [{"name": "Placebo", "aliases": ["PBO"]}, {"name": "Drug A", "class": "SGLT2i", "color": "#2f6fd6"}],
        "studies": [
            {"id": "S1", "arms": [{"treatment": "PBO", "events": 1, "n": 10}, {"treatment": "drug-a", "events": 2, "n": 10}]},
            {"id": "S2", "arms": [{"treatment": "placebo ", "events": 1, "n": 10}, {"treatment": "DRUG_A", "events": 2, "n": 10}]},
        ],
    })

    assert [study.treatments for study in model.studies] == [["Placebo", "Drug A"], ["Placebo", "Drug A"]]
    assert model.reference == "Placebo"
    assert model.treatments[1].treatment_class == "SGLT2i"
    assert normalize_name(" Drug - A ") == normalize_name("drug_a")


def test_treatment_list_is_built_from_data_when_omitted() -> None:
    model = resolve_nma({"studies": [
        {"id": "S1", "arms": [{"treatment": "B", "events": 1, "n": 10}, {"treatment": "A", "events": 2, "n": 10}]},
        {"id": "S2", "arms": [{"treatment": "C", "events": 1, "n": 10}, {"treatment": "b", "events": 2, "n": 10}]},
    ]})

    assert model.treatment_names == ["B", "A", "C"]
    assert model.reference == "B"
    assert model.studies[1].treatments == ["C", "B"]


def test_contrast_from_point_estimate_and_ci_on_ratio_scale() -> None:
    model = resolve_nma({
        "spec": {"outcome": "contrast", "measure": "HR"},
        "studies": [{"id": "S1", "contrasts": [{"treatment": "Z", "versus": "P", "effect": 0.70, "lower": 0.55, "upper": 0.89}]}],
    })
    contrast = model.studies[0].contrasts[0]

    assert contrast.estimate == pytest.approx(math.log(0.70))
    assert contrast.se == pytest.approx((math.log(0.89) - math.log(0.55)) / (2 * 1.959963984540054))


def test_se_from_ci_on_difference_scale() -> None:
    assert se_from_ci(-2.0, 2.0, ratio=False) == pytest.approx(4 / (2 * 1.959963984540054))
    with pytest.raises(ValueError):
        se_from_ci(0.0, 1.2, ratio=True)


def test_arm_rows_are_grouped_into_studies_in_order() -> None:
    studies = studies_from_arm_rows([
        {"study": "S2", "treatment": "P", "events": "3", "n": "50"},
        {"study": "S1", "treatment": "P", "events": 1, "n": 20},
        {"study": "S2", "treatment": "A", "events": 5, "n": 49, "source": "Table 2"},
    ])

    assert [study.id for study in studies] == ["S2", "S1"]
    assert studies[0].arms[1].events == 5.0 and studies[0].arms[1].source == "Table 2"
    assert studies[0].arms[0].n == 50.0


def test_contrast_rows_accept_comparator_column_and_baseline_se() -> None:
    studies = studies_from_contrast_rows([
        {"study": "S1", "treatment": "A", "comparator": "P", "estimate": -0.3, "se": 0.1, "baseline_se": 0.05},
        {"study": "S1", "treatment": "B", "comparator": "P", "estimate": -0.1, "se": 0.12},
    ], NMASpec(outcome="contrast", measure="HR"))

    assert len(studies) == 1
    assert studies[0].treatments == ["P", "A", "B"]
    assert studies[0].baseline_se == 0.05


def test_bad_numbers_and_missing_fields_raise() -> None:
    with pytest.raises(ValueError, match="events"):
        resolve_nma({"studies": [{"id": "S1", "arms": [{"treatment": "P", "events": "many", "n": 10}]}]})
    with pytest.raises(ValueError, match="studies"):
        resolve_nma({})
    with pytest.raises(ValueError, match="'se' or both"):
        resolve_nma({"spec": {"outcome": "contrast", "measure": "HR"},
                     "studies": [{"id": "S1", "contrasts": [{"treatment": "A", "versus": "P", "effect": 0.8}]}]})
