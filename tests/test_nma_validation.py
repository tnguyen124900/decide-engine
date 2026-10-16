from heor_tree_engine.nma_parser import resolve_nma
from heor_tree_engine.nma_validation import validate_nma


def _arm(treatment: str, events: float = 5, n: float = 50) -> dict:
    return {"treatment": treatment, "events": events, "n": n}


def _document(studies: list, **spec) -> dict:
    return {"spec": {"outcome": "binary", "reference": "P", **spec}, "studies": studies}


def _codes(document: dict) -> set[str]:
    return {error.code for error in validate_nma(resolve_nma(document))}


def test_valid_network_has_no_errors() -> None:
    assert _codes(_document([
        {"id": "S1", "arms": [_arm("P"), _arm("A")]},
        {"id": "S2", "arms": [_arm("A"), _arm("B")]},
    ])) == set()


def test_disconnected_network_is_flagged_with_groups() -> None:
    model = resolve_nma(_document([
        {"id": "S1", "arms": [_arm("P"), _arm("A")]},
        {"id": "S2", "arms": [_arm("B"), _arm("C")]},
    ]))
    errors = [error for error in validate_nma(model) if error.code == "disconnected_network"]

    assert len(errors) == 1
    assert "P + A" in errors[0].message and "B + C" in errors[0].message


def test_arm_value_problems_are_flagged_per_study() -> None:
    model = resolve_nma(_document([
        {"id": "S1", "arms": [_arm("P", events=60, n=50), _arm("A", events=-1)]},
        {"id": "S2", "arms": [{"treatment": "P", "n": 40}, _arm("A", n=0)]},
    ]))
    errors = validate_nma(model)
    codes = {(error.node_id, error.code) for error in errors}

    assert ("S1", "events_exceed_n") in codes
    assert ("S1", "invalid_events") in codes
    assert ("S2", "missing_arm_value") in codes
    assert ("S2", "invalid_n") in codes


def test_unknown_treatment_gets_a_suggestion() -> None:
    model = resolve_nma({
        "spec": {"outcome": "binary", "reference": "Placebo"},
        "treatments": ["Placebo", "Dapagliflozin"],
        "studies": [{"id": "S1", "arms": [_arm("Placebo"), _arm("Dapaglifozin")]}],
    })
    messages = [error.message for error in validate_nma(model) if error.code == "unknown_treatment"]

    assert messages and "did you mean 'Dapagliflozin'" in messages[0]


def test_study_structure_problems() -> None:
    codes = _codes(_document([
        {"id": "S1", "arms": [_arm("P"), _arm("P")]},
        {"id": "S1", "arms": [_arm("P"), _arm("A")]},
        {"id": "S3", "arms": [_arm("P")], "contrasts": [{"treatment": "A", "versus": "P", "estimate": 0.1, "se": 0.1}]},
    ]))
    assert {"duplicate_arm", "too_few_arms", "duplicate_study", "mixed_study_data"} <= codes


def test_multi_arm_contrast_study_needs_shared_comparator_and_baseline_se() -> None:
    codes = _codes({
        "spec": {"outcome": "contrast", "measure": "HR", "reference": "P"},
        "studies": [{"id": "S1", "contrasts": [
            {"treatment": "A", "versus": "P", "estimate": -0.2, "se": 0.1},
            {"treatment": "B", "versus": "A", "estimate": -0.1, "se": 0.1}]}],
    })
    assert {"multi_arm_comparator", "missing_baseline_se"} <= codes


def test_spec_problems() -> None:
    assert "unknown_outcome" in _codes(_document([{"id": "S1", "arms": [_arm("P"), _arm("A")]}], outcome="survival"))
    assert "unsupported_measure" in _codes(_document([{"id": "S1", "arms": [_arm("P"), _arm("A")]}], measure="SMD"))
    assert "unknown_reference" in _codes(_document([{"id": "S1", "arms": [_arm("X"), _arm("A")]}]))
    assert "no_studies" in _codes(_document([]))


def test_listed_treatment_without_data_is_flagged() -> None:
    codes = _codes({
        "spec": {"outcome": "binary", "reference": "P"},
        "treatments": ["P", "A", "Z"],
        "studies": [{"id": "S1", "arms": [_arm("P"), _arm("A")]}],
    })
    assert "unused_treatment" in codes
