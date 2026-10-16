"""GRADE-style Summary of Findings table (nma_sof)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import nma_editor_support as sup  # noqa: E402
import nma_sof  # noqa: E402
from heor_tree_engine.nma_cinema import run_cinema  # noqa: E402
from heor_tree_engine.nma_freq import run_nma_frequentist  # noqa: E402
from heor_tree_engine.nma_network import summarize_network  # noqa: E402
from heor_tree_engine.nma_parser import resolve_nma  # noqa: E402


def _setup():
    document = sup.default_document()
    document["absolute"] = {"baseline_risk": 0.2}
    model = resolve_nma(document)
    result = run_nma_frequentist(model)
    return document, model, summarize_network(model), result


def test_layouts_and_columns() -> None:
    document, model, summary, result = _setup()
    ref = nma_sof.sof_frame(document, model, summary, result.random, "OR", "reference")
    assert list(ref.columns) == nma_sof.COLUMNS
    assert list(ref["Comparison"]) == [f"{t} vs Placebo" for t in ("Drug A", "Drug B", "Drug C", "Drug D")]
    assert ref["Assumed risk /1000"].iloc[0] == "200 (Placebo)"  # user's baseline risk for the reference
    assert ref["Certainty (CINeMA)"].iloc[0] == "not rated" and ref["Ranking"].iloc[0].startswith("P-score")
    assert ref["Relative effect"].iloc[0].startswith("OR ")
    assert int(ref["Studies (direct)"].sum()) > 0
    allp = nma_sof.sof_frame(document, model, summary, result.random, "OR", "all")
    assert len(allp) == 10  # 5 treatments -> 10 pairs
    comp = nma_sof.sof_frame(document, model, summary, result.random, "OR", "comparator", "Drug A")
    assert len(comp) == 4 and all(c.endswith("vs Drug A") for c in comp["Comparison"])
    assert comp["Assumed risk /1000"].iloc[0].endswith("(Drug A)")  # observed risk of Drug A arms
    assert nma_sof.observed_risk(model, "Drug A") is not None


def test_certainty_from_cinema_and_non_binary_measure() -> None:
    document, model, summary, result = _setup()
    cinema = run_cinema(model, result)
    frame = nma_sof.sof_frame(document, model, summary, result.random, "OR", "reference", cinema=cinema,
                              ranking_label="SUCRA", interval_label="95% CrI")
    assert set(frame["Certainty (CINeMA)"]) <= {"High", "Moderate", "Low", "Very low"}
    assert frame["Ranking"].iloc[0].startswith("SUCRA")
    assert frame.attrs["interval"] == "95% CrI"
    notes = nma_sof.footnotes("OR", "95% CrI", True, True)
    assert any("SUCRA" in n for n in notes) and any("Altman" in n for n in notes)
    # MD: no absolute columns
    document["spec"].update({"outcome": "continuous", "measure": "MD"})
    for study in document["studies"]:
        for arm in study["arms"]:
            arm.update({"mean": 10 + arm["events"] / 10, "sd": 4.0})
    model = resolve_nma(document)
    result = run_nma_frequentist(model)
    frame = nma_sof.sof_frame(document, model, summarize_network(model), result.random, "MD", "reference")
    assert "NNT" not in frame.columns and frame["Relative effect"].iloc[0].startswith("MD ")
