"""PRISMA-NMA checklist (nma_prisma): 32 items, editor suggestions, author
overrides and the report table."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import nma_prisma  # noqa: E402


def _auto(**kw):
    base = dict(has_characteristics=False, has_rob=False, has_inconsistency=False, has_cinema=False,
                has_funnel=False, bayes=False, has_absolute=False)
    base.update(kw)
    return nma_prisma.auto_status(**base)


def test_checklist_has_27_prisma_items_and_5_nma_items() -> None:
    ids = [item[0] for item in nma_prisma.ITEMS]
    assert len(ids) == 32 and len(set(ids)) == 32
    assert [i for i in ids if i.startswith("S")] == ["S1", "S2", "S3", "S4", "S5"]
    assert [i for i in ids if not i.startswith("S")] == [str(n) for n in range(1, 28)]
    assert all(item[4] == item[0] for item in nma_prisma.ITEMS if item[0].startswith("S"))


def test_suggestions_follow_what_was_run() -> None:
    bare = _auto()
    assert bare["S3"][0] == "Reported" and bare["S4"][0] == "Reported"
    assert bare["S2"][0] == "To do" and bare["S5"][0] == "To do"
    assert bare["1"] == ("To do", nma_prisma.MANUSCRIPT) and bare["27"][0] == "To do"
    full = _auto(has_characteristics=True, has_rob=True, has_inconsistency=True, has_cinema=True, bayes=True,
                 has_absolute=True)
    assert full["S2"][0] == full["S5"][0] == "Reported" and "UME" in full["S5"][1]
    assert full["18"][0] == "Reported" and full["19"][0] == "Reported"
    assert "SUCRA" in full["13"][1] and "P-score" in bare["13"][1]
    assert set(full) == {item[0] for item in nma_prisma.ITEMS}


def test_author_entries_override_and_round_trip() -> None:
    auto = _auto()
    document = {"prisma": {"1": {"status": "Reported", "note": "Title page"}, "S2": {"note": "Methods 2.4"}}}
    frame = nma_prisma.checklist_frame(document, auto)
    assert len(frame) == 32
    row = frame.set_index("Item")
    assert row.loc["1", "Status"] == "Reported" and row.loc["1", "Where reported / notes"] == "Title page"
    assert row.loc["S2", "Status"] == "To do"  # note only, status stays the suggestion
    counts = nma_prisma.summary(frame)
    assert counts["Reported"] >= 1 and sum(counts.values()) == 32

    edited = frame.copy()
    edited.loc[edited["Item"] == "27", "Status"] = "Not applicable"
    edited.loc[edited["Item"] == "1", "Status"] = "To do"  # back to the suggestion -> no override kept
    edited.loc[edited["Item"] == "1", "Where reported / notes"] = ""
    assert nma_prisma.apply_edits(document, edited, auto) is True
    assert document["prisma"] == {"27": {"status": "Not applicable"}, "S2": {"note": "Methods 2.4"}}
    assert nma_prisma.apply_edits(document, edited, auto) is False  # idempotent

    report = nma_prisma.report_frame(nma_prisma.checklist_frame(document, auto))
    assert list(report.columns) == ["Item", "Section", "Topic", "Checklist item", "Status", "Where reported"]
    where = report.set_index("Item")["Where reported"]
    assert where["S2"] == "Methods 2.4" and where["S3"].startswith("Report:") and where["1"] == nma_prisma.MANUSCRIPT
