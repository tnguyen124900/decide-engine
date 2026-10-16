"""Tests for nma_import (RevMan .rm5/CSV and Covidence CSV import)."""

from __future__ import annotations

import pytest

from heor_tree_engine.nma_import import (
    NMAImportError,
    detect_import_format,
    parse_covidence_csv,
    parse_revman_csv,
    parse_revman_rm5,
    studies_to_document_rows,
)

# --- RevMan .rm5 -----------------------------------------------------------

RM5_BINARY = b"""<?xml version="1.0" encoding="UTF-8"?>
<RM5>
  <ANALYSES_AND_DATA>
    <DICH_DATA STUDY_ID="Smith 2020" EVENTS_1="12" N_1="100" EVENTS_2="25" N_2="98"
               GROUP_LABEL_1="DrugA" GROUP_LABEL_2="Placebo"/>
    <DICH_DATA STUDY_ID="Jones 2021" EVENTS_1="8" N_1="60" EVENTS_2="15" N_2="61"
               GROUP_LABEL_1="DrugA" GROUP_LABEL_2="Placebo"/>
    <DICH_DATA STUDY_ID="Multi 2022" EVENTS_1="5" N_1="50" EVENTS_2="10" N_2="50"
               GROUP_LABEL_1="DrugA" GROUP_LABEL_2="Placebo"/>
    <DICH_DATA STUDY_ID="Multi 2022" EVENTS_1="7" N_1="52" EVENTS_2="10" N_2="50"
               GROUP_LABEL_1="DrugB" GROUP_LABEL_2="Placebo"/>
  </ANALYSES_AND_DATA>
</RM5>
"""

RM5_CONTINUOUS = b"""<?xml version="1.0" encoding="UTF-8"?>
<RM5>
  <ANALYSES_AND_DATA>
    <CONT_DATA STUDY_ID="Lee 2019" MEAN_1="5.2" SD_1="1.1" N_1="40"
               MEAN_2="6.8" SD_2="1.4" N_2="42"
               GROUP_LABEL_1="DrugA" GROUP_LABEL_2="Placebo"/>
  </ANALYSES_AND_DATA>
</RM5>
"""


def test_rm5_binary_two_studies() -> None:
    studies = {s["id"]: s for s in parse_revman_rm5(RM5_BINARY)}
    assert set(studies) == {"Smith 2020", "Jones 2021", "Multi 2022"}
    smith = studies["Smith 2020"]
    assert len(smith["arms"]) == 2
    by_arm = {a["treatment"]: a for a in smith["arms"]}
    assert by_arm["DrugA"]["events"] == 12.0
    assert by_arm["DrugA"]["n"] == 100.0
    assert by_arm["Placebo"]["events"] == 25.0


def test_rm5_multi_arm_regrouped() -> None:
    """A 3-arm trial stored as two pairwise rows becomes ONE study with
    three arms (not two pseudo-2-arm studies)."""
    studies = {s["id"]: s for s in parse_revman_rm5(RM5_BINARY)}
    multi = studies["Multi 2022"]
    treatments = {a["treatment"] for a in multi["arms"]}
    assert treatments == {"DrugA", "DrugB", "Placebo"}
    placebo = next(a for a in multi["arms"] if a["treatment"] == "Placebo")
    assert placebo["n"] == 50.0


def test_rm5_continuous() -> None:
    studies = parse_revman_rm5(RM5_CONTINUOUS)
    arm = studies[0]["arms"][0]
    assert arm["mean"] == 5.2
    assert arm["sd"] == 1.1
    assert arm["n"] == 40.0


def test_rm5_rejects_garbage() -> None:
    with pytest.raises(NMAImportError):
        parse_revman_rm5(b"not xml at all")


# --- RevMan CSV ------------------------------------------------------------

REV_CSV = b"""Study,Experimental events,Experimental n,Control events,Control n
Smith 2020,12,100,25,98
Jones 2021,8,60,15,61
"""


def test_revman_csv_binary() -> None:
    studies = parse_revman_csv(REV_CSV)
    assert len(studies) == 2
    assert studies[0]["id"] == "Smith 2020"
    by_arm = {a["treatment"]: a for a in studies[0]["arms"]}
    assert by_arm["Experimental"]["events"] == 12.0
    assert by_arm["Control"]["n"] == 98.0


# --- Covidence CSV ---------------------------------------------------------

COV_CSV = b"""Study,Intervention,Events,N
Smith 2020,DrugA,12,100
Smith 2020,Placebo,25,98
Jones 2021,DrugA,8,60
Jones 2021,Placebo,15,61
"""

COV_CSV_CONT = b"""Study,Intervention,Mean,SD,N
Lee 2019,DrugA,5.2,1.1,40
Lee 2019,Placebo,6.8,1.4,42
"""


def test_covidence_binary() -> None:
    studies = parse_covidence_csv(COV_CSV)
    assert len(studies) == 2
    smith = studies[0]
    assert smith["id"] == "Smith 2020"
    assert {a["treatment"] for a in smith["arms"]} == {"DrugA", "Placebo"}


def test_covidence_continuous() -> None:
    studies = parse_covidence_csv(COV_CSV_CONT)
    arm = studies[0]["arms"][0]
    assert arm["mean"] == 5.2 and arm["sd"] == 1.1


def test_covidence_needs_two_arms() -> None:
    one_arm = b"Study,Intervention,Events,N\nSolo 2020,DrugA,5,50\n"
    with pytest.raises(NMAImportError):
        parse_covidence_csv(one_arm)


# --- Format detection + outcome summary ------------------------------------


def test_detect_rm5_by_extension() -> None:
    assert detect_import_format("review.rm5", b"anything") == "rm5"


def test_detect_covidence_by_name() -> None:
    assert detect_import_format("covidence_extraction.csv", b"a,b,c") == "covidence"


def test_detect_rejects_unknown() -> None:
    with pytest.raises(NMAImportError):
        detect_import_format("mystery.txt", b"hello world\nfoo bar")


def test_outcome_summary_binary() -> None:
    studies = parse_covidence_csv(COV_CSV)
    assert studies_to_document_rows(studies)["outcome"] == "binary"


def test_outcome_summary_continuous() -> None:
    studies = parse_covidence_csv(COV_CSV_CONT)
    assert studies_to_document_rows(studies)["outcome"] == "continuous"
