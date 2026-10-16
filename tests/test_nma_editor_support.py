import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # app-level modules

import nma_editor_support as sup  # noqa: E402
from heor_tree_engine.nma_freq import run_nma_frequentist  # noqa: E402
from heor_tree_engine.nma_network import summarize_network  # noqa: E402
from heor_tree_engine.nma_parser import resolve_nma  # noqa: E402
from heor_tree_engine.nma_validation import validate_nma  # noqa: E402


def test_example_document_is_valid_and_runs() -> None:
    model = resolve_nma(sup.default_document())

    assert validate_nma(model) == []
    result = run_nma_frequentist(model)
    assert result.errors == []
    assert len(result.random.effects) == 4


def test_tables_round_trip_to_the_same_studies() -> None:
    document = sup.default_document()
    document["studies"].append({"id": "Study 10", "baseline_se": 0.05, "contrasts": [
        {"treatment": "Drug A", "versus": "Placebo", "estimate": -0.3, "se": 0.12},
        {"treatment": "Drug D", "versus": "Placebo", "effect": 0.8, "lower": 0.6, "upper": 1.1},
    ]})

    studies = sup.studies_from_frames(sup.arms_frame(document), sup.contrasts_frame(document))

    assert studies == [
        {**study, "arms": [{**arm, "events": float(arm["events"]), "n": float(arm["n"])} for arm in study["arms"]]}
        if "arms" in study else study
        for study in document["studies"]
    ]
    assert sup.treatments_from_frame(sup.treatments_frame(document)) == document["treatments"]


def test_half_typed_rows_are_ignored() -> None:
    arms = pd.DataFrame([
        {"study": "S1", "treatment": "P", "events": 3, "n": 30},
        {"study": "S1", "treatment": None, "events": 5, "n": 30},
        {"study": "", "treatment": "A", "events": 5, "n": 30},
    ], columns=sup.ARM_COLUMNS)
    studies = sup.studies_from_frames(arms, pd.DataFrame(columns=sup.CONTRAST_COLUMNS))

    assert studies == [{"id": "S1", "arms": [{"treatment": "P", "events": 3.0, "n": 30.0}]}]


def test_csv_import_detects_arm_or_contrast_layout() -> None:
    kind, frame = sup.frames_from_csv(pd.DataFrame({"Study": ["S1"], "Treatment": ["A"], "Events": [3], "N": [30]}))
    assert kind == "arms" and frame.loc[0, "events"] == 3.0 and list(frame.columns) == sup.ARM_COLUMNS

    kind, frame = sup.frames_from_csv(pd.DataFrame({"study": ["S1"], "treatment": ["A"], "comparator": ["P"],
                                                    "estimate": [-0.2], "se": [0.1]}))
    assert kind == "contrasts" and frame.loc[0, "versus"] == "P"


def test_node_colours_by_class_single_and_custom() -> None:
    model = resolve_nma(sup.default_document())

    by_class = sup.node_colors(model, "class")
    assert by_class["Placebo"] == sup.UNCLASSIFIED_COLOR  # "Control" class is grey
    assert by_class["Drug A"] == by_class["Drug B"] != by_class["Drug C"]
    assert set(sup.node_colors(model, "single").values()) == {sup.SINGLE_COLOR}
    assert sup.node_colors(model, "custom")["Drug B"] == "#0891b2"
    assert [label for label, _, _ in sup.class_legend(model)] == ["Control", "Class X", "Class Y"]


def test_network_payload_marks_selected_edge_and_scales_nodes() -> None:
    model = resolve_nma(sup.default_document())
    summary = summarize_network(model)
    payload = sup.network_payload(model, summary, "class", "patients", ("Placebo", "Drug A"))

    selected = [edge for edge in payload["edges"] if edge["selected"]]
    assert len(selected) == 1 and {selected[0]["a"], selected[0]["b"]} == {"Placebo", "Drug A"}
    radii = {node["id"]: node["r"] for node in payload["nodes"]}
    assert radii["Placebo"] == max(radii.values())
    assert sorted(sup.indirect_paths(summary, "Placebo", "Drug A")) == ["Drug B", "Drug C"]


def test_league_frame_lower_triangle() -> None:
    result = run_nma_frequentist(resolve_nma(sup.default_document()))
    order = sup.ranking_order(result.random)
    league = sup.league_frame(result.random, order)

    assert list(league.index) == order and list(league.columns) == order
    assert league.iloc[0, 0] == order[0]
    effect = result.random.league[(order[0], order[1])]
    assert league.iloc[1, 0] == sup.format_effect(effect.natural, effect.natural_lower, effect.natural_upper)
    assert league.iloc[0, 1] == ""


def test_document_hash_ignores_displayed_effects_choice() -> None:
    document = sup.default_document()
    other = sup.default_document()
    other["spec"]["effects"] = "fixed"

    assert sup.document_hash(document) == sup.document_hash(other)
    other["studies"][0]["arms"][0]["events"] = 19
    assert sup.document_hash(document) != sup.document_hash(other)


def test_node_split_frame_flags_small_p_values() -> None:
    from heor_tree_engine.nma_inconsistency import run_inconsistency

    document = sup.default_document()
    # Make Study 07 (A-B-C) conflict with the rest.
    document["studies"][6]["arms"] = [{"treatment": "Drug A", "events": 40, "n": 90},
                                      {"treatment": "Drug B", "events": 5, "n": 92},
                                      {"treatment": "Drug C", "events": 17, "n": 91}]
    incons = run_inconsistency(resolve_nma(document))
    frame = sup.node_split_frame(incons.splits_fixed, "OR")

    assert list(frame.columns)[:2] == ["Comparison", "Direct studies"]
    assert "Ratio (indirect / direct)" in frame.columns
    assert frame["_flag"].any()
    assert len(frame) == len(incons.splits_fixed) > 0


def test_html_table_has_dark_header_escapes_and_flags_rows() -> None:
    frame = pd.DataFrame([{"Comparison": "A <b>vs</b> P", "p": 0.01}, {"Comparison": "B vs P", "p": 0.4}])
    html = sup.html_table(frame, flags=[True, False])

    assert f"background:{sup.TABLE_HEADER_BG};color:{sup.TABLE_HEADER_FG}" in html
    assert "&lt;b&gt;vs&lt;/b&gt;" in html and "<b>vs</b>" not in html
    assert html.count(sup.TABLE_FLAG_BG) == 1
    assert "text-align:right" in html  # numbers right-aligned


def test_table_rows_round_trip_for_the_editable_component() -> None:
    document = sup.default_document()
    arms = sup.arms_frame(document)
    rows = sup.frame_to_rows(arms)

    assert rows[0] == {"study": "Study 01", "treatment": "Placebo", "events": 18.0, "n": 120.0,
                       "mean": None, "sd": None, "exposure": None, "source": None}
    rows[0]["events"] = 5
    rows.append({"study": "Study 10", "treatment": "Drug E", "events": "3", "n": None})
    frame = sup.rows_to_frame(rows, sup.ARM_COLUMNS, sup.NUMERIC_ARM_COLUMNS)

    assert list(frame.columns) == sup.ARM_COLUMNS
    assert frame.loc[0, "events"] == 5.0 and frame.loc[len(frame) - 1, "events"] == 3.0
    studies = sup.studies_from_frames(frame, sup.contrasts_frame(document))
    assert studies[-1] == {"id": "Study 10", "arms": [{"treatment": "Drug E", "events": 3.0}]}
    assert [column["key"] for column in sup.ARM_COLUMN_SPEC] == sup.ARM_COLUMNS
    assert {column["key"]: column["type"] for column in sup.ARM_COLUMN_SPEC}["events"] == "number"


def test_characteristics_rows_rob_normalisation_and_transitivity() -> None:
    document = sup.default_document()
    document["study_characteristics"] = [
        {"study": "Study 01", "year": 2015, "mean_age": 62, "rob": "high risk"},
        {"study": "Study 02", "year": 2019, "mean_age": 66, "rob": "L"},
        {"study": "Old study", "year": 2001},
    ]
    frame = sup.characteristics_frame(document)

    assert list(frame["study"])[:3] == ["Study 01", "Study 02", "Study 03"]
    assert list(frame["study"])[-1] == "Old study"  # kept, even without data rows
    rows = sup.characteristics_from_frame(frame)
    assert rows[0]["rob"] == "High" and rows[1]["rob"] == "Low"
    assert sup.high_rob_studies(rows) == ["Study 01"]
    assert sup.normalize_rob("Unclear") == "Some concerns"

    summary = summarize_network(resolve_nma(document))
    table = sup.transitivity_frame(summary, rows)
    first = table[table["Comparison"] == "Drug A vs Placebo"].iloc[0]
    assert first["Studies"] == 3 and first["Year"] == "2015–2019"
    assert first["Mean age"] == "64.0 (62.0–66.0)"
    assert first["Risk of bias"] == "Low: 1, High: 1"


def test_rank_probability_frame_is_long_format() -> None:
    frame = sup.rank_probability_frame({"A": [0.7, 0.3], "B": [0.3, 0.7]}, ["A", "B"])
    assert list(frame.columns) == ["treatment", "rank", "probability"]
    assert len(frame) == 4 and frame["probability"].sum() == 2.0


def test_figure_legend_and_filename_for_png_export() -> None:
    model = resolve_nma(sup.default_document())

    legend = sup.figure_legend(model, "class")
    assert [row["label"] for row in legend] == ["Control", "Class X", "Class Y"]
    assert legend[1]["members"] == "Drug A, Drug B"
    assert [row["label"] for row in sup.figure_legend(model, "custom")][:2] == ["Placebo", "Drug A"]
    assert sup.figure_legend(model, "single") == []
    assert sup.figure_filename("Example network (illustrative data)", "3D") == \
        "example-network-illustrative-data-network-3d.png"
    assert sup.figure_filename("", "2D") == "nma-network-2d.png"


def test_several_analyses_round_trip_through_one_file() -> None:
    mortality = sup.ensure_analysis_id({**sup.default_document(), "title": "Mortality",
                                        "study_characteristics": [{"study": "Study 01", "rob": "Low"}]})
    hospital = sup.new_analysis(mortality, "Hospitalisation", copy_studies=False)
    adverse = sup.new_analysis(mortality, "Adverse events", copy_studies=True)

    assert hospital["studies"] == [] and hospital["treatments"] == mortality["treatments"]
    assert adverse["studies"] == mortality["studies"] and adverse["analysis_id"] != mortality["analysis_id"]
    hospital["spec"]["outcome"] = "rate"

    data = sup.bundle_documents([mortality, hospital, adverse], active=1)
    assert data["format"] == "decide-nma" and data["active"] == 1
    assert [a["title"] for a in data["analyses"]] == ["Mortality", "Hospitalisation", "Adverse events"]
    assert "treatments" not in data["analyses"][0]  # shared once at the top

    documents, active = sup.documents_from_file(json.loads(json.dumps(data)))
    assert active == 1 and len(documents) == 3
    assert documents[1]["spec"]["outcome"] == "rate"
    assert documents[2]["study_characteristics"] == [{"study": "Study 01", "rob": "Low"}]
    assert documents[0]["name"] == mortality["name"]


def test_single_analysis_files_stay_in_the_old_format() -> None:
    document = sup.default_document()
    assert sup.bundle_documents([document]) is document
    documents, active = sup.documents_from_file(sup.default_document())
    assert active == 0 and len(documents) == 1 and documents[0]["analysis_id"]
    assert sup.analysis_title({}, 2) == "Analysis 3"


def test_share_common_copies_treatments_and_characteristics_only() -> None:
    source = {"name": "Net", "treatments": [{"name": "A"}], "study_characteristics": [{"study": "s"}],
              "studies": [{"id": "s"}], "spec": {"measure": "OR"}}
    target = {"title": "Other", "studies": [], "spec": {"measure": "HR"}}
    sup.share_common(source, target)
    assert target["treatments"] == [{"name": "A"}] and target["spec"] == {"measure": "HR"} and target["studies"] == []


def test_document_hash_ignores_characteristics_and_cinema() -> None:
    document = sup.default_document()
    other = sup.default_document()
    other["study_characteristics"] = [{"study": "Study 01", "rob": "High"}]
    other["cinema"] = {"margin": 1.5}
    other["title"] = "Mortality"
    assert sup.document_hash(document) == sup.document_hash(other)


def test_illustrative_note_only_for_untouched_example_data() -> None:
    document = sup.default_document()
    assert sup.is_example_data(document)
    document["studies"] = sup.studies_from_frames(sup.arms_frame(document), sup.contrasts_frame(document))
    assert sup.is_example_data(document)  # after the editor's table round trip
    document["studies"][0]["arms"][0]["events"] = 19.0
    assert not sup.is_example_data(document)
    assert not sup.is_example_data({"name": "Example network (illustrative data)", "studies": []})


def test_rank_heat_frame_and_html() -> None:
    frame = sup.rank_heat_frame([("Mortality", {"A": 0.9, "P": 0.1}), ("Adverse events", {"A": 0.3, "B": 0.6, "P": 0.7})],
                                ["P", "A", "B"])
    assert list(frame.index) == ["P", "A", "B"] and list(frame.columns) == ["Mortality", "Adverse events"]
    assert frame.loc["A", "Mortality"] == 0.9 and pd.isna(frame.loc["B", "Mortality"])
    html = sup.rank_heat_html(frame)
    assert ">90<" in html and ">–<" in html and sup.TABLE_HEADER_BG in html
    assert sup._heat_color(0.0).startswith("rgba(239,68,68") and sup._heat_color(1.0).startswith("rgba(34,197,94")


def test_characteristic_variables_are_configurable() -> None:
    document = sup.default_document()
    assert [v["key"] for v in sup.characteristic_variables(document)] == ["year", "mean_age", "female_pct", "follow_up"]
    assert sup.characteristic_columns(document) == sup.CHARACTERISTIC_COLUMNS
    document["characteristic_columns"] = [{"key": "year", "label": "Year", "type": "number"},
                                          {"key": "mean_bmi", "label": "Mean BMI", "type": "number"},
                                          {"key": "region", "label": "Region", "type": "text"},
                                          {"key": "study", "label": "bogus", "type": "text"},  # fixed key: ignored
                                          {"key": "region", "label": "dup", "type": "text"}]  # duplicate: ignored
    assert [v["key"] for v in sup.characteristic_variables(document)] == ["year", "mean_bmi", "region"]
    assert sup.characteristic_columns(document) == ["study", "year", "mean_bmi", "region", "rob", "indirectness", "notes"]
    assert sup.numeric_characteristic_columns(document) == ["year", "mean_bmi"]
    assert sup.characteristic_labels(document)["mean_bmi"] == "Mean BMI"
    assert [c["key"] for c in sup.characteristic_column_spec(document)] == sup.characteristic_columns(document)
    assert sup.variable_key("Mean BMI", ["year"]) == "mean_bmi"
    assert sup.variable_key("Mean BMI", ["mean_bmi"]) == "mean_bmi_2"
    assert sup.variable_key("Study", []) == "study_2"

    document["study_characteristics"] = [{"study": "Study 01", "year": 2015, "mean_bmi": 27.5, "region": "EU"},
                                         {"study": "Study 02", "year": 2019, "mean_bmi": 30.0, "region": "US"},
                                         {"study": "Study 03", "region": "EU"}]
    frame = sup.characteristics_frame(document)
    assert list(frame.columns) == sup.characteristic_columns(document)
    assert frame["mean_bmi"].dtype == "float64" and frame.loc[0, "region"] == "EU"
    summary = summarize_network(resolve_nma(document))
    table = sup.transitivity_frame(summary, document["study_characteristics"], sup.characteristic_variables(document))
    assert list(table.columns) == ["Comparison", "Studies", "Year", "Mean BMI", "Region", "Risk of bias"]
    first = table[table["Comparison"] == "Drug A vs Placebo"].iloc[0]
    assert first["Mean BMI"] == "28.8 (27.5–30.0)" and first["Region"] == "EU, US"
    assert sup.document_hash(document) == sup.document_hash({**document, "characteristic_columns": None})
    assert "characteristic_columns" in sup.SHARED_KEYS


def test_transitivity_points_and_categories() -> None:
    document = sup.default_document()
    document["characteristic_columns"] = [
        {"key": "mean_age", "label": "Mean age", "type": "number"},
        {"key": "setting", "label": "Setting", "type": "text"},
    ]
    document["study_characteristics"] = [
        {"study": "Study 01", "mean_age": 62, "setting": "Hospital", "rob": "high risk"},
        {"study": "Study 02", "mean_age": 66, "setting": "Hospital", "rob": "L"},
        {"study": "Study 03", "setting": "Community"},  # no age, no rob
    ]
    summary = summarize_network(resolve_nma(document))
    variables = sup.characteristic_variables(document)

    points = sup.transitivity_points(summary, document["study_characteristics"], variables)
    assert list(points.columns) == ["variable", "label", "comparison", "study", "value", "rob"]
    assert set(points["variable"]) == {"mean_age"}  # text variables are not plotted
    assert set(points["study"]) == {"Study 01", "Study 02"}  # Study 03 has no age
    by_study = {row.study: row for row in points.itertuples()}
    assert by_study["Study 01"].rob == "High" and by_study["Study 02"].rob == "Low"
    assert by_study["Study 01"].comparison == "Drug A vs Placebo" and by_study["Study 01"].value == 62.0
    # a study without a risk-of-bias entry is labelled, not dropped
    document["study_characteristics"][1]["rob"] = ""
    again = sup.transitivity_points(summary, document["study_characteristics"], variables)
    assert {row.study: row.rob for row in again.itertuples()}["Study 02"] == "Not recorded"

    counts = sup.transitivity_categories(summary, document["study_characteristics"], variables)
    hospital = counts[(counts["comparison"] == "Drug A vs Placebo") & (counts["category"] == "Hospital")]
    assert list(hospital["studies"]) == [2]
    assert counts["studies"].sum() == 3
    empty = sup.transitivity_points(summary, [], variables)
    assert empty.empty and list(empty.columns) == ["variable", "label", "comparison", "study", "value", "rob"]
