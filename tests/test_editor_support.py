from heor_tree_engine.editor_support import editor_row_key, editor_state_canvas_label


def test_editor_row_key_stays_with_row_after_preceding_row_is_deleted() -> None:
    rows = [
        {"name": "first"},
        {"name": "middle"},
        {"name": "last"},
    ]
    keys_before = {row["name"]: editor_row_key("variable", row) for row in rows}

    rows.pop(1)

    assert [editor_row_key("variable", row) for row in rows] == [
        keys_before["first"],
        keys_before["last"],
    ]
    assert len(set(keys_before.values())) == len(rows) + 1


def test_microsim_abm_state_labels_follow_markov_depth_rules() -> None:
    assert editor_state_canvas_label("Healthy", "healthy", "chance", True) == "Healthy"
    assert editor_state_canvas_label("chance_2", "chance_2", "chance", False) == ""
    assert editor_state_canvas_label("Relapse", "chance_2", "chance", False) == "Relapse"
    assert editor_state_canvas_label("Dead", "dead", "terminal", False) == "Dead"