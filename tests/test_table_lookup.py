import pytest

from heor_tree_engine import (
    Engine,
    Param,
    build_tables_mapping,
    evaluate_expression,
    resolve,
    resolve_markov,
    run_markov,
)
from heor_tree_engine.expressions import ExpressionError


MORTALITY_RAW = [
    {
        "name": "t_mortality",
        "columns": ["Age", "Mortality"],
        "data": [
            {"Age": 0, "Mortality": 0.01},
            {"Age": 10, "Mortality": 0.02},
            {"Age": 20, "Mortality": 0.05},
            {"Age": 30, "Mortality": 0.10},
            {"Age": 40, "Mortality": 0.20},
        ],
    },
]

MULTI_RAW = [
    {
        "name": "t_multi",
        "columns": ["Age", "Male", "Female"],
        "data": [
            {"Age": 0, "Male": 0.01, "Female": 0.011},
            {"Age": 10, "Male": 0.02, "Female": 0.021},
            {"Age": 20, "Male": 0.05, "Female": 0.051},
        ],
        "default_value_column": "Male",
    },
]


def _ev(expr: str, variables: dict | None = None, raw_tables=MORTALITY_RAW) -> float:
    tables = build_tables_mapping(raw_tables)
    return evaluate_expression(expr, {}, variables or {}, None, tables)


def test_one_arg_shorthand_uses_the_sole_non_key_column() -> None:
    assert _ev("t_mortality[20]") == pytest.approx(0.05)


def test_linear_interpolation_is_the_default_method() -> None:
    assert _ev("t_mortality[25]") == pytest.approx(0.075)
    assert _ev('t_mortality[25, "Mortality"]') == pytest.approx(0.075)
    assert _ev('t_mortality[25, "Age", "Mortality"]') == pytest.approx(0.075)
    assert _ev("t_mortality[25, 0, 1]") == pytest.approx(0.075)


def test_out_of_range_values_clamp_to_the_nearest_edge() -> None:
    assert _ev('t_mortality[-5, "Age", "Mortality"]') == pytest.approx(0.01)
    assert _ev('t_mortality[999, "Age", "Mortality"]') == pytest.approx(0.20)


def test_exact_method_requires_a_real_match() -> None:
    assert _ev('t_mortality[20, "Age", "Mortality", "exact"]') == pytest.approx(0.05)
    with pytest.raises(ExpressionError, match="No exact match"):
        _ev('t_mortality[25, "Age", "Mortality", "exact"]')


def test_truncate_method_floors_to_the_highest_key_at_or_below() -> None:
    assert _ev('t_mortality[25, "Age", "Mortality", "truncate"]') == pytest.approx(0.05)
    assert _ev('t_mortality[-5, "Age", "Mortality", "truncate"]') == pytest.approx(0.01)


def test_lookup_value_can_be_an_arbitrary_formula() -> None:
    assert _ev('t_mortality[Current_age, "Age", "Mortality"]', {"Current_age": 20}) == pytest.approx(0.05)
    assert _ev('t_mortality[Current_age + 5, "Age", "Mortality"]', {"Current_age": 20}) == pytest.approx(0.075)


def test_default_result_column_setting_drives_the_one_arg_form() -> None:
    assert _ev("t_multi[10]", raw_tables=MULTI_RAW) == pytest.approx(0.02)
    assert _ev('t_multi[10, "Female"]', raw_tables=MULTI_RAW) == pytest.approx(0.021)


def test_one_arg_form_without_a_default_raises_a_clear_error() -> None:
    raw = [{"name": "t3", "columns": ["Age", "A", "B"], "data": [{"Age": 1, "A": 2, "B": 3}]}]
    with pytest.raises(ExpressionError, match="default result column"):
        _ev("t3[1]", raw_tables=raw)


def test_unknown_table_column_and_method_raise_clear_errors() -> None:
    with pytest.raises(ExpressionError, match="Unknown table"):
        _ev("t_unknown[1]")
    with pytest.raises(ExpressionError, match="no column"):
        _ev('t_mortality[20, "Bogus"]')
    with pytest.raises(ExpressionError, match="Unknown lookup method"):
        _ev('t_mortality[20, "Age", "Mortality", "bogus"]')


def test_columns_field_falls_back_to_row_key_order_when_absent() -> None:
    # Backward compatibility: a table saved before the explicit "columns"
    # field existed should still resolve from its row dict keys.
    raw = [
        {
            "name": "t_legacy",
            "data": [
                {"column_1": 0, "column_2": 0.01},
                {"column_1": 10, "column_2": 0.02},
            ],
        }
    ]
    assert _ev("t_legacy[10]", raw_tables=raw) == pytest.approx(0.02)


def test_lookup_works_inside_a_decision_tree_terminal_cost() -> None:
    document = {
        "root_id": "d1",
        "nodes": [
            {"id": "d1", "type": "decision", "branches": [{"label": "Treat", "child_id": "t1"}]},
            {
                "id": "t1",
                "type": "terminal",
                "cost": 't_cost[Age, "Age", "Cost"]',
                "effect": 1.0,
                "year": 0,
            },
        ],
        "parameters": [{"name": "Age", "base": 45}],
        "tables": [
            {
                "name": "t_cost",
                "columns": ["Age", "Cost"],
                "data": [{"Age": 40, "Cost": 100}, {"Age": 50, "Cost": 200}],
            }
        ],
    }
    tree, parameters = resolve(document)
    tables = build_tables_mapping(document.get("tables"))
    result = Engine().run(tree, parameters, tables=tables)
    assert not result.errors
    assert result.strategies["Treat"].cost == pytest.approx(150.0)


def test_lookup_works_inside_a_markov_transition_and_sees_cycle() -> None:
    document = {
        "id": "m1",
        "name": "Model",
        "cycles": 3,
        "half_cycle_correction": False,
        "states": [
            {"id": "Healthy", "cost": 10, "effect": 1.0, "initial_prob": 1.0},
            {"id": "Dead", "cost": 0, "effect": 0.0, "initial_prob": 0.0, "is_absorbing": True},
        ],
        "transitions": [
            {"from_state": "Healthy", "to_state": "Dead", "probability": 't_mortality[_cycle, "Cycle", "Prob"]'},
            {"from_state": "Healthy", "to_state": "Healthy", "probability": "#"},
        ],
        "tables": [
            {
                "name": "t_mortality",
                "columns": ["Cycle", "Prob"],
                "data": [
                    {"Cycle": 0, "Prob": 0.1},
                    {"Cycle": 1, "Prob": 0.2},
                    {"Cycle": 2, "Prob": 0.3},
                ],
            }
        ],
    }
    model = resolve_markov(document)
    tables = build_tables_mapping(document.get("tables"))
    result = run_markov(model, {}, tables=tables)
    assert not result.errors
    assert result.trace[1]["Dead"] == pytest.approx(0.1)
    assert result.trace[2]["Dead"] == pytest.approx(0.1 + 0.9 * 0.2)
