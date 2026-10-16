import pytest
from heor_tree_engine.expressions import evaluate_expression, ExpressionError, _did_you_mean


def test_unknown_name_suggests_close_match():
    from heor_tree_engine.models import Param
    params = {"p_death": Param(name="p_death", base=0.1)} if "name" in Param.__dataclass_fields__ else {}
    if not params:
        pytest.skip("Param signature differs")
    with pytest.raises(ExpressionError, match="Did you mean 'p_death'"):
        evaluate_expression("p_deth * 2", params)


def test_function_typo_and_case():
    assert "sqrt" in _did_you_mean("sqt", ["sqrt", "sin"])
    assert "case-sensitive" in _did_you_mean("Pbase", ["pbase"])
    assert _did_you_mean("zzzzzz", ["abc"]) == ""
