"""Gamma follows the flexsurv convention (shape, RATE; scale = 1 / rate).

Documents saved before the switch stored k (shape) and theta (scale); the
engine still evaluates them and the migration rewrites them to shape/rate with
identical results."""

import math

import pytest

from heor_tree_engine.distributions import (
    distribution_cdf,
    distribution_mean,
    distribution_ppf,
    migrate_legacy_gamma_parameters,
    reciprocal_parameter_value,
)

scipy = pytest.importorskip("scipy")


def _resolve(value):
    return float(value)


def _gamma(**parameters):
    return {"name": "g", "type": "Gamma", "parameters": parameters}


def test_gamma_shape_rate_matches_flexsurv_definition():
    # flexsurv / R dgamma: shape = 4, rate = 0.5 -> mean 8, variance 16, scale 2
    definition = _gamma(shape=4, rate=0.5)
    assert distribution_mean(definition, _resolve) == pytest.approx(8.0)
    # P(X <= x) for Gamma(4, rate 0.5) at x = 8 (regularised lower incomplete gamma, known value)
    assert distribution_cdf(definition, 8.0, _resolve) == pytest.approx(0.5665298796, abs=1e-9)
    assert distribution_ppf(definition, 0.5, _resolve) == pytest.approx(7.3441, abs=1e-3)


def test_legacy_k_theta_documents_still_evaluate_identically():
    legacy = _gamma(k=4, theta=2)
    modern = _gamma(shape=4, rate=0.5)
    for x in (0.5, 2.0, 8.0, 20.0):
        assert distribution_cdf(legacy, x, _resolve) == pytest.approx(distribution_cdf(modern, x, _resolve), abs=1e-12)
    assert distribution_mean(legacy, _resolve) == pytest.approx(distribution_mean(modern, _resolve), abs=1e-12)


def test_migration_rewrites_k_theta_to_shape_rate_without_changing_the_distribution():
    parameters = {"k": "25", "theta": "4"}
    migrate_legacy_gamma_parameters(parameters)
    assert parameters == {"shape": "25", "rate": "0.25"}
    assert distribution_mean(_gamma(**parameters), _resolve) == pytest.approx(25 * 4)


def test_migration_keeps_expression_values_meaningful():
    parameters = {"k": "shape_cost", "theta": "scale_cost"}
    migrate_legacy_gamma_parameters(parameters)
    assert parameters == {"shape": "shape_cost", "rate": "1/(scale_cost)"}


def test_migration_non_terminating_reciprocal_is_exact():
    parameters = {"k": 2.0, "theta": 3.0}
    migrate_legacy_gamma_parameters(parameters)
    assert parameters["shape"] == 2.0
    assert eval(str(parameters["rate"])) == pytest.approx(1.0 / 3.0, abs=1e-12)


def test_migration_leaves_new_keys_alone():
    parameters = {"shape": "4", "rate": "0.5"}
    migrate_legacy_gamma_parameters(parameters)
    assert parameters == {"shape": "4", "rate": "0.5"}


def test_reciprocal_values():
    assert reciprocal_parameter_value(4) == "0.25"
    assert reciprocal_parameter_value("2") == "0.5"
    assert reciprocal_parameter_value("x + 1") == "1/(x + 1)"
    assert math.isclose(eval(reciprocal_parameter_value(3)), 1 / 3, rel_tol=1e-9)
