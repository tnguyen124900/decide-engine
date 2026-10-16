import math

from heor_tree_engine import cma_cba as m


def test_cma_cheapest_and_equivalence():
    r = m.cma_analysis(["A", "B", "C"], [100.0, 80.0, 120.0], [5.0, 5.0, 5.02], tolerance=0.01)
    assert r.cheapest == "B"
    assert r.effects_equivalent
    assert [x.is_cheapest for x in r.rows] == [False, True, False]
    assert r.rows[0].extra_cost_vs_cheapest == 20.0


def test_cma_warns_when_effects_differ():
    r = m.cma_analysis(["A", "B"], [100.0, 80.0], [5.0, 4.0], tolerance=0.01)
    assert not r.effects_equivalent
    assert math.isclose(r.effect_spread, 1.0 / 4.5)


def test_cma_tie_and_zero_effects():
    r = m.cma_analysis(["A", "B"], [50.0, 50.0], [0.0, 0.0])
    assert r.effects_equivalent and r.ties == ["B"] and r.cheapest == "A"


def test_cba_basic_ratios():
    r = m.cba_analysis(["Std", "New"], [1000.0, 3000.0], [1.0, 2.0], value_per_effect=5000.0)
    assert r.baseline == "Std"
    std, new = r.rows
    assert std.benefit == 5000.0 and std.net_benefit == 4000.0 and std.bcr_note == "baseline"
    assert new.benefit == 10000.0 and new.net_benefit == 7000.0
    assert new.delta_cost == 2000.0 and new.delta_benefit == 5000.0
    assert new.delta_net_benefit == 3000.0
    assert math.isclose(new.bcr, 2.5) and math.isclose(new.roi, 1.5)
    assert r.best == "New"


def test_cba_extra_benefits_and_minimize():
    # minimize objective: fewer events -> benefit counted as avoided events (sign flip)
    r = m.cba_analysis(["Std", "New"], [100.0, 150.0], [10.0, 8.0], 1000.0,
                       maximize_effect=False, extra_benefits={"New": 500.0}, baseline="Std")
    new = r.rows[1]
    assert new.benefit == -8000.0 + 500.0
    assert new.delta_benefit == (-8000.0 + 500.0) - (-10000.0)
    assert math.isclose(new.bcr, 2500.0 / 50.0)


def test_cba_dominant_dominated_cases():
    r = m.cba_analysis(["Std", "Cheap", "Bad"], [100.0, 80.0, 150.0], [1.0, 1.0, 1.0], 1000.0, baseline="Std")
    cheap, bad = r.rows[1], r.rows[2]
    assert cheap.bcr is None and cheap.bcr_note == "dominant"
    assert bad.bcr_note == "dominated" and bad.bcr == 0.0


def test_cba_default_baseline_is_cheapest_and_unique_names():
    r = m.cba_analysis(["A", "B"], [10.0, 5.0], [1.0, 1.0], 1.0)
    assert r.baseline == "B"
    try:
        m.cba_analysis(["A", "A"], [1, 2], [1, 1], 1)
    except ValueError:
        pass
    else:
        raise AssertionError("duplicate names must fail")


def test_translations_complete():
    keysets = [set(v) for v in m.TEXT.values()]
    assert all(k == keysets[0] for k in keysets)
    assert set(m.TEXT) == {"en", "fr", "es", "zh", "ko", "ja"}
    for lang, d in m.TEXT.items():
        assert d["cma_warning"].format(spread=0.1, tol=0.01)
        assert d["cba_best"].format(name="x")
        assert d["cma_tie"].format(names="a")


def test_cba_with_value_equal_wtp_equals_nmb():
    wtp = 50000.0
    costs, effects = [1000.0, 4000.0], [1.0, 1.2]
    r = m.cba_analysis(["A", "B"], costs, effects, wtp)
    for row, c, e in zip(r.rows, costs, effects):
        assert math.isclose(row.net_benefit, wtp * e - c)
