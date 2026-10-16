"""The per-patient uniform-draw cache must not change any number (Common Random Numbers: same seed, same draws)."""
import numpy as np

from heor_tree_engine import microsim_engine as me


def test_prefix_of_a_stream_is_independent_of_requested_length() -> None:
    seeds = np.random.SeedSequence(7).spawn(5)
    for i in range(5):
        long = np.random.default_rng(seeds[i]).random(80)
        short = np.random.default_rng(seeds[i]).random(31)
        assert (long[:31] == short).all()


def test_cached_matrix_equals_fresh_matrix_for_any_requested_length() -> None:
    seeds = np.random.SeedSequence(11).spawn(300)
    me._UNIFORM_CACHE.clear()
    fresh = lambda cols: np.array([np.random.default_rng(seeds[100 + j]).random(cols) for j in range(40)])  # noqa: E731
    for cols in (20, 50, 35, 90, 10):  # growing, shrinking, growing again
        got = me._patient_uniforms(11, seeds, 100, 40, cols)
        assert got.shape == (40, cols) and (got == fresh(cols)).all()
    assert len(me._UNIFORM_CACHE) == 1


def test_no_caching_without_a_fixed_seed_or_over_the_size_limit() -> None:
    seeds = np.random.SeedSequence(3).spawn(10)
    me._UNIFORM_CACHE.clear()
    me._patient_uniforms(None, seeds, 0, 10, 12)
    assert not me._UNIFORM_CACHE
    old = me._UNIFORM_CACHE_MAX_BYTES
    me._UNIFORM_CACHE_MAX_BYTES = 100
    try:
        got = me._patient_uniforms(3, seeds, 0, 10, 12)
        assert got.shape == (10, 12) and not me._UNIFORM_CACHE
    finally:
        me._UNIFORM_CACHE_MAX_BYTES = old


def test_cache_is_bounded() -> None:
    seeds = np.random.SeedSequence(5).spawn(50)
    me._UNIFORM_CACHE.clear()
    old = me._UNIFORM_CACHE_MAX_BYTES
    me._UNIFORM_CACHE_MAX_BYTES = 3 * 10 * 20 * 8  # room for three 10x20 matrices
    try:
        for start in range(0, 50, 10):
            me._patient_uniforms(5, seeds, start, 10, 20)
        assert sum(m.nbytes for m in me._UNIFORM_CACHE.values()) <= me._UNIFORM_CACHE_MAX_BYTES
        assert len(me._UNIFORM_CACHE) == 3
    finally:
        me._UNIFORM_CACHE_MAX_BYTES = old
        me._UNIFORM_CACHE.clear()
