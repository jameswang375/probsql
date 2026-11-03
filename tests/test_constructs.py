import numpy as np
import pytest

from probsql import constructs as c
from probsql.models.base import DEFAULT_LEVELS

LEVELS = np.array(DEFAULT_LEVELS)
# Two rows whose quantile at level p is 100*p (row 0) and 1000 + 100*p (row 1).
GRID = np.array([LEVELS * 100, 1000 + LEVELS * 100])


def test_exact_grid_level():
    np.testing.assert_allclose(c.quantile_at(GRID, LEVELS, 0.5), [50, 1050])


def test_grid_edges_are_allowed():
    np.testing.assert_allclose(c.quantile_at(GRID, LEVELS, 0.01), [1, 1001])
    np.testing.assert_allclose(c.quantile_at(GRID, LEVELS, 0.99), [99, 1099])


def test_level_between_grid_points_is_interpolated():
    np.testing.assert_allclose(c.quantile_at(GRID, LEVELS, 0.055), [5.5, 1005.5])


def test_float_noise_still_hits_the_grid():
    p = (1 - 0.9) / 2  # 0.04999999999999999
    assert p != 0.05
    np.testing.assert_allclose(c.quantile_at(GRID, LEVELS, p), [5, 1005])


@pytest.mark.parametrize("p", [0.005, 0.995, 0.0, 1.0, np.nan])
def test_level_outside_grid_raises(p):
    with pytest.raises(ValueError):
        c.quantile_at(GRID, LEVELS, p)


def test_grid_and_levels_must_agree():
    with pytest.raises(ValueError, match="shape"):
        c.quantile_at(GRID[:, :50], LEVELS, 0.5)


def test_predict_is_the_median():
    np.testing.assert_allclose(c.predict(GRID, LEVELS), [50, 1050])


def test_low_and_high_at_90():
    np.testing.assert_allclose(c.low(GRID, LEVELS, 0.9), [5, 1005])
    np.testing.assert_allclose(c.high(GRID, LEVELS, 0.9), [95, 1095])


def test_widest_supported_interval_is_98():
    np.testing.assert_allclose(c.low(GRID, LEVELS, 0.98), [1, 1001])
    np.testing.assert_allclose(c.high(GRID, LEVELS, 0.98), [99, 1099])


@pytest.mark.parametrize("level", [0.99, 0.995, 0.0, 1.0, -0.5])
def test_interval_beyond_grid_raises(level):
    with pytest.raises(ValueError):
        c.low(GRID, LEVELS, level)
    with pytest.raises(ValueError):
        c.high(GRID, LEVELS, level)


def test_coarse_grid():
    levels = np.array([0.1, 0.5, 0.9])
    grid = np.array([[10.0, 50.0, 90.0]])
    np.testing.assert_allclose(c.low(grid, levels, 0.8), [10])
    with pytest.raises(ValueError):
        c.low(grid, levels, 0.9)


def test_prob_exceeds_inside_grid():
    # Row 0: threshold 5.5 sits at level 0.055, so P(>) = 0.945.
    out = c.prob_exceeds(GRID, LEVELS, 5.5)
    assert out[0] == pytest.approx(0.945)


def test_prob_exceeds_is_clamped_at_both_ends():
    # 50 is below row 1's lowest quantile (1001): the answer is "at least 0.99".
    # 5000 is above both rows' highest quantile: "at most 0.01".
    assert c.prob_exceeds(GRID, LEVELS, 50)[1] == pytest.approx(0.99)
    np.testing.assert_allclose(c.prob_exceeds(GRID, LEVELS, 5000), [0.01, 0.01])


def test_prob_exceeds_rejects_nan_threshold():
    with pytest.raises(ValueError):
        c.prob_exceeds(GRID, LEVELS, np.nan)