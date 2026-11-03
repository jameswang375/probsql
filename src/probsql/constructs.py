import numpy as np
from dataclasses import dataclass

# Levels closer than this are treated as equal. It absorbs float noise like
# (1 - 0.9) / 2 == 0.04999999999999999.
_TOL = 1e-9


def _check_grid(grid, levels):
    grid = np.asarray(grid, dtype=float)
    levels = np.asarray(levels, dtype=float)
    if grid.ndim != 2 or grid.shape[1] != len(levels):
        raise ValueError(
            f"grid must have shape (n_rows, {len(levels)}), got {grid.shape}"
        )
    return grid, levels


def quantile_at(grid, levels, p):
    """Return the p-quantile for every row, as an array of shape (n_rows,).

    Levels on the grid return that column exactly; levels between two grid
    points are linearly interpolated. Levels outside [levels[0], levels[-1]]
    raise ValueError, because the grid has no information there.
    """
    grid, levels = _check_grid(grid, levels)
    if not np.isfinite(p):
        raise ValueError("level must be a finite number")
    lo, hi = levels[0], levels[-1]
    if p < lo - _TOL or p > hi + _TOL:
        raise ValueError(
            f"level {p} is outside the supported range [{lo}, {hi}]"
        )
    p = min(max(float(p), lo), hi)  # snap float noise at the edges

    j = int(np.searchsorted(levels, p))  # first index with levels[j] >= p
    if abs(levels[j] - p) <= _TOL:
        return grid[:, j]
    if j > 0 and abs(levels[j - 1] - p) <= _TOL:
        return grid[:, j - 1]
    # Liner interpolation between levels[j - 1] and levels[j].
    weight = (p - levels[j - 1]) / (levels[j] - levels[j - 1])
    return grid[:, j - 1] * (1 - weight) + grid[:, j] * weight


def interval_levels(level):
    """Central interval of the given coverage -> (lower level, upper level)."""
    if not (0 < level < 1):
        raise ValueError("interval level must be strictly between 0 and 1")
    tail = (1 - level) / 2
    return tail, 1 - tail


def predict(grid, levels):
    """Point prediction: the median."""
    return quantile_at(grid, levels, 0.5)


def low(grid, levels, level):
    """Lower end of the central interval with the given coverage."""
    return quantile_at(grid, levels, interval_levels(level)[0])


def high(grid, levels, level):
    """Upper end of the central interval with the given coverage."""
    return quantile_at(grid, levels, interval_levels(level)[1])


def prob_exceeds(grid, levels, threshold):
    """P(value > threshold) for every row, as an array of shape (n_rows,).

    The grid only resolves probabilities between levels[0] and levels[-1],
    so results are clamped to [1 - levels[-1], 1 - levels[0]]. Beyond that
    range the value is a bound, not an exact probability.
    """
    grid, levels = _check_grid(grid, levels)
    if not np.isfinite(threshold):
        raise ValueError("threshold must be a finite number")
    n, k = grid.shape
    idx = (grid < threshold).sum(axis=1)  # how many prices in each row are below the threshold
    cdf = np.empty(n)
    cdf[idx == 0] = levels[0] # below every price: clamp to the first level
    cdf[idx == k] = levels[-1] # above every price: clamp to the last level

    mid = (idx > 0) & (idx < k) # Rows where the threshold is between two grid points
    rows = np.nonzero(mid)[0]
    i = idx[mid]
    x0, x1 = grid[rows, i - 1], grid[rows, i] # price just below, price just above
    y0, y1 = levels[i - 1], levels[i] # level just below, level just abovce
    weight = (threshold - x0)/ (x1 - x0) # linear interpolation weight
    cdf[mid] = y0 * (1 - weight) + y1 * weight

    return 1.0 - cdf

@dataclass(frozen=True)
class Construct:
    name: str  # SQL name, uppercase
    n_args: int # number of arguments in the SQL call



REGISTRY = {
    "PSQL_PREDICT": Construct("PSQL_PREDICT", 1),   # PSQL_PREDICT(col)
    "PSQL_LOW": Construct("PSQL_LOW", 2),           # PSQL_LOW(col, level)
    "PSQL_HIGH": Construct("PSQL_HIGH", 2),         # PSQL_HIGH(col, level)
}

