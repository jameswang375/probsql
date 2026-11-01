from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

DEFAULT_LEVELS = tuple(round(i / 100, 2) for i in range(1, 100))


class PredictiveModel(ABC):
    """Interface every model backend implements.

    The model knows nothing about SQL, column names, or constructs. It receives
    tables of features and a target, and returns a predictive distribution for
    each query row, as a grid of quantiles.

    Promises every backend must keep (enforced or checked here):
      - Shape: output is (n_query_rows, n_levels).
      - Alignment: output row i is the answer for query row i.
      - Sorted: each output row is ascending, so column j is the quantile
        at levels[j].
      - Finite: no NaN or infinite values in the output.
      - Batch independence: a row's output depends only on that row's values
        and the fitted context, never on which other rows are in the batch.
        (Documented and tested per backend; the base class cannot enforce it.)
      - Repeatable: same context, same row, same seed -> same output.
      - Context has no missing targets, and query rows have the same columns
        as the context rows.
    """

    def __init__(self, levels=DEFAULT_LEVELS, seed: int = 0):
        levels = np.asarray(levels, dtype=float)
        if levels.ndim != 1 or len(levels) == 0:
            raise ValueError("levels must be a non-empty 1-D sequence")
        if np.any(levels <= 0) or np.any(levels >= 1):
            raise ValueError("levels must be strictly between 0 and 1")
        if np.any(np.diff(levels) <= 0):
            raise ValueError("levels must be strictly increasing")
        self.levels = levels
        self.seed = seed
        self._columns = None  # set by fit() once fitting succeeds

    def fit(self, X_context: pd.DataFrame, y_context: pd.Series) -> "PredictiveModel":
        y = pd.Series(y_context)
        if len(X_context) != len(y):
            raise ValueError("X_context and y_context must have the same number of rows")
        if len(y) == 0:
            raise ValueError("context must have at least one row")
        if y.isna().any():
            raise ValueError("y_context has missing values; drop those rows first")
        self._columns = None  # a failed refit must not look fitted
        self._fit(X_context, y)
        self._columns = list(X_context.columns)  # only after _fit succeeds
        return self

    def predict_quantiles(self, X_query: pd.DataFrame) -> np.ndarray:
        if self._columns is None:
            raise RuntimeError("call fit() before predict_quantiles()")
        if list(X_query.columns) != self._columns:
            raise ValueError(
                f"X_query columns {list(X_query.columns)} do not match "
                f"context columns {self._columns}"
            )
        out = np.asarray(self._predict_quantiles(X_query), dtype=float)
        expected = (len(X_query), len(self.levels))
        if out.shape != expected:
            raise RuntimeError(f"backend returned shape {out.shape}, expected {expected}")
        if not np.all(np.isfinite(out)):
            raise RuntimeError("backend returned non-finite quantiles")
        if np.any(np.diff(out, axis=1) < 0):
            raise RuntimeError("backend returned quantiles that are not sorted")
        return out

    @abstractmethod
    def _fit(self, X_context: pd.DataFrame, y_context: pd.Series) -> None:
        """Condition on the context rows (store them, or train)."""

    @abstractmethod
    def _predict_quantiles(self, X_query: pd.DataFrame) -> np.ndarray:
        """Return an array of shape (len(X_query), len(self.levels))."""