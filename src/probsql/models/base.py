from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

DEFAULT_LEVELS = tuple(round(i / 100, 2) for i in range(1, 100))

class PredictiveModel(ABC):
    """Interface every model backend implements.

    The model knows nothing about SQL, column names, or constructs.
    It receives tables of features and a target, and returns a predictive
    distribution for each query row, represented as a grid of quantiles.
    """

    def __init__(self, levels=DEFAULT_LEVELS, seed: int = 0):
        levels = np.asarray(levels, dtype=float)
        if levels.ndim != 1 or len(levels) == 0:
            raise ValueError("levels must be a non-empty 1-D sequence")
        if np.any(levels <= 0) or np.any(levels >= 1):
            raise ValueError("levels must be strictly in between 0 and 1")
        if np.any(np.diff(levels) <= 0):
            raise ValueError("levels must be strictly increasing")
        self.levels = levels
        self.seed = seed

    @abstractmethod
    def fit(self, X_context: pd.DataFrame, y_context: pd.Series) -> "PredictiveModel":
        """Condition on the context rows. For a foundation model this stores
        the rows rather than training. Returns self."""

    @abstractmethod
    def predict_quantiles(self, X_query: pd.DataFrame) -> np.ndarray:
        """Return an array of shape (len(X_query), len(self.levels)).

        Row i corresponds to row i of X_query. Each row is sorted ascending,
        so column j holds the quantile at self.levels[j].
        """

    