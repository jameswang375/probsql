import numpy as np
import pandas as pd

from probsql.models.base import PredictiveModel
from statistics import NormalDist

class StubModel(PredictiveModel):
    """Fake model for building and testing the pipeline.

    Predicts around the context mean, with spread from the context std.
    Output is deterministic: the same inputs and seed give the same numbers.
    It makes no statistical claims. It only has the right shape and behaves
    sensibly (sorted quantiles, wider at higher confidence).
    """

    def fit(self, X_context: pd.DataFrame, y_context:pd.Series) -> "StubModel":
        y = pd.Series(y_context).dropna()
        if len(y) == 0:
            raise ValueError("StubModel requires at least one context row with a target")
        self._center = float(y.mean())
        self._scale = float(y.std(ddof=0)) or 1.0
        return self

    def predict_quantiles(self, X_query: pd.DataFrame) -> np.ndarray:
        if not hasattr(self, "_center"):
            raise RuntimeError("call fit() before predict_quantiles()")
        n = len(X_query)
        rng = np.random.default_rng(self.seed)
        # Per-row shift, so rows differ but reruns match.
        shifts = rng.normal(0.0, 1.0 *self._scale, size=(n, 1))
        # Standard normal quantiles via the inverse of the CDF.
        z = np.array([NormalDist().inv_cdf(level) for level in self.levels])
        return self._center + shifts + self._scale * z[None, :]

    