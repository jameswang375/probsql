from probsql.models.base import PredictiveModel
from tabicl import TabICLRegressor
from typing import Any
import pandas as pd
import numpy as np

class TabICLModel(PredictiveModel):
    """Backend using TabICL's regressor to return a grid of quantiles.
        Device set to None means TabICL will automatically try CUDA, then XPU, then MPS, then CPU via TabICL README.
        Pass in any device as an argument to use that specifically.
    """
    def __init__(self, device: str | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.device = device

    def _fit(self, X_context: pd.DataFrame, y_context: pd.Series) -> None:
        self._reg = TabICLRegressor(random_state=self.seed, device=self.device)
        self._reg.fit(X_context, y_context)

    def _predict_quantiles(self, X_query: pd.DataFrame) -> np.ndarray:
        return self._reg.predict(
            X_query, output_type="quantiles", alphas=self.levels.tolist()
        )