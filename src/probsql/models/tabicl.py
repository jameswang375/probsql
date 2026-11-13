from probsql.models.base import PredictiveModel
from tabicl import TabICLRegressor

class TabICLModel(PredictiveModel):
    """Backend using TabICL's regressor to return a grid of quantiles.
        Device set to None means TabICL will automatically try CUDA, then XPU, then MPS, then CPU via TabICL README.
        Pass in any device as an argument to use that specifically.
    """
    def __init__(self, device=None, **kwargs):
        super().__init__(**kwargs)
        self.device = device

    def _fit(self, X_context, y_context):
        self._reg = TabICLRegressor(random_state=self.seed, device=self.device)
        self._reg.fit(X_context, y_context)

    def _predict_quantiles(self, X_query):
        return self._reg.predict(
            X_query, output_type="quantiles", alphas=self.levels.tolist()
            )