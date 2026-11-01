import hashlib
from statistics import NormalDist

import numpy as np
import pandas as pd

from probsql.models.base import PredictiveModel


def _row_z(values, seed: int) -> float:
    """A repeatable pseudo-random normal draw determined only by the row's
    values and the seed (not by the row's position in a batch)."""
    parts = []
    for v in values:
        if pd.isna(v):
            parts.append("nan")
        elif isinstance(v, (int, float, np.integer, np.floating)):
            parts.append(repr(float(v)))  # 900 and 900.0 count as the same
        else:
            parts.append(str(v))
    digest = hashlib.sha256(f"{seed}|{'|'.join(parts)}".encode()).digest()
    # Use 53 bits so the float math is exact and u stays strictly inside (0, 1).
    bits = int.from_bytes(digest[:8], "big") >> 11
    u = (bits + 0.5) / 2.0**53
    return NormalDist().inv_cdf(u)


class StubModel(PredictiveModel):
    """Fake model for building and testing the pipeline.

    Predicts around the context mean, with spread from the context std, plus a
    small per-row shift derived from the row's own values. It makes no
    statistical claims; it only has the right shape and keeps the interface
    promises.
    """

    def _fit(self, X_context, y_context):
        y = y_context.astype(float)
        self._center = float(y.mean())
        self._scale = float(y.std(ddof=0)) or 1.0

    def _predict_quantiles(self, X_query):
        nd = NormalDist()
        z = np.array([nd.inv_cdf(p) for p in self.levels])
        rows = X_query.itertuples(index=False, name=None)
        shifts = np.array(
            [0.1 * self._scale * _row_z(r, self.seed) for r in rows]
        ).reshape(-1, 1)
        return self._center + shifts + self._scale * z[None, :]