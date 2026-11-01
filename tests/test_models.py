import numpy as np
import pandas as pd
import pytest

from probsql.models.base import PredictiveModel
from probsql.models.stub import StubModel


@pytest.fixture
def data():
    X = pd.DataFrame({"area": [900, 1400, 1100, 2000, 750, 1250]})
    y = pd.Series([250000, 410000, 320000, 575000, 198000, 365000], dtype=float)
    return X, y


# basic behavior

def test_output_shape(data):
    X, y = data
    model = StubModel().fit(X, y)
    out = model.predict_quantiles(X.iloc[:4])
    assert out.shape == (4, len(model.levels))


def test_quantiles_sorted(data):
    X, y = data
    out = StubModel().fit(X, y).predict_quantiles(X)
    assert np.all(np.diff(out, axis=1) > 0)


def test_deterministic_with_seed(data):
    X, y = data
    a = StubModel(seed=1).fit(X, y).predict_quantiles(X)
    b = StubModel(seed=1).fit(X, y).predict_quantiles(X)
    np.testing.assert_array_equal(a, b)


def test_predict_before_fit_raises(data):
    X, _ = data
    with pytest.raises(RuntimeError):
        StubModel().predict_quantiles(X)


def test_bad_levels_rejected():
    with pytest.raises(ValueError):
        StubModel(levels=[0.9, 0.1])


# promises

def test_row_prediction_independent_of_batch(data):
    X, y = data
    model = StubModel().fit(X, y)
    full = model.predict_quantiles(X)
    for i in range(len(X)):
        alone = model.predict_quantiles(X.iloc[[i]])
        np.testing.assert_array_equal(alone[0], full[i])
    reversed_out = model.predict_quantiles(X.iloc[::-1])
    np.testing.assert_array_equal(reversed_out, full[::-1])


def test_identical_rows_get_identical_predictions(data):
    X, y = data
    model = StubModel().fit(X, y)
    out = model.predict_quantiles(pd.concat([X.iloc[[0]], X.iloc[[0]]]))
    np.testing.assert_array_equal(out[0], out[1])


# input checks in the base class

def test_fit_rejects_missing_target(data):
    X, y = data
    y = y.copy()
    y.iloc[0] = np.nan
    with pytest.raises(ValueError, match="missing"):
        StubModel().fit(X, y)


def test_fit_rejects_length_mismatch(data):
    X, y = data
    with pytest.raises(ValueError):
        StubModel().fit(X, y.iloc[:3])


def test_predict_rejects_column_mismatch(data):
    X, y = data
    model = StubModel().fit(X, y)
    with pytest.raises(ValueError, match="columns"):
        model.predict_quantiles(X.rename(columns={"area": "sqft"}))


# the base class catches misbehaving backends

class _BadBackend(PredictiveModel):
    """Throwaway backend whose output is controlled by the test."""

    def __init__(self, output_fn, **kwargs):
        super().__init__(**kwargs)
        self._output_fn = output_fn

    def _fit(self, X, y):
        pass

    def _predict_quantiles(self, X):
        return self._output_fn(len(X), len(self.levels))


def test_unsorted_backend_rejected(data):
    X, y = data
    model = _BadBackend(lambda n, k: np.tile(np.arange(k)[::-1], (n, 1)))
    with pytest.raises(RuntimeError, match="sorted"):
        model.fit(X, y).predict_quantiles(X)


def test_wrong_shape_backend_rejected(data):
    X, y = data
    model = _BadBackend(lambda n, k: np.zeros((n + 1, k)))
    with pytest.raises(RuntimeError, match="shape"):
        model.fit(X, y).predict_quantiles(X)


def test_nan_backend_rejected(data):
    X, y = data
    model = _BadBackend(lambda n, k: np.full((n, k), np.nan))
    with pytest.raises(RuntimeError, match="non-finite"):
        model.fit(X, y).predict_quantiles(X)


def test_failed_fit_does_not_look_fitted(data):
    X, y = data

    class Exploding(_BadBackend):
        def _fit(self, X, y):
            raise ValueError("boom")

    model = Exploding(lambda n, k: np.zeros((n, k)))
    with pytest.raises(ValueError, match="boom"):
        model.fit(X, y)
    with pytest.raises(RuntimeError, match="fit"):
        model.predict_quantiles(X)