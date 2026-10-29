import numpy as np
import pandas as pd
import pytest

from probsql.models.stub import StubModel

@pytest.fixture
def data():
    X = pd.DataFrame({"area": [900, 1400, 1100, 2000, 750, 1250]})
    y = pd.Series([250000, 410000, 320000, 575000, 198000, 365000], dtype=float)
    return X, y

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

    