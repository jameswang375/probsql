import duckdb
import pytest

from probsql.engine import Engine
from probsql.models.stub import StubModel


@pytest.fixture
def engine():
    con = duckdb.connect()
    con.execute("CREATE TABLE listings (id INTEGER, price DOUBLE, area DOUBLE)")
    con.execute(
        """
        INSERT INTO listings VALUES
            (1, 250000, 900), (2, 410000, 1400), (3, 320000, 1100),
            (4, NULL, 1000), (5, 198000, 750), (6, 365000, 1250)
        """
    )
    return Engine(con=con, model_factory=StubModel)


def test_predict_runs_end_to_end(engine):
    out = engine.query("SELECT id, PSQL_PREDICT(price) AS est FROM listings")
    assert list(out.columns) == ["id", "est"]
    assert len(out) == 6
    assert out["est"].notna().all()


def test_predicted_column_works_with_order_by(engine):
    out = engine.query(
        "SELECT id, PSQL_PREDICT(price) AS est FROM listings ORDER BY est DESC"
    )
    assert out["est"].is_monotonic_decreasing


def test_unsupported_query_raises_through_engine(engine):
    with pytest.raises(ValueError, match="not supported yet"):
        engine.query("SELECT id FROM listings WHERE PSQL_PREDICT(price) > 300000")


def test_engine_uses_the_given_model_factory(engine):
    calls = []

    def factory():
        calls.append(1)
        return StubModel()

    engine.model_factory = factory
    engine.query("SELECT PSQL_PREDICT(price) AS est FROM listings")
    assert len(calls) == 1