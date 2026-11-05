import duckdb
import pytest
import sqlglot
import numpy as np

from probsql import constructs
from probsql.models.stub import StubModel
from probsql.rewriter import rewrite


@pytest.fixture
def con():
    con = duckdb.connect()
    con.execute(
        "CREATE TABLE listings (id INTEGER, price DOUBLE, area DOUBLE, label VARCHAR)"
    )
    con.execute(
        """
        INSERT INTO listings VALUES
            (1, 250000, 900, 'a'),
            (2, 410000, 1400, 'b'),
            (3, 320000, 1100, 'a'),
            (4, NULL, 1000, 'b'),
            (5, 198000, NULL, 'a'),
            (6, 365000, 1250, 'b')
        """
    )
    return con


def parse(sql):
    return sqlglot.parse_one(sql, read="duckdb")


def run(con, sql):
    ast = rewrite(parse(sql), con, StubModel)
    return con.sql(ast.sql(dialect="duckdb")).df()


def expected_grid(con):
    df = con.sql("SELECT * FROM listings").df()
    feats = ["id", "area", "label"]
    y = df["price"]
    known = np.flatnonzero(y.notna().to_numpy())
    missing = np.flatnonzero(y.isna().to_numpy())
    rows = {}
    for i in known:
        rest = known[known != i]
        m = StubModel().fit(df[feats].iloc[rest], y.iloc[rest])
        rows[i] = m.predict_quantiles(df[feats].iloc[[i]])[0]
    m = StubModel().fit(df[feats].iloc[known], y.iloc[known])
    for i, r in zip(missing, m.predict_quantiles(df[feats].iloc[missing])):
        rows[i] = r
    grid = np.vstack([rows[i] for i in range(len(df))])
    return df["id"], grid, m.levels


def expected_by_id(con):
    ids, grid, levels = expected_grid(con)
    return dict(zip(ids, constructs.predict(grid, levels)))


def test_query_without_constructs_is_unchanged(con):
    sql = "SELECT id, price FROM listings WHERE price > 300000"
    assert rewrite(parse(sql), con, StubModel).sql() == parse(sql).sql()


def test_input_tree_is_not_modified(con):
    ast = parse("SELECT id, PSQL_PREDICT(price) AS est FROM listings")
    before = ast.sql()
    rewrite(ast, con, StubModel)
    assert ast.sql() == before


def test_prediction_matches_model_output(con):
    out = run(con, "SELECT id, PSQL_PREDICT(price) AS est FROM listings")
    expected = expected_by_id(con)
    assert list(out.columns) == ["id", "est"]
    for id_, est in zip(out["id"], out["est"]):
        assert est == pytest.approx(expected[id_])


def test_no_rows_are_lost(con):
    out = run(con, "SELECT id, PSQL_PREDICT(price) AS est FROM listings")
    assert len(out) == 6
    assert out["est"].notna().all()  # includes id 4, whose price is NULL


def test_works_with_table_alias(con):
    out = run(con, "SELECT l.id, PSQL_PREDICT(l.price) AS est FROM listings AS l")
    expected = expected_by_id(con)
    for id_, est in zip(out["id"], out["est"]):
        assert est == pytest.approx(expected[id_])


def test_case_insensitive_name(con):
    out = run(con, "SELECT id, psql_predict(price) AS est FROM listings")
    assert len(out) == 6


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT PSQL_PREDICT(price + 1) AS est FROM listings",
        "SELECT *, PSQL_PREDICT(price) AS est FROM listings",
        "SELECT PSQL_PREDICT(price) AS est FROM listings a JOIN listings b ON a.id = b.id",
        "SELECT PSQL_PREDICT(price) AS est FROM (SELECT * FROM listings)",
        "WITH l AS (SELECT * FROM listings) SELECT PSQL_PREDICT(price) AS est FROM l",
        "SELECT PSQL_PREDICT(price) AS est",
        "SELECT PSQL_PREDICT(price) AS est FROM listings UNION SELECT 1.0 FROM listings",
        "SELECT (SELECT PSQL_PREDICT(price) FROM listings) AS est FROM listings",
    ],
)
def test_unsupported_shapes_raise(con, sql):
    with pytest.raises(ValueError, match="not supported yet"):
        rewrite(parse(sql), con, StubModel)


def test_unknown_column_raises(con):
    with pytest.raises(ValueError, match="not found"):
        rewrite(parse("SELECT PSQL_PREDICT(nope) AS est FROM listings"), con, StubModel)


def test_non_numeric_target_raises(con):
    with pytest.raises(ValueError, match="numeric"):
        rewrite(parse("SELECT PSQL_PREDICT(label) AS est FROM listings"), con, StubModel)

def test_low_and_high_match_model_output(con):
    out = run(
        con,
        "SELECT id, PSQL_LOW(price, 0.9) AS lo, PSQL_HIGH(price, 0.9) AS hi FROM listings",
    )
    ids, grid, levels = expected_grid(con)
    lo = dict(zip(ids, constructs.low(grid, levels, 0.9)))
    hi = dict(zip(ids, constructs.high(grid, levels, 0.9)))
    for id_, a, b in zip(out["id"], out["lo"], out["hi"]):
        assert a == pytest.approx(lo[id_])
        assert b == pytest.approx(hi[id_])


def test_low_is_below_predict_is_below_high(con):
    out = run(
        con,
        "SELECT PSQL_LOW(price, 0.9) AS lo, PSQL_PREDICT(price) AS mid, "
        "PSQL_HIGH(price, 0.9) AS hi FROM listings",
    )
    assert (out["lo"] < out["mid"]).all()
    assert (out["mid"] < out["hi"]).all()


def test_model_fits_are_shared_across_calls_on_a_column(con):
    calls = []

    def factory():
        calls.append(1)
        return StubModel()

    sql = (
        "SELECT PSQL_PREDICT(price) AS a, PSQL_LOW(PRICE, 0.9) AS b, "
        "PSQL_HIGH(price, 0.8) AS c FROM listings"
    )
    rewrite(parse(sql), con, factory)
    assert len(calls) == 6


def test_model_runs_once_for_each_different_column(con):
    calls = []

    def factory():
        calls.append(1)
        return StubModel()

    sql = "SELECT PSQL_PREDICT(price) AS a, PSQL_PREDICT(area) AS b FROM listings"
    rewrite(parse(sql), con, factory)
    assert len(calls) == 12


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT PSQL_LOW(price, '0.9') AS lo FROM listings",
        "SELECT PSQL_LOW(price, area) AS lo FROM listings",
        "SELECT PSQL_HIGH(price, -0.9) AS hi FROM listings",
    ],
)
def test_level_must_be_a_plain_number(con, sql):
    with pytest.raises(ValueError, match="not supported yet"):
        rewrite(parse(sql), con, StubModel)


def test_level_beyond_the_grid_raises(con):
    with pytest.raises(ValueError, match="outside"):
        rewrite(parse("SELECT PSQL_LOW(price, 0.99) AS lo FROM listings"), con, StubModel)

def test_a_rows_own_price_does_not_affect_its_prediction(con):
    sql = "SELECT id, PSQL_PREDICT(price) AS est FROM listings"
    before = run(con, sql).set_index("id")["est"]
    con.execute("UPDATE listings SET price = 999999 WHERE id = 1")
    after = run(con, sql).set_index("id")["est"]
    assert after[1] == pytest.approx(before[1])
    assert after[2] != pytest.approx(before[2])  # other rows do see it