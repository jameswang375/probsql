import duckdb
import pytest
import sqlglot

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


def expected_by_id(con):
    """What the stub should predict for each id, computed directly."""
    df = con.sql("SELECT * FROM listings").df()
    feats = ["id", "area", "label"]
    known = df["price"].notna()
    model = StubModel().fit(df.loc[known, feats], df.loc[known, "price"])
    medians = constructs.predict(model.predict_quantiles(df[feats]), model.levels)
    return dict(zip(df["id"], medians))


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
        "SELECT id FROM listings WHERE PSQL_PREDICT(price) > 300000",
        "SELECT PSQL_LOW(price, 0.9) AS lo FROM listings",
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