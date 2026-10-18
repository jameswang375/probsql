import duckdb
import pytest
from pandas.testing import assert_frame_equal
from sqlglot.errors import ParseError
from probsql.engine import Engine

@pytest.fixture
def con():
    con = duckdb.connect()
    con.execute("CREATE TABLE neighborhoods (id INTEGER, name VARCHAR)")
    con.execute(
        "INSERT INTO neighborhoods VALUES (1, 'Downtown'), (2, 'Harbor'), (3, 'Hillside')"
    )
    con.execute(
        "CREATE TABLE listings "
        "(id INTEGER, price DOUBLE, area DOUBLE, neighborhood_id INTEGER)"
    )
    con.execute(
        """
        INSERT INTO listings VALUES
            (1, 250000, 900, 1),
            (2, 410000, 1400, 1),
            (3, 320000, 1100, 2),
            (4, 575000, 2000, 2),
            (5, 198000, 750, 3),
            (6, 365000, 1250, 3),
            (7, NULL, 1000, 1),
            (8, 289000, NULL, 2),
            (9, 450000, 1600, NULL),
            (10, 305000, 1050, 3),
            (11, 520000, 1800, 1),
            (12, 275000, 950, 2)
        """
    )
    return con

@pytest.fixture
def engine(con):
    return Engine(con=con)

def normalize(df):
    # SQL doesn't guarantee row order without ORDER BY, so sort before comparing.
    return df.sort_values(list(df.columns)).reset_index(drop=True)

QUERIES = [
    "SELECT * FROM listings",
    "SELECT id, price FROM listings WHERE price > 300000",
    "SELECT id, price FROM listings ORDER BY price DESC, id LIMIT 5",
    "SELECT neighborhood_id, AVG(price) AS avg_price, COUNT(*) AS n "
    "FROM listings GROUP BY neighborhood_id",
    "SELECT l.id, l.price, n.name FROM listings l "
    "JOIN neighborhoods n ON l.neighborhood_id = n.id",
    "WITH pricey AS (SELECT * FROM listings WHERE price > 300000) "
    "SELECT COUNT(*) AS n FROM pricey",
    "SELECT id, UPPER(name) AS upper_name FROM neighborhoods WHERE name <> 'Harbor'",
]

@pytest.mark.parametrize("sql", QUERIES)
def test_passthrough_matches_duckdb(con, engine, sql):
    expected = normalize(con.sql(sql).df())
    actual = normalize(engine.query(sql))
    assert_frame_equal(actual, expected)

def test_unknown_function_fails_in_duckdb(engine):
    with pytest.raises(duckdb.Error):
        engine.query("SELECT not_a_real_fn(price) FROM listings")

@pytest.mark.parametrize(
    "sql",
    [
        "CREATE TABLE t (x INTEGER)",
        "INSERT INTO listings VALUES (99, 1, 1, 1)",
    ],
)
def test_non_read_statements_rejected(engine, sql):
    with pytest.raises(ValueError):
        engine.query(sql)

def test_multiple_statementes_rejected(engine):
    with pytest.raises(ValueError):
        engine.query("SELECT 1; SELECT 2")

def test_malformed_sql_raises_parse_error(engine):
    with pytest.raises(ParseError):
        engine.query("SELECT price FROM listings WHERE (")

