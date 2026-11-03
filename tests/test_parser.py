import pytest
import sqlglot

from probsql.parser import find_constructs

def parse(sql):
    return sqlglot.parse_one(sql, read="duckdb")

def test_find_constructs_in_select():
    sql = "SELECT PSQL_PREDICT(price) FROM listings"
    nodes = find_constructs(parse(sql))
    assert [n.name for n in nodes] == ["PSQL_PREDICT"]

def test_find_constructs_in_where():
    sql = "SELECT id FROM listings WHERE PSQL_LOW(price, 0.9) > 300000"
    nodes = find_constructs(parse(sql))
    assert [n.name for n in nodes] == ["PSQL_LOW"]

def test_find_all_constructs():
    sql = "SELECT PSQL_PREDICT(price), PSQL_LOW(price, 0.9), PSQL_HIGH(price, 0.9) FROM listings"
    assert len(find_constructs(parse(sql))) == 3

def test_name_match_is_case_insensitive():
    sql = "SELECT psql_predict(price) FROM listings"
    nodes = find_constructs(parse(sql))
    assert len(nodes) == 1

def test_ordinary_sql_has_no_constructs():
    assert find_constructs(parse("SELECT id, UPPER(name) FROM listings")) == []

def test_unknown_function_is_ignored():
    assert find_constructs(parse("SELECT not_a_real_fn(price) FROM listings")) == []

def test_wrong_argument_count_raises():
    with pytest.raises(ValueError, match="PSQL_LOW"):
        find_constructs(parse("SELECT PSQL_LOW(price) FROM listings"))