import duckdb
import polars as pl
from sqlglot.dialects.dialect import Dialect

from probsql.parser import parse_read_query
from probsql.rewriter import rewrite
from probsql.models.base import ModelFactory
from probsql.dialect import ProbSQL


class Engine:
    def __init__(self, model_factory: ModelFactory, con: duckdb.DuckDBPyConnection | None = None, dialect: str | type[Dialect] = ProbSQL) -> None:
        # Use the caller's connection if given, otherwise a fresh in-memory DB.
        self.con = con if con is not None else duckdb.connect()
        self.dialect = dialect  # must match the database behind con
        self.model_factory = model_factory  # creates a fresh model per prediction

    def query(self, sql: str) -> pl.DataFrame:
        ast = parse_read_query(sql, self.dialect)
        ast = rewrite(ast, self.con, self.model_factory)
        return self.con.sql(ast.sql(dialect=self.dialect)).pl()