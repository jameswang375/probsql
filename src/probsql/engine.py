import duckdb
import pandas as pd

from probsql.parser import parse_read_query
from probsql.rewriter import rewrite


class Engine:
    def __init__(self, model_factory, con=None, dialect="duckdb"):
        # Use the caller's connection if given, otherwise a fresh in-memory DB.
        self.con = con if con is not None else duckdb.connect()
        self.dialect = dialect  # must match the database behind con
        self.model_factory = model_factory  # creates a fresh model per prediction

    def query(self, sql: str) -> pd.DataFrame:
        ast = parse_read_query(sql, self.dialect)
        ast = rewrite(ast, self.con, self.model_factory)
        return self.con.sql(ast.sql(dialect=self.dialect)).df()