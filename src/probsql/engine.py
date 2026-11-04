import duckdb
import pandas as pd
import sqlglot
from sqlglot import exp

from probsql.rewriter import rewrite


class Engine:
    def __init__(self, model_factory, con=None, dialect="duckdb"):
        # Use the caller's connection if given, otherwise a fresh in-memory DB.
        self.con = con if con is not None else duckdb.connect()
        self.dialect = dialect  # must match the database behind con
        self.model_factory = model_factory  # creates a fresh model per prediction

    def query(self, sql: str) -> pd.DataFrame:
        statements = [s for s in sqlglot.parse(sql, read=self.dialect) if s is not None]
        if len(statements) != 1:
            raise ValueError("Engine.query expects exactly one statement")

        ast = statements[0]
        if not isinstance(ast, exp.Query):
            raise ValueError("Engine only supports read queries")

        ast = rewrite(ast, self.con, self.model_factory)

        final_sql = ast.sql(dialect=self.dialect)
        return self.con.sql(final_sql).df()