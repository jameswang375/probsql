import duckdb
import pandas as pd
import sqlglot
from sqlglot import exp

class Engine:
    def __init__(self, con=None):
        # Use the caller's connection if given, otherwise a fresh in-memory DB.
        self.con = con if con is not None else duckdb.connect()

        
    def query(self, sql: str) -> pd.DataFrame:
        # sqlglot.parse returns a list, one entry per ;-separated statement.
        statements = [s for s in sqlglot.parse(sql, read="duckdb") if s is not None]
        if len(statements) != 1:
            raise ValueError("Engine.query expects exactly one statement")

        ast = statements[0]
        if not isinstance(ast, exp.Query):
            raise ValueError("Engine only supports read queries")

        # Always regenerate SQL from the tree, never run the original string.
        final_sql = ast.sql(dialect="duckdb")
        return self.con.sql(final_sql).df()
