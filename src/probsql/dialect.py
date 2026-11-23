from sqlglot import exp
from sqlglot.dialects.duckdb import DuckDB

class ProbSQL(DuckDB):
    """DuckDB plus keyword forms that compile to the PSQL_* function calls:
        PREDICTION OF price            -> PSQL_PREDICT(price)
        PREDICTION LOW 0.9 OF price    -> PSQL_LOW(price, 0.9)
        PREDICTION HIGH 0.9 OF price   -> PSQL_HIGH(price, 0.9)
        PROBABILITY OF price > 300000  -> PSQL_PROBABILITY(price > 300000)
    """

    class Parser(DuckDB.Parser):
        def _parse_unary(self) -> exp.Expression | None:
            if self._match_text_seq("PROBABILITY", "OF"):
                condition = self._parse_assignment()
                return exp.Anonymous(this="PSQL_PROBABILITY", expressions=[condition])

            start = self._index
            if self._match_text_seq("PREDICTION", "OF"):
                column = self._parse_column()
                if column is not None:
                    return exp.Anonymous(this="PSQL_PREDICT", expressions=[column])
                self._retreat(start)  # not our syntax, so let DuckDB handle it

            for word, name in (("LOW", "PSQL_LOW"), ("HIGH", "PSQL_HIGH")):
                start = self._index
                if self._match_text_seq("PREDICTION", word):
                    level = self._parse_number()
                    if level is None:
                        self._retreat(start)  # e.g. a column "prediction" aliased "low"
                        continue
                    if not self._match_text_seq("OF"):
                        self.raise_error(f"Expected OF after PREDICTION {word} <number>")
                    column = self._parse_column()
                    if column is None:
                        self.raise_error(f"Expected a column after PREDICTION {word} <number> OF")
                    return exp.Anonymous(this=name, expressions=[column, level])

            return super()._parse_unary()