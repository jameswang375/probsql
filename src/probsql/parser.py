import sqlglot
from sqlglot import exp
from probsql.constructs import REGISTRY
from sqlglot.dialects.dialect import Dialect

def find_constructs(ast: exp.Expression) -> list[exp.Anonymous]:
    """Return every PSQL_* call in the query tree, as a list of nodes.

    Names are matched case-insensitively. A call with the wrong number of
    arguments raises ValueError.
    """
    found = []
    for node in ast.find_all(exp.Anonymous):
        name = node.name.upper()
        if name not in REGISTRY:
            continue
        expected = REGISTRY[name].n_args
        got = len(node.expressions)
        if got != expected:
            raise ValueError(f"{name} takes {expected} arguments, got {got}")
        found.append(node)
    return found

def parse_read_query(sql: str, dialect: str | type[Dialect]) -> exp.Query:
    """Parse sql into a single read query, or raise ValueError.

    Accepts exactly one statement, and it must be a query (SELECT and the like).
    sqlglot's ParseError is not caught, so malformed SQL raises it unchanged.
    """
    statements = [s for s in sqlglot.parse(sql, read=dialect) if s is not None]
    if len(statements) != 1:
        raise ValueError("Engine.query expects exactly one statement")
    ast = statements[0]
    if not isinstance(ast, exp.Query):
        raise ValueError("Engine only supports read queries")
    return ast