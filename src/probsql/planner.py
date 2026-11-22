from dataclasses import dataclass
from sqlglot import exp
from probsql.constructs import REGISTRY
from probsql.parser import find_constructs
from typing import NoReturn

@dataclass(frozen=True)
class Group:
    column: str     # target column as written in query
    nodes: tuple[exp.Anonymous, ...]    # every PSQL_* call that uses this column

@dataclass(frozen=True)
class Plan:
    table: str      # the base table in FROM
    alias: str      # the name the query uses for it
    groups: tuple[Group, ...]   # one Group per target column; the model runs once per group

def _unsupported(something: str) -> NoReturn:
    raise ValueError(f"{something} is not supported yet")

def _target(node: exp.Anonymous) -> exp.Expression:
    if node.name.upper() == "PSQL_PROBABILITY":
        return _split(node.expressions[0])[0]
    return node.expressions[0]

def _number(node: exp.Expression) -> float | None:
    """Value of a constant like 500000, -5 or (100 + 5) * 2, or None if it's anything else."""
    if isinstance(node, exp.Paren):
        return _number(node.this)
    if isinstance(node, exp.Neg):
        inner_value = _number(node.this)
        return None if inner_value is None else -inner_value
    if isinstance(node, (exp.Add, exp.Sub, exp.Mul, exp.Div)):
        left_value, right_value = _number(node.this), _number(node.expression)
        if left_value is None or right_value is None:
            return None
        if isinstance(node, exp.Add):
            return left_value + right_value
        if isinstance(node, exp.Sub):
            return left_value - right_value
        if isinstance(node, exp.Mul):
            return left_value * right_value
        return None if right_value == 0 else left_value / right_value
    if isinstance(node, exp.Literal) and not node.is_string:
        return float(node.name)
    return None


def _split(comparison: exp.Expression) -> tuple[exp.Column, float, bool] | None:
    """For a comparison between a column and a constant, in either order, return
    (column, number, asks_below), where asks_below is True if it asks for the chance the
    column is under the number. Returns None for anything else."""
    if not isinstance(comparison, (exp.GT, exp.GTE, exp.LT, exp.LTE)):
        return None
    asks_below = isinstance(comparison, (exp.LT, exp.LTE))
    left, right = comparison.this, comparison.expression
    if isinstance(left, exp.Column) and _number(right) is not None:
        return left, _number(right), asks_below
    if isinstance(right, exp.Column) and _number(left) is not None:
        return right, _number(left), not asks_below
    return None


def source_table_name(ast: exp.Expression) -> str | None:
     """Name of the single table in FROM, or None if the query has no plain table."""

     if not isinstance(ast, exp.Select):
         return None
     from_ = ast.args.get("from_")
     if from_ is None or not isinstance(from_.this, exp.Table):
         return None
     return from_.this.name

def _check_query(ast: exp.Expression, source_is_base_table: bool) -> None:
    if not isinstance(ast, exp.Select):
        _unsupported("PSQL_* in this kind of query")
    from_ = ast.args.get("from_")
    if from_ is None or not isinstance(from_.this, exp.Table):
        _unsupported("PSQL_* on anything other than a base table")
    if ast.args.get("joins"):
        _unsupported("PSQL_* in a query with joins")
    if not source_is_base_table:
        _unsupported("PSQL_* on a CTE, view or subquery")
    for e in ast.expressions:
        is_star = isinstance(e, exp.Star) or (
            isinstance(e, exp.Column) and isinstance(e.this, exp.Star)
        )
        if is_star:
            _unsupported("SELECT * together with PSQL_*")

def _check_node(node: exp.Anonymous, ast: exp.Expression) -> None:
    name = node.name.upper()
    if name not in REGISTRY:
        _unsupported(name)
    if node.find_ancestor(exp.Select) is not ast:
        _unsupported("PSQL_* inside a subquery")
    top = node
    while top.parent is not ast:
        top = top.parent
    clauses = [ast.args.get(k) for k in ("where", "group", "having", "order")]
    in_select = any(top is e for e in ast.expressions)
    in_clause = any(top is c for c in clauses if c is not None)
    if not (in_select or in_clause):
        _unsupported("PSQL_* outside SELECT, WHERE, GROUP BY, HAVING or ORDER BY")
    if name == "PSQL_PROBABILITY":
        if _split(node.expressions[0]) is None:
            _unsupported("PSQL_PROBABILITY comparing anything other than a column to a constant number")
    elif not isinstance(node.expressions[0], exp.Column):
        _unsupported(f"{name} on anything other than a plain column")
    if name in ("PSQL_LOW", "PSQL_HIGH"):
        level = node.expressions[1]
        if not isinstance(level, exp.Literal) or level.is_string:
            _unsupported(f"{name} with a level that is not a plain number")

def plan(ast: exp.Expression, source_is_base_table: bool) -> Plan:
    """Decide what work the query needs, without doing any of it.

    ast: a parsed query. source_is_base_table: whether the table in FROM is a
    real table, looked up by the caller. Raises ValueError for unsupported
    queries. Does not modify ast.
    """

    nodes = find_constructs(ast)
    _check_query(ast, source_is_base_table)
    for node in nodes:
        _check_node(node, ast)

    by_column = {}
    for node in nodes:
        by_column.setdefault(_target(node).name.lower(), []).append(node)

    table = ast.args["from_"].this
    groups = tuple(Group(_target(g[0]).name, tuple(g)) for g in by_column.values())
    return Plan(table.name, table.alias_or_name, groups)