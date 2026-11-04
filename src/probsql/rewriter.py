import pandas as pd
from sqlglot import exp

from probsql import constructs
from probsql.parser import find_constructs

_RID = "_psql_rid"


def _unsupported(what):
    raise ValueError(f"{what} is not supported yet")


def _quote(name):
    return '"' + name.replace('"', '""') + '"'


def _is_base_table(con, name):
    rows = con.execute(
        "SELECT 1 FROM information_schema.tables "
        "WHERE lower(table_name) = lower(?) AND table_type = 'BASE TABLE'",
        [name],
    ).fetchall()
    return bool(rows)


def _check_query(ast, con):
    if not isinstance(ast, exp.Select):
        _unsupported("PSQL_* in this kind of query")
    from_ = ast.args.get("from_")
    if from_ is None or not isinstance(from_.this, exp.Table):
        _unsupported("PSQL_* on anything other than a base table")
    if ast.args.get("joins"):
        _unsupported("PSQL_* in a query with joins")
    if not _is_base_table(con, from_.this.name):
        _unsupported("PSQL_* on a CTE, view or subquery")
    for e in ast.expressions:
        is_star = isinstance(e, exp.Star) or (
            isinstance(e, exp.Column) and isinstance(e.this, exp.Star)
        )
        if is_star:
            _unsupported("SELECT * together with PSQL_*")


def _check_node(node, ast):
    if node.name.upper() != "PSQL_PREDICT":
        _unsupported(node.name.upper())
    if node.find_ancestor(exp.Select) is not ast:
        _unsupported("PSQL_* inside a subquery")
    top = node
    while top.parent is not ast:
        top = top.parent
    if not any(top is e for e in ast.expressions):
        _unsupported("PSQL_* outside the SELECT list")
    if not isinstance(node.expressions[0], exp.Column):
        _unsupported("PSQL_PREDICT on anything other than a plain column")


def _predictions(con, table_name, column_name, model_factory):
    """Return a DataFrame of (rid, value): one median prediction per table row."""
    df = con.sql(f"SELECT rowid AS {_RID}, * FROM {_quote(table_name)}").df()
    target = next((c for c in df.columns if c.lower() == column_name.lower()), None)
    if target is None or target == _RID:
        raise ValueError(f"column {column_name!r} not found in {table_name!r}")
    if not pd.api.types.is_numeric_dtype(df[target]):
        raise ValueError(f"PSQL_PREDICT needs a numeric column, but {target!r} is not")

    features = [c for c in df.columns if c not in (_RID, target)]
    if not features:
        raise ValueError("the table has no other columns to predict from")

    known = df[target].notna()
    model = model_factory().fit(df.loc[known, features], df.loc[known, target])
    grid = model.predict_quantiles(df[features])
    values = constructs.predict(grid, model.levels)
    return pd.DataFrame({"rid": df[_RID], "value": values})


def rewrite(ast, con, model_factory):
    """Return a copy of ast with each PSQL_PREDICT(col) replaced by a column
    from a registered table of predictions, joined back on rowid."""
    ast = ast.copy()
    nodes = find_constructs(ast)
    if not nodes:
        return ast

    _check_query(ast, con)
    for node in nodes:
        _check_node(node, ast)

    table = ast.args["from_"].this
    alias = table.alias_or_name
    for i, node in enumerate(nodes):
        name = f"_psql_pred_{i}"
        frame = _predictions(con, table.name, node.expressions[0].name, model_factory)
        con.register(name, frame)
        join_condition = exp.column("rowid", table=alias).eq(exp.column("rid", table=name))
        ast.join(exp.to_table(name), on=join_condition, join_type="left", copy=False)
        node.replace(exp.column("value", table=name))
    return ast