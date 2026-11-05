import pandas as pd
from sqlglot import exp

from probsql import constructs
from probsql.parser import find_constructs

_RID = "_psql_rid"

_REDUCERS = {
    "PSQL_PREDICT": lambda grid, levels, args: constructs.predict(grid, levels),
    "PSQL_LOW": lambda grid, levels, args: constructs.low(grid, levels, args[0]),
    "PSQL_HIGH": lambda grid, levels, args: constructs.high(grid, levels, args[0]),
}


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
    name = node.name.upper()
    if name not in _REDUCERS:
        _unsupported(name)
    if node.find_ancestor(exp.Select) is not ast:
        _unsupported("PSQL_* inside a subquery")
    top = node
    while top.parent is not ast:
        top = top.parent
    if not any(top is e for e in ast.expressions):
        _unsupported("PSQL_* outside the SELECT list")
    if not isinstance(node.expressions[0], exp.Column):
        _unsupported(f"{name} on anything other than a plain column")
    if name in ("PSQL_LOW", "PSQL_HIGH"):
        level = node.expressions[1]
        if not isinstance(level, exp.Literal) or level.is_string:
            _unsupported(f"{name} with a level that is not a plain number")


def _grid(con, table_name, column_name, model_factory):
    """Run the model once. Returns (rowids, quantile grid, levels)."""
    df = con.sql(f"SELECT rowid AS {_RID}, * FROM {_quote(table_name)}").df()
    target = next((c for c in df.columns if c.lower() == column_name.lower()), None)
    if target is None or target == _RID:
        raise ValueError(f"column {column_name!r} not found in {table_name!r}")
    if not pd.api.types.is_numeric_dtype(df[target]):
        raise ValueError(f"PSQL_* needs a numeric column, but {target!r} is not")

    features = [c for c in df.columns if c not in (_RID, target)]
    if not features:
        raise ValueError("the table has no other columns to predict from")

    known = df[target].notna()
    model = model_factory().fit(df.loc[known, features], df.loc[known, target])
    grid = model.predict_quantiles(df[features])
    return df[_RID].to_numpy(), grid, model.levels


def _reduce(node, grid, levels):
    args = [float(a.name) for a in node.expressions[1:]]
    return _REDUCERS[node.name.upper()](grid, levels, args)


def rewrite(ast, con, model_factory):
    """Return a copy of ast with each PSQL_* call replaced by a column from a
    registered table of results, joined back on rowid. The model runs once per
    target column."""
    ast = ast.copy()
    nodes = find_constructs(ast)
    if not nodes:
        return ast

    _check_query(ast, con)
    for node in nodes:
        _check_node(node, ast)

    
    groups = {}
    for node in nodes:
        groups.setdefault(node.expressions[0].name.lower(), []).append(node)

    table = ast.args["from_"].this
    alias = table.alias_or_name
    for i, group in enumerate(groups.values()):
        name = f"_psql_pred_{i}"
        column = group[0].expressions[0].name
        rids, grid, levels = _grid(con, table.name, column, model_factory)

        data = {"rid": rids}
        for j, node in enumerate(group):
            data[f"v{j}"] = _reduce(node, grid, levels)
        con.register(name, pd.DataFrame(data))

        join_condition = exp.column("rowid", table=alias).eq(exp.column("rid", table=name))
        ast.join(exp.to_table(name), on=join_condition, join_type="left", copy=False)
        for j, node in enumerate(group):
            node.replace(exp.column(f"v{j}", table=name))
    return ast