import pandas as pd
import numpy as np
from sqlglot import exp

from probsql import constructs
from probsql.parser import find_constructs

_RID = "_psql_rid"

def _probability(grid, levels, threshold, asks_below):
    """Prob here means probability above a certain quantity. 
    So return prob as is if you want probability above a certain value, otherwise return the remaining probability (1 - prob), 
    which would be the probability below a certain value."""

    prob = constructs.prob_exceeds(grid, levels, threshold)
    return 1.0 - prob if asks_below else prob

def _number(node):
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


def _split(comparison):
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

_REDUCERS = {
    "PSQL_PREDICT": lambda grid, levels, args: constructs.predict(grid, levels),
    "PSQL_LOW": lambda grid, levels, args: constructs.low(grid, levels, args[0]),
    "PSQL_HIGH": lambda grid, levels, args: constructs.high(grid, levels, args[0]),
    "PSQL_PROBABILITY": lambda grid, levels, args: _probability(grid, levels, args[0], args[1]),
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

def _target(node):
    if node.name.upper() == "PSQL_PROBABILITY":
        return _split(node.expressions[0])[0]
    return node.expressions[0]


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

    X, y = df[features], df[target]
    known = np.flatnonzero(y.notna().to_numpy())
    missing = np.flatnonzero(y.isna().to_numpy())
    if len(known) < 2:
        raise ValueError("leave-one-out needs at least two rows with a known value")

    rows = {}
    for i in known:
        rest = known[known != i]
        model = model_factory().fit(X.iloc[rest], y.iloc[rest])
        rows[i] = model.predict_quantiles(X.iloc[[i]])[0]
    if len(missing):
        model = model_factory().fit(X.iloc[known], y.iloc[known])
        for i, r in zip(missing, model.predict_quantiles(X.iloc[missing])):
            rows[i] = r

    grid = np.vstack([rows[i] for i in range(len(df))])
    return df[_RID].to_numpy(), grid, model.levels

def _reduce(node, grid, levels):
    name = node.name.upper()
    if name == "PSQL_PROBABILITY":
        _, number, asks_below = _split(node.expressions[0])
        args = [number, asks_below]
    else:
        args = [float(a.name) for a in node.expressions[1:]]
    return _REDUCERS[name](grid, levels, args)


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
        groups.setdefault(_target(node).name.lower(), []).append(node)

    table = ast.args["from_"].this
    alias = table.alias_or_name
    for i, group in enumerate(groups.values()):
        name = f"_psql_pred_{i}"
        column = _target(group[0]).name
        rids, grid, levels = _grid(con, table.name, column, model_factory)

        data = {"rid": rids}
        keys = [n.sql().upper() for n in group]
        slots = {}  # identical calls share one column
        for key, node in zip(keys, group):
            if key not in slots:
                slots[key] = f"v{len(slots)}"
                data[slots[key]] = _reduce(node, grid, levels)

        con.register(name, pd.DataFrame(data))
        join_condition = exp.column("rowid", table=alias).eq(exp.column("rid", table=name))
        ast.join(exp.to_table(name), on=join_condition, join_type="left", copy=False)
        for key, node in zip(keys, group):
            node.replace(exp.column(slots[key], table=name))

    return ast