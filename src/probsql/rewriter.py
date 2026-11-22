import pandas as pd
import numpy as np
from sqlglot import exp

from probsql import constructs
from probsql.parser import find_constructs
from probsql.planner import _split, plan, source_table_name
from probsql.context import choose_columns, fit_plan, split_rows

_RID = "_psql_rid"

def _probability(grid, levels, threshold, asks_below):
    """Prob here means probability above a certain quantity. 
    So return prob as is if you want probability above a certain value, otherwise return the remaining probability (1 - prob), 
    which would be the probability below a certain value."""

    prob = constructs.prob_exceeds(grid, levels, threshold)
    return 1.0 - prob if asks_below else prob

_CONSTRUCT_LOGIC = {
    "PSQL_PREDICT": lambda grid, levels, args: constructs.predict(grid, levels),
    "PSQL_LOW": lambda grid, levels, args: constructs.low(grid, levels, args[0]),
    "PSQL_HIGH": lambda grid, levels, args: constructs.high(grid, levels, args[0]),
    "PSQL_PROBABILITY": lambda grid, levels, args: _probability(grid, levels, args[0], args[1]),
}

def _quote(name):
    return '"' + name.replace('"', '""') + '"'

def _is_base_table(con, name):
    rows = con.execute(
        "SELECT 1 FROM information_schema.tables "
        "WHERE lower(table_name) = lower(?) AND table_type = 'BASE TABLE'",
        [name],
    ).fetchall()
    return bool(rows)

def compute_grid(con, table_name, column_name, model_factory):
    """Run the model on every row. Returns (rowids, quantile grid, levels)."""
    df = con.sql(f"SELECT rowid AS {_RID}, * FROM {_quote(table_name)}").df()
    target, features = choose_columns(list(df.columns), column_name, table_name, _RID)
    X, y = df[features], df[target]
    known, missing = split_rows(y, target)

    rows = {}
    for context, query in fit_plan(known, missing):
        model = model_factory().fit(X.iloc[context], y.iloc[context])
        for i, r in zip(query, model.predict_quantiles(X.iloc[query])):
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
    return _CONSTRUCT_LOGIC[name](grid, levels, args)

def _result_table(i):
    """Name of the results table for the i-th target column."""
    return f"_psql_pred_{i}"

def reduce_calls(group, grid, levels):
    """Compute the result columns for one group of PSQL_* calls.

    Takes the model's grid for one target column and returns two things:
    columns: the computed values, named v0, v1, ... with one number per row.
    names: for each call in group.nodes, the name of the column it reads.

    If the same call appears twice, it is computed once and both share a column,
    so names can repeat.
    """
    columns = {}
    slots = {}
    names = []

    for node in group.nodes:
        key = node.sql().upper()
        if key not in slots:
            slots[key] = f"v{len(slots)}"
            columns[slots[key]] = _reduce(node, grid, levels)
        names.append(slots[key])

    return columns, names

def apply(ast, work, names_per_group):
    """Edit ast so each PSQL_* call reads its column from a joined results table.

    names_per_group[i] is the list of column names for work.groups[i].nodes.
    Edits ast in place and returns it. The nodes in work belong to this ast, so
    the caller must pass the same tree it gave to plan().
    """
    for i, (group, names) in enumerate(zip(work.groups, names_per_group)):
        table_name = _result_table(i)
        join_condition = exp.column("rowid", table=work.alias).eq(
            exp.column("rid", table=table_name)
        )
        ast.join(exp.to_table(table_name), on=join_condition, join_type="left", copy=False)
        for node, column in zip(group.nodes, names):
            node.replace(exp.column(column, table=table_name))
    return ast

def rewrite(ast, con, model_factory):
    """Return a copy of ast with each PSQL_* call replaced by a column from a
    registered table of results, joined back on rowid. The model runs once per
    target column."""
    ast = ast.copy()
    if not find_constructs(ast):
        return ast

    name = source_table_name(ast)
    work = plan(ast, name is not None and _is_base_table(con, name))

    names_per_group = []
    for i, group in enumerate(work.groups):
        rids, grid, levels = compute_grid(con, work.table, group.column, model_factory)
        columns, names = reduce_calls(group, grid, levels)
        con.register(_result_table(i), pd.DataFrame({"rid": rids, **columns}))
        names_per_group.append(names)

    return apply(ast, work, names_per_group)