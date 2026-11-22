import numpy as np
import polars as pl

def choose_columns(columns: list[str], column_name: str, table_name: str, rid_column: str) -> tuple[str, list[str]]:
    """Find the target column (case-insensitive) and use every other column,
    except the row id, as features. Returns (target, features)."""
    target = next((c for c in columns if c.lower() == column_name.lower()), None)
    if target is None or target == rid_column:
        raise ValueError(f"column {column_name!r} not found in {table_name!r}")
    features = [c for c in columns if c not in (rid_column, target)]
    if not features:
        raise ValueError("the table has no other columns to predict from")
    return target, features

def split_rows(y: pl.Series, target_name: str) -> tuple[np.ndarray, np.ndarray]:
    """Positions of rows with a known target and rows with a missing one."""

    if not y.dtype.is_numeric():
        raise ValueError(f"PSQL_* needs a numeric column, but {target_name!r} is not")
    known = np.flatnonzero(y.is_not_null().to_numpy())
    missing = np.flatnonzero(y.is_null().to_numpy())
    if len(known) < 2:
        raise ValueError("leave-one-out needs at least two rows with a known value")
    return known, missing

def fit_plan(known: np.ndarray, missing: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """List of (context rows, rows to predict). Each known row is predicted from
    all the other known rows, so its own value is never in its context. Rows with
    no value are all predicted from every known row."""

    jobs = [(known[known != i], np.array([i])) for i in known]
    if len(missing):
        jobs.append((known, missing))
    return jobs