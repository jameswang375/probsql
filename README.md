# ProbSQL

Plain SQL can only return what is already stored in a table. It can sum, filter and join, but it cannot say what is probably true about something that is not there, such as a missing value, or how confident that answer is. You could achieve this through traditional machine learning, but that involves models outside of the database.

probsql dialect enables probabilistic querying by extending SQL with a few probabilistic keywords. A query can ask for a prediction, a range around it, or the chance that a value is above a threshold. The answers come back as ordinary columns, so filtering, sorting and grouping are all valid.

```sql
SELECT home_id,
       PREDICTION OF cost AS estimate,
       PREDICTION LOW 0.9 OF cost AS low,
       PREDICTION HIGH 0.9 OF cost AS high
FROM homes
WHERE (PROBABILITY OF cost > 300000) > 0.8
```

A natural language layer and a chat interface sit on top, so a question like "Which homes are probably worth more than 300000?" is translated into SQL (text to SQL) like the above and run.

## Key Ideas

**Prediction with uncertainty.** A prediction is more useful with a range. "About $310k, probably between $260k and $370k" supports better decisions than "$310k".

**Tabular foundation model.** A neural network pretrained once on a very large number of synthetic tables. Given a new table, it makes predictions without being trained on that table. Interested in learning more? Here is a nice read on Tabular Foundation Models [Tabular Foundation Models](https://tabularfoundationmodels.com/).

**In-context learning.** The model does not update its weights on your data. The table is passed in as input at prediction time, and the model infers the missing values from the rows it can see. The cost of prediction is therefore paid on every call, not once at training time.

## The four constructs and new SQL syntax

Each has a keyword form and an equivalent function form. Both work.

| Keyword form | Function form | Returns |
|---|---|---|
| `PREDICTION OF price` | `PSQL_PREDICT(price)` | The median prediction |
| `PREDICTION LOW 0.9 OF price` | `PSQL_LOW(price, 0.9)` | The low end of a 90% range |
| `PREDICTION HIGH 0.9 OF price` | `PSQL_HIGH(price, 0.9)` | The high end of a 90% range |
| `(PROBABILITY OF price > 300000)` | `PSQL_PROBABILITY(price > 300000)` | The chance the value is above (or, with `<`, below) the number |

They can be used in `SELECT`, `WHERE`, `GROUP BY`, `HAVING` and `ORDER BY`. `PROBABILITY OF` must be wrapped in parentheses.


```
question (optional) -> LLM writes SQL
SQL text -> parse -> find PSQL_* calls -> plan -> run the model
         -> turn the grid into columns -> rewrite the query -> DuckDB runs plain SQL
```

DuckDB, or any other SQL engine, never sees a model, and the model never sees SQL. In between, the code rewrites each probabilistic call into a column of a small results table, joined back to the original table on `rowid`.

## Design decisions

**Functional core, imperative shell.** Decisions are made in pure functions that take plain data and return plain data: `plan`, `choose_columns`, `split_rows`, `fit_plan`, `reduce_calls`, `parse_read_query`, and the math in `constructs.py`. Side effects sit at the edge: reading tables from DuckDB, calling the model, registering tables, and calling the LLM. For example, whether the query's table is a real table is looked up by the shell and passed into `plan` as a plain boolean, so `plan` never touches the database. The model is allowed to be impure, since it sits in the shell. This keeps the logic easy to read on its own and means changes such as speeding up model fits are confined to one function (`fit_plan`).

**A swappable model interface.** `PredictiveModel` defines what any backend must do: take rows with known values, and return a quantile grid for new rows. The base class checks inputs and outputs (shape, finite values, sorted rows, matching columns), so every backend gets the same checks.

## Install and run

```bash
pip install -e ".[ui]"
export OPENAI_API_KEY="your-key"   # only needed for the natural language layer
streamlit run app.py
```

The first model use downloads the TabICL checkpoint, so it needs internet access. 