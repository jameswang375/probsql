# ProbSQL

Plain SQL can only return what is already stored in a table. It can sum, filter and join, but it cannot answer questions like 'what is the value likely to be, and how confident is that estimate?". To illustrate the usefulness of such a concept, people practicing in medicine could ask or query what the probability of this patient getting heart disease is, what their expected blood pressure would be, or the chance that their cholestorel would be above a certain number etc. You could achieve this through traditional machine learning means or using generative AI, but that involves models outside of the database. In addition, you risk the possibility that generative AI will [hallucinate and not get it right](https://priorlabs.ai/blog-posts/manifesto), and dedicating traditional machine learning efforts to achieve this would take a lot of effort to get it right.

ProbSQL dialect enables probabilistic querying by extending SQL with a few probabilistic keywords. A query can ask for a prediction, a range around it, or the chance that a value is above a threshold. The answers come back as ordinary columns, so filtering, sorting and grouping are all valid. This allows users to utilize these new functionalities generally only achievable through AI and/or Bayesian methods, purely by writing SQL. The underlying AI engine that makes this possible are tabular foundation models, which are designed for structured data. In short, think of them as GPTs for tables (although they aren't LLMs, but they are transfomer-based with neural nets/attention and such). 

```sql
SELECT home_id,
       PREDICTION OF cost AS estimate,
       PREDICTION LOW 0.9 OF cost AS low,
       PREDICTION HIGH 0.9 OF cost AS high
FROM homes
WHERE (PROBABILITY OF cost > 300000) > 0.8
```

A natural language layer and a chat interface sit on top, so a question like "Which homes are worth more than $300000?" is translated into SQL (text to SQL) like the above and run.

## Key Ideas

**Prediction with uncertainty.** A prediction is more useful when it says how likely different outcomes are. Instead of "$310k", a query can return a range ("probably between $260k and $370k") or a probability ("85% chance it is worth more than $300k").

**Tabular foundation model.** A model pretrained once on a very large number of synthetic tables that serves as our prior. Given a new table and performing **in-context learning**, it makes predictions without being trained on that table. This is significantly more flexible than traditional machine learning where you have to train again on datasets. Interested in learning more? Here is a nice read on [Tabular Foundation Models](https://tabularfoundationmodels.com/).

## The four constructs/new SQL syntax

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

DuckDB, or any other SQL engine, never sees a model, and the model never sees SQL. In between, the code rewrites each probabilistic call into a column of a small results table, joined back to the original table.
## Design decisions

**Functional core, imperative shell.** Decisions are made in pure functions that take plain data and return plain data: `plan`, `choose_columns`, `split_rows`, `fit_plan`, `reduce_calls`, `parse_read_query`, and the math in `constructs.py`. Side effects sit at the edge: reading tables from DuckDB, calling the model, registering tables, and calling the LLM. For example, whether the query's table is a real table is looked up by the shell and passed into `plan` as a plain boolean, so `plan` never touches the database. The model is allowed to be impure, since it sits in the shell. This keeps the logic easy to read on its own and means changes such as speeding up model fits are confined to one function.

**A swappable model interface.** `PredictiveModel` defines what any backend must do: take rows with known values, and return a quantile grid for new rows. The base class checks inputs and outputs (shape, finite values, sorted rows, matching columns), so every backend gets the same checks.

## Install and run

```bash
pip install -e ".[ui]"
export OPENAI_API_KEY="your-key"   # only needed for the natural language layer
streamlit run app.py
```

The first model use downloads the TabICL checkpoint, so it needs internet access.

## Some technologies used
- sqlglot
- TabICL (tabular foundation model)
- gpt-5-mini (OpenAI API)
- Polars and Pandas (DataFrames)
- DuckDB database