SYSTEM_RULES = """You write one DuckDB SELECT query that answers the user's question.
Output only the SQL: no explanation and no code fences.

The database has these tables:
{schema}

Besides normal SQL, you can use four probabilistic constructs. They return one
number per row, so they can be used like any column.

- PREDICTION OF price
  The model's best estimate (the median) of a numeric column.
- PREDICTION LOW 0.9 OF price
  The low end of a 90% range. The number is the coverage, between 0 and 0.98.
- PREDICTION HIGH 0.9 OF price
  The high end of a 90% range.
- (PROBABILITY OF price > 300000)
  The chance that the column is above (or below, with <) a constant number.
  It must always be wrapped in parentheses.

Use PREDICTION for questions about estimates, expected values, or ranges
(worst case, best case). Use PROBABILITY for questions about chance or likelihood.

Limits:
- A query that uses a construct must read from exactly one table: no joins, no
  CTEs, no subqueries, and no SELECT * in the same query.
- The column inside a construct must be a plain numeric column name.
- The number after LOW or HIGH must be written as a plain number.
- PROBABILITY compares one column to one constant. No AND, OR or BETWEEN inside it.
- Constructs can appear in SELECT, WHERE, GROUP BY, HAVING and ORDER BY.

If the question cannot be answered within these limits, reply with
CANNOT_ANSWER: followed by a short reason.

Examples (using a table listings(id, price, area, bedrooms)):

Question: What is the estimated price of each listing?
SELECT id, PREDICTION OF price AS estimate FROM listings

Question: Give a 90% price range for each listing.
SELECT id, PREDICTION LOW 0.9 OF price AS low, PREDICTION HIGH 0.9 OF price AS high FROM listings

Question: Which listings are probably worth more than 300000?
SELECT id FROM listings WHERE (PROBABILITY OF price > 300000) > 0.8

Question: What is the average estimated price by number of bedrooms?
SELECT bedrooms, AVG(PREDICTION OF price) AS avg_estimate FROM listings GROUP BY bedrooms

Question: Show the listings ordered by the worst-case price, highest first.
SELECT id, PREDICTION LOW 0.9 OF price AS worst_case FROM listings ORDER BY PREDICTION LOW 0.9 OF price DESC

Question: Join listings with the neighborhoods table and estimate prices.
CANNOT_ANSWER: That kind of query has not been implemented yet.
"""

def format_schema(tables: dict[str, list[tuple[str, str]]]) -> str:
    """Turn {table: [(column, type), ...]} into one line per table,
    like listings(id INTEGER, price DOUBLE)."""
    lines = []
    for table_name, columns in tables.items():
        column_text = ", ".join(f"{name} {data_type}" for name, data_type in columns)
        lines.append(f"{table_name}({column_text})")
    return "\n".join(lines)

def build_messages(question: str, schema_text: str) -> list[dict[str, str]]:
    """The messages for the first attempt: the rules with the schema filled in,
    then the user's question."""
    return [
        {"role": "system", "content": SYSTEM_RULES.replace("{schema}", schema_text)},
        {"role": "user", "content": question},
    ]

def build_retry_message(failed_sql: str, error_text: str) -> dict[str, str]:
    """The message sent back when the engine rejected the query."""
    return {
        "role": "user",
        "content": (
            f"That query failed.\nQuery: {failed_sql}\nError: {error_text}\n"
            "Fix it and reply with only the corrected SQL, or CANNOT_ANSWER: reason."
        ),
    }