import re
from collections.abc import Callable

import duckdb
from openai import OpenAI
from sqlglot.dialects.dialect import Dialect
from sqlglot.errors import ParseError

from probsql.parser import find_constructs, parse_read_query
from probsql.planner import plan, source_table_name
from probsql.prompt import build_messages, build_retry_message, format_schema
from probsql.rewriter import _is_base_table

Messages = list[dict[str, str]]
LLM = Callable[[Messages], str]

class CannotAnswer(Exception):
    """The model said the question can't be answered within the limits."""

def clean_reply(text: str) -> str: # A helper pure function.
    """Strip whitespace and a surrounding ``` code fence, if there is one."""
    text = text.strip()
    fenced = re.match(r"^```[a-zA-Z]*\n(.*?)\n?```$", text, re.DOTALL)
    return fenced.group(1).strip() if fenced else text

def read_schema(con: duckdb.DuckDBPyConnection) -> dict[str, list[tuple[str, str]]]:
    """Read {table: [(column, type), ...]} for the real tables in the database."""
    rows = con.execute(
        "SELECT c.table_name, c.column_name, c.data_type "
        "FROM information_schema.columns c "
        "JOIN information_schema.tables t "
        "  ON c.table_name = t.table_name AND c.table_schema = t.table_schema "
        "WHERE t.table_type = 'BASE TABLE' "
        "ORDER BY c.table_name, c.ordinal_position"
    ).fetchall()
    schema = {}
    for table_name, column_name, data_type in rows:
        schema.setdefault(table_name, []).append((column_name, data_type))
    return schema

def openai_llm(model: str) -> LLM:
    """An LLM function backed by OpenAI's chat API."""
    client = OpenAI()

    def call(messages: Messages) -> str:
        response = client.chat.completions.create(model=model, messages=messages)
        return response.choices[0].message.content or ""

    return call

def check_sql(sql: str, con: duckdb.DuckDBPyConnection, dialect: str | type[Dialect]) -> None:
    """Raise if the engine would reject this query. Parses it and, if it uses a
    construct, runs the planner's checks. The model is not run."""
    ast = parse_read_query(sql, dialect)
    if find_constructs(ast):
        name = source_table_name(ast)
        plan(ast, name is not None and _is_base_table(con, name))

def question_to_sql(question: str, con: duckdb.DuckDBPyConnection, dialect: str | type[Dialect], llm: LLM, max_retires: int = 2) -> str:
    """Turn a question into SQL the engine accepts. If the reply is rejected,
    the error is sent back to the model, up to max_retries times, and then the
    last error is raised. Raises CannotAnswer if the model gives up."""
    messages = build_messages(question, format_schema(read_schema(con)))
    attempt = 0
    while True:
        sql = clean_reply(llm(messages))
        if sql.startswith("CANNOT_ANSWER"):
            raise CannotAnswer(sql.removeprefix("CANNOT_ANSWER").lstrip(": ").strip())
        try:
            check_sql(sql, con, dialect)
            return sql
        except (ValueError, ParseError) as error:
            if attempt >= max_retires:
                raise
            messages = messages + [
                {"role": "assistant", "content": sql},
                build_retry_message(sql, str(error)),
            ]
            attempt += 1