import duckdb
import polars as pl
import streamlit as st

from probsql.assistant import CannotAnswer, LLM, openai_llm, question_to_sql
from probsql.dialect import ProbSQL
from probsql.engine import Engine
from probsql.models.tabicl import TabICLModel

MODEL = "gpt-5-mini"  # change to any model your key can use


@st.cache_resource
def load_engine() -> Engine:
    """Build the demo database and engine once, not on every rerun."""
    con = duckdb.connect()
    con.execute("CREATE TABLE homes (home_id INTEGER, cost DOUBLE, sqft DOUBLE, beds INTEGER)")
    con.execute("""
        INSERT INTO homes VALUES
            (1, 250000, 900, 2), (2, 410000, 1400, 3), (3, 320000, 1100, 3),
            (4, 575000, 2000, 4), (5, 198000, 750, 1), (6, 365000, 1250, 3),
            (7, 470000, 1600, 4), (8, 300000, 1000, 2), (9, NULL, 1200, 3),
            (10, 520000, 1800, 4)
    """)
    return Engine(con=con, model_factory=lambda: TabICLModel(device="cpu"))


@st.cache_resource
def load_llm() -> LLM:
    return openai_llm(MODEL)


def answer(question: str, engine: Engine, llm: LLM) -> dict:
    try:
        note, sql, after = question_to_sql(question, engine.con, ProbSQL, llm)
    except CannotAnswer as reason:
        return {"note": None, "sql": None, "table": None, "after": None, "message": f"Cannot answer: {reason}"}
    except Exception as error:
        return {"note": None, "sql": None, "table": None, "after": None, "message": f"Could not write a valid query: {error}"}
    try:
        return {"note": note, "sql": sql, "table": engine.query(sql), "after": after, "message": None}
    except Exception as error:
        return {"note": note, "sql": sql, "table": None, "after": after, "message": f"The query failed: {error}"}


def show(entry: dict) -> None:
    """Draw one assistant reply."""
    if entry["note"]:
        st.write(entry["note"])
    if entry["sql"]:
        st.code(entry["sql"], language="sql")
    if entry["message"]:
        st.warning(entry["message"])
    table: pl.DataFrame | None = entry["table"]
    if table is not None:
        st.dataframe(table)
    if entry["after"]:
        st.write(entry["after"])


st.title("probsql demo")
st.caption("Ask questions about the homes table. Predictions come from TabICL.")

engine = load_engine()
llm = load_llm()

if "history" not in st.session_state:
    st.session_state.history = []

for item in st.session_state.history:
    with st.chat_message("user"):
        st.write(item["question"])
    with st.chat_message("assistant"):
        show(item["entry"])

question = st.chat_input("Ask a question")
if question:
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            entry = answer(question, engine, llm)
        show(entry)
    st.session_state.history.append({"question": question, "entry": entry})