import os
import time
import sqlite3
import streamlit as st

from schema_index import build_schema_index, select_schema
from retriever import retrieve_rows
from groq_models import generate_answer

st.set_page_config(page_title="Retail Schema Aware Chatbot", layout="wide")

DB_PATH = "data/retail.db"
TABLE_NAME = "transactions"

def db_exists():
    return os.path.exists(DB_PATH)

def get_row_count():
    conn = sqlite3.connect(DB_PATH)
    count = conn.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}").fetchone()[0]
    conn.close()
    return count

st.title("Retail Schema Aware Chatbot")

api_key = st.text_input("Groq API Key", type="password")
if not api_key:
    st.stop()

if not db_exists():
    st.error("Database not found. Ingest dataset first.")
    st.stop()

st.success(f"Database loaded with {get_row_count():,} rows")

@st.cache_resource
def init_schema():
    build_schema_index(db_path=DB_PATH, table_name=TABLE_NAME)

init_schema()

model_name = st.selectbox(
    "Model",
    ["llama-3.3-70b-versatile", "openai-oss-120b"]
)

query = st.text_input("Ask a question")

if "metrics" not in st.session_state:
    st.session_state.metrics = []

if query:
    t0 = time.perf_counter()
    schema_cols = select_schema(query)
    schema_ms = (time.perf_counter() - t0) * 1000

    rows, sql_ms = retrieve_rows(
        schema_cols,
        db_path=DB_PATH,
        table_name=TABLE_NAME
    )

    answer, llm_ms = generate_answer(
        query=query,
        context=rows,
        model_name=model_name,
        api_key=api_key
    )

    st.session_state.metrics.append({
        "schema_ms": schema_ms,
        "sql_ms": sql_ms,
        "llm_ms": llm_ms,
        "rows": len(rows)
    })

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Schema")
        st.write(schema_cols)
        st.subheader("Rows Used")
        st.write(len(rows))

    with c2:
        st.subheader("Answer")
        st.write(answer)

    st.subheader("Latency Metrics")
    m1, m2, m3 = st.columns(3)
    m1.metric("Schema ms", f"{schema_ms:.1f}")
    m2.metric("SQL ms", f"{sql_ms:.1f}")
    m3.metric("LLM ms", f"{llm_ms:.1f}")

    st.line_chart(st.session_state.metrics)
