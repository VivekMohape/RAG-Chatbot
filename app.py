import os
import time
import sqlite3
import streamlit as st

from schema_index import build_schema_index, select_schema
from retriever import retrieve_rows
from groq_models import generate_answer

st.set_page_config(page_title="RAG Chatbot", layout="wide")

DB_PATH = "data/retail.db"
TABLE_NAME = "transactions"

st.title("RAG Chatbot")

api_key = st.text_input("Groq API Key", type="password")

if not api_key:
    st.info("Please enter your Groq API key to continue")
    st.stop()

def db_exists():
    return os.path.exists(DB_PATH)

def get_row_count():
    conn = sqlite3.connect(DB_PATH)
    count = conn.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}").fetchone()[0]
    conn.close()
    return count

if not db_exists():
    st.error("Database not found. Please ingest dataset first.")
    st.stop()

st.success(f"Database loaded with {get_row_count():,} rows")

@st.cache_resource
def init_schema():
    build_schema_index(db_path=DB_PATH, table_name=TABLE_NAME)

init_schema()

model_name = st.selectbox(
    "Select model",
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

    st.subheader("Answer")
    st.write(answer)

    st.subheader("Latency")
    st.write({
        "schema_ms": round(schema_ms, 2),
        "sql_ms": round(sql_ms, 2),
        "llm_ms": round(llm_ms, 2),
        "rows": len(rows)
    })
