import os
import time
import sqlite3
import streamlit as st

from schema_index import build_schema_index, select_schema
from retriever import retrieve_rows
from groq_models import generate_answer

st.set_page_config(page_title="RAG Chatbot", layout="wide")

DB_PATH = "data/retail.db"
TABLE = "transactions"

st.title("RAG Chatbot")

api_key = st.text_input("Groq API Key", type="password")
if not api_key:
    st.info("Enter your Groq API key to continue")
    st.stop()

if not os.path.exists(DB_PATH):
    st.error("Database not found. Run ingestion first.")
    st.stop()

@st.cache_resource
def init_schema():
    build_schema_index(DB_PATH, TABLE)

init_schema()

model = st.selectbox(
    "Select model",
    ["llama-3.3-70b-versatile", "openai-oss-120b"]
)

query = st.text_input("Ask a question")

if query:
    t0 = time.perf_counter()
    schema = select_schema(query)
    schema_ms = (time.perf_counter() - t0) * 1000

    rows, sql_ms = retrieve_rows(schema, DB_PATH, TABLE)

    answer, llm_ms = generate_answer(
        query, rows, model, api_key
    )

    st.subheader("Answer")
    st.write(answer)

    st.subheader("Latency (ms)")
    st.json({
        "schema": round(schema_ms, 2),
        "sql": round(sql_ms, 2),
        "llm": round(llm_ms, 2),
        "rows": len(rows)
    })
