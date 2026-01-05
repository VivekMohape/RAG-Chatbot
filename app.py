import os
import time
import sqlite3
import streamlit as st
import pandas as pd

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

os.makedirs("data", exist_ok=True)

if not os.path.exists(DB_PATH):
    st.warning("Database not found. Upload Excel file to ingest.")

    uploaded_file = st.file_uploader(
        "Upload Online Retail Excel file",
        type=["xlsx"]
    )

    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        conn = sqlite3.connect(DB_PATH)
        df.to_sql(TABLE, conn, if_exists="replace", index=False)
        conn.close()

        st.success("Ingestion complete. Please refresh the app.")
        st.stop()

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

if "metrics" not in st.session_state:
    st.session_state.metrics = []

if query:
    t0 = time.perf_counter()
    schema_cols = select_schema(query)
    schema_ms = (time.perf_counter() - t0) * 1000

    rows, sql_ms = retrieve_rows(schema_cols, DB_PATH, TABLE)

    answer, llm_ms = generate_answer(
        query=query,
        context=rows,
        model_name=model,
        api_key=api_key
    )

    st.subheader("Answer")
    st.write(answer)

    st.subheader("Schema Used")
    st.write(schema_cols)

    st.subheader("Latency Metrics (ms)")
    st.json({
        "schema": round(schema_ms, 2),
        "sql": round(sql_ms, 2),
        "llm": round(llm_ms, 2),
        "rows": len(rows)
    })
