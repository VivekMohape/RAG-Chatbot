import os
import time
import sqlite3
import streamlit as st

from schema_index import build_schema_index, select_schema
from retriever import retrieve_rows
from groq_models import generate_answer

# ----------------------------
# basic config
# ----------------------------
st.set_page_config(
    page_title="Retail Schema Aware Chatbot",
    layout="wide"
)

DB_PATH = "data/retail.db"
TABLE_NAME = "transactions"

# ----------------------------
# helpers
# ----------------------------
def db_exists():
    return os.path.exists(DB_PATH)

def get_db_row_count():
    conn = sqlite3.connect(DB_PATH)
    count = conn.execute(
        f"SELECT COUNT(*) FROM {TABLE_NAME}"
    ).fetchone()[0]
    conn.close()
    return count

# ----------------------------
# schema index init (cached)
# ----------------------------
@st.cache_resource
def init_schema_index():
    build_schema_index(db_path=DB_PATH, table_name=TABLE_NAME)

# ----------------------------
# UI
# ----------------------------
st.title("Retail Schema Aware Chatbot")

st.markdown(
    "Schema-first RAG chatbot over large Excel data "
    "using Groq hosted large language models."
)

# ----------------------------
# dataset upload
# ----------------------------
st.subheader("Dataset")

uploaded_file = st.file_uploader(
    "Upload Online Retail Excel file",
    type=["xlsx"]
)

if uploaded_file:
    os.makedirs("data", exist_ok=True)
    with open("data/online_retail_II.xlsx", "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("File uploaded. Restart app to ingest if required.")

# ----------------------------
# database status
# ----------------------------
if not db_exists():
    st.warning(
        "SQLite database not found. "
        "Run ingestion script before using chatbot."
    )
    st.stop()

st.success(f"Database loaded with {get_db_row_count():,} rows")

# ----------------------------
# build schema index
# ----------------------------
init_schema_index()

# ----------------------------
# model selection
# ----------------------------
st.subheader("Model Selection")

model_name = st.selectbox(
    "Choose Groq model",
    [
        "llama-3.3-70b-versatile",
        "openai-oss-120b"
    ]
)

# ----------------------------
# query input
# ----------------------------
st.subheader("Ask a Question")

query = st.text_input(
    "Enter your question about the retail data"
)

# ----------------------------
# session metrics
# ----------------------------
if "metrics" not in st.session_state:
    st.session_state.metrics = []

# ----------------------------
# run query
# ----------------------------
if query:
    # schema selection
    t0 = time.perf_counter()
    schema_cols = select_schema(query)
    schema_time = (time.perf_counter() - t0) * 1000

    # data retrieval
    rows, sql_time = retrieve_rows(
        schema_cols,
        db_path=DB_PATH,
        table_name=TABLE_NAME
    )

    # llm answer
    answer, llm_time = generate_answer(
        query,
        rows,
        model_name
    )

    # record metrics
    st.session_state.metrics.append({
        "schema_ms": schema_time,
        "sql_ms": sql_time,
        "llm_ms": llm_time,
        "rows": len(rows)
    })

    # ----------------------------
    # output
    # ----------------------------
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Selected Schema")
        st.write(schema_cols)

        st.subheader("Retrieved Rows")
        st.write(f"{len(rows)} rows used for answer")

    with col2:
        st.subheader("Chatbot Answer")
        st.write(answer)

    # ----------------------------
    # metrics
    # ----------------------------
    st.subheader("Performance Metrics")

    st.metric("Schema Selection (ms)", f"{schema_time:.1f}")
    st.metric("SQL Retrieval (ms)", f"{sql_time:.1f}")
    st.metric("LLM Generation (ms)", f"{llm_time:.1f}")

    st.line_chart(st.session_state.metrics)

# ----------------------------
# footer
# ----------------------------
st.caption(
    "This PoC demonstrates schema-aware retrieval augmented generation "
    "with measurable latency and model comparison."
)
