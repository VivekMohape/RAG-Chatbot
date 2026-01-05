import os
import time
import sqlite3
import streamlit as st

from schema_index import build_schema_index, select_schema
from retriever import retrieve_rows
from groq_models import generate_answer


st.set_page_config(
    page_title="RAG Chatbot",
    layout="wide"
)

DB_PATH = "data/retail.db"
TABLE_NAME = "transactions"
UPLOAD_PATH = "data/online_retail_II.xlsx"


def db_exists():
    return os.path.exists(DB_PATH)

def get_row_count():
    conn = sqlite3.connect(DB_PATH)
    count = conn.execute(
        f"SELECT COUNT(*) FROM {TABLE_NAME}"
    ).fetchone()[0]
    conn.close()
    return count

@st.cache_resource
def init_schema_index():
    build_schema_index(
        db_path=DB_PATH,
        table_name=TABLE_NAME
    )


st.title("RAG Chatbot")

st.markdown(
    "Schema-first RAG chatbot over large Excel datasets "
    "using Groq hosted large language models."
)


st.subheader("Dataset")

uploaded_file = st.file_uploader(
    "Upload Online Retail Excel file",
    type=["xlsx"]
)

if uploaded_file:
    os.makedirs("data", exist_ok=True)
    with open(UPLOAD_PATH, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success(
        "File uploaded successfully. "
        "Run ingestion script if database is not created."
    )


if not db_exists():
    st.warning(
        "SQLite database not found.\n\n"
        "Please run the ingestion script to create data/retail.db "
        "before using the chatbot."
    )
    st.stop()

st.success(f"Database ready with {get_row_count():,} rows")


init_schema_index()


st.subheader("Model Selection")

model_name = st.selectbox(
    "Choose Groq model",
    [
        "llama-3.3-70b-versatile",
        "openai-oss-120b"
    ]
)


st.subheader("Ask a Question")

query = st.text_input(
    "Enter a question about the retail dataset"
)


if "metrics" not in st.session_state:
    st.session_state.metrics = []


if query:
    # schema selection
    t0 = time.perf_counter()
    schema_cols = select_schema(query)
    schema_ms = (time.perf_counter() - t0) * 1000

    # sql retrieval
    rows, sql_ms = retrieve_rows(
        schema_cols,
        db_path=DB_PATH,
        table_name=TABLE_NAME
    )

    # llm generation
    answer, llm_ms = generate_answer(
        query,
        rows,
        model_name
    )

    # record metrics
    st.session_state.metrics.append({
        "schema_ms": schema_ms,
        "sql_ms": sql_ms,
        "llm_ms": llm_ms,
        "rows": len(rows)
    })


    left, right = st.columns(2)

    with left:
        st.subheader("Selected Schema Columns")
        st.write(schema_cols)

        st.subheader("Retrieved Rows")
        st.write(f"{len(rows)} rows used")

    with right:
        st.subheader("Chatbot Answer")
        st.write(answer)


    st.subheader("Performance Metrics")

    c1, c2, c3 = st.columns(3)
    c1.metric("Schema Selection (ms)", f"{schema_ms:.1f}")
    c2.metric("SQL Retrieval (ms)", f"{sql_ms:.1f}")
    c3.metric("LLM Generation (ms)", f"{llm_ms:.1f}")

    st.line_chart(st.session_state.metrics)


st.caption(
    "This proof of concept demonstrates schema-aware "
    "retrieval augmented generation with measurable latency "
    "and model comparison."
)
