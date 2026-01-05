import streamlit as st
from schema_index import select_schema
from retriever import retrieve_rows
from groq_models import generate_answer
from metrics import log_metrics

st.title("RAG Chatbot")

model_choice = st.selectbox(
    "Select model",
    ["llama-3.3-70b-versatile", "openai-oss-120b"]
)

query = st.text_input("Ask a question")

if "metrics" not in st.session_state:
    st.session_state.metrics = []

if query:
    schema_cols = select_schema(query)
    rows, sql_time = retrieve_rows(schema_cols)

    answer, llm_time = generate_answer(
        query, rows, model_choice
    )

    log_metrics(
        st.session_state.metrics,
        schema_ms=0,
        sql_ms=sql_time,
        llm_ms=llm_time,
        rows=len(rows)
    )

    st.subheader("Answer")
    st.write(answer)

    st.subheader("Performance Metrics")
    st.line_chart(st.session_state.metrics)
