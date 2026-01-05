# Schema-Aware RAG Chatbot for Large Excel Datasets

## Overview

This project is a **proof of concept** for a **schema-aware Retrieval-Augmented Generation (RAG) chatbot** designed to work on **large Excel datasets with complex schemas**.

Instead of embedding or loading the entire dataset into an LLM context, the system **first identifies the most relevant subset of the schema (columns)** using NLP and embeddings. Only data from this reduced schema is retrieved and used to ground the LLM response.

This approach:

* Reduces token usage
* Improves response relevance
* Scales better for wide tabular datasets
* Enables measurable latency and retrieval metrics

The PoC is implemented as a **CLI-based system** for robustness, reproducibility, and clarity.

---

## Key Idea (What Makes This Different)

Traditional RAG over tabular data often:

* Embeds entire rows or tables
* Loads unnecessary columns
* Suffers from noise, latency, and hallucinations

This PoC introduces a **schema-first pipeline**:

```
User Query
   ↓
Semantic Schema Selection (NLP + embeddings)
   ↓
Targeted SQL Retrieval (only relevant columns)
   ↓
LLM Answer (grounded on filtered data)
```

The LLM never sees irrelevant columns.

---

## Architecture

```
Excel Dataset (Online Retail II)
        ↓  (one-time)
SQLite Database
        ↓
Schema Index (FAISS + embeddings)
        ↓
User Query (CLI)
        ↓
Relevant Schema Selection
        ↓
SQL Retrieval (filtered columns)
        ↓
Groq-hosted LLM (LLaMA 70B / OSS 120B)
        ↓
Answer + Latency Metrics
```

---

## NLP & AI Concepts Used

### 1. Semantic Schema Understanding

* Column names are enriched with semantic descriptions
* Embeddings are generated using **Sentence Transformers**
* FAISS is used for fast similarity search over schema

### 2. Query-to-Schema Matching

* User query is embedded
* Top-K most relevant columns are selected
* Prevents full-table scanning

### 3. Retrieval-Augmented Generation (RAG)

* SQL retrieval is constrained to selected columns
* Retrieved rows are passed as grounding context
* LLM is instructed to answer **strictly from data**

### 4. Deterministic LLM Inference

* Temperature = 0
* Predictable answers
* Stable latency measurements

---

## Models Used

### Embedding Model (Open Source)

* **BAAI/bge-m3**

  * Strong performance on semantic similarity
  * Works well for short text (schema / column names)
  * Fully open source

### LLMs (via Groq API)

* **llama-3.3-70b-versatile**
* **openai-oss-120b**

The system supports **easy model switching** for comparison.

---

## Dataset

* **Online Retail II Dataset**
* Source: Kaggle
* Contains transactional retail data with multiple columns
* Large enough to demonstrate schema complexity and performance impact

The dataset is ingested **once** into SQLite to avoid repeated Excel parsing.

---

## Repository Structure

```
rag-schema-chatbot/
│
├── data/
│   ├── online_retail_II.xlsx
│   └── retail.db
│
├── ingest.py
├── schema_index.py
├── schema_enrich.py
├── retriever.py
├── groq_models.py
├── query_runner.py
│
├── config.py
├── requirements.txt
└── README.md
```

---

## Setup Instructions

### 1. Clone Repository

```bash
git clone <repo-url>
cd rag-schema-chatbot
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Add Dataset

Place the Excel file at:

```
data/online_retail_II.xlsx
```

---

## One-Time Data Ingestion

Convert Excel → SQLite:

```bash
python ingest.py
```

This creates:

```
data/retail.db
```

---

## Configure API Key

Edit `config.py`:

```python
GROQ_API_KEY = "your_groq_api_key"
MODEL_PRIMARY = "llama-3.3-70b-versatile"
MODEL_SECONDARY = "openai-oss-120b"
```

For production, this would be replaced by environment variables or a secrets manager.

---

## Run the Chatbot (CLI)

```bash
python query_runner.py
```

Example interaction:

```
Ask a question: What is the total revenue by country?

--- Answer ---
United Kingdom has the highest total revenue...

--- Metrics ---
{
  "schema_ms": 18.4,
  "sql_ms": 22.7,
  "llm_ms": 410.2,
  "rows_used": 500
}
```

---

## Metrics Reported

For every query, the system reports:

* Schema selection latency (ms)
* SQL retrieval latency (ms)
* LLM inference latency (ms)
* Number of rows used for grounding

This makes performance **observable and debuggable**.

---

## Why CLI Instead of UI?

The initial prototype used Streamlit, but was intentionally refactored to CLI to:

* Avoid UI-related instability
* Improve reproducibility
* Enable deterministic benchmarking
* Focus on core AI engineering logic



## Limitations 

* Accuracy depends on column naming quality
* SQLite used for simplicity (can be replaced with Postgres)
* No automated evaluation metrics yet

---

## Future Improvements

* Hybrid retriever (BM25 + embeddings)
* Query caching
* Automated schema selection evaluation
* FastAPI service wrapper
* Role-based data access
* Batch benchmarking scripts


* write an evaluation section
* prepare interview Q&A based on this PoC
