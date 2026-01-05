import time
from schema_index import build_schema_index, select_schema
from retriever import retrieve_rows
from groq_models import generate_answer
from config import GROQ_API_KEY, MODEL_PRIMARY

DB_PATH = "data/retail.db"
TABLE = "transactions"

build_schema_index(DB_PATH, TABLE)

while True:
    query = input("\nAsk a question (or type 'exit'): ")
    if query.lower() == "exit":
        break

    t0 = time.perf_counter()
    schema = select_schema(query)
    schema_time = (time.perf_counter() - t0) * 1000

    rows, sql_time = retrieve_rows(schema, DB_PATH, TABLE)

    answer, llm_time = generate_answer(
        query=query,
        context=rows,
        model_name=MODEL_PRIMARY,
        api_key=GROQ_API_KEY
    )

    print("\n--- Answer ---")
    print(answer)

    print("\n--- Metrics ---")
    print({
        "schema_ms": round(schema_time, 2),
        "sql_ms": round(sql_time, 2),
        "llm_ms": round(llm_time, 2),
        "rows_used": len(rows)
    })
