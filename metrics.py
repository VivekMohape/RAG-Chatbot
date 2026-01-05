def log_metrics(state, schema_ms, sql_ms, llm_ms, rows):
    state.append({
        "schema_ms": schema_ms,
        "sql_ms": sql_ms,
        "llm_ms": llm_ms,
        "rows": rows
    })
