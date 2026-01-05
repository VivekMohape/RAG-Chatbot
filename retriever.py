import sqlite3
from time import perf_counter

def retrieve_rows(columns):
    start = perf_counter()
    conn = sqlite3.connect("data/retail.db")
    query = f"SELECT {', '.join(columns)} FROM transactions LIMIT 200"
    rows = conn.execute(query).fetchall()
    conn.close()
    return rows, (perf_counter() - start) * 1000
