import pandas as pd
import sqlite3
import os

DB_PATH = "data/retail.db"
EXCEL_PATH = "data/online_retail_II.xlsx"
TABLE = "transactions"

os.makedirs("data", exist_ok=True)

df = pd.read_excel(EXCEL_PATH)
conn = sqlite3.connect(DB_PATH)
df.to_sql(TABLE, conn, if_exists="replace", index=False)
conn.close()

print(f"Ingested {len(df)} rows into {DB_PATH}")
