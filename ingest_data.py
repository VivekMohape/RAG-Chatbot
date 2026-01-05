import pandas as pd
import sqlite3
import os

EXCEL_PATH = "data/online_retail_II.xlsx"
DB_PATH = "data/retail.db"
TABLE_NAME = "transactions"

os.makedirs("data", exist_ok=True)

print("Reading Excel...")
df = pd.read_excel(EXCEL_PATH)

print("Creating SQLite DB...")
conn = sqlite3.connect(DB_PATH)
df.to_sql(TABLE_NAME, conn, if_exists="replace", index=False)
conn.close()

print("Ingestion complete.")
print(f"Rows inserted: {len(df)}")
print(f"Database created at: {DB_PATH}")
