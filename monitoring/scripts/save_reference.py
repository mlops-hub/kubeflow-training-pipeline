import sqlite3
import pandas as pd
from pathlib import Path

DB_DIR = Path(__file__).resolve().parent.parent / "db"
DB_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = DB_DIR / "reference_data.db"
# print(DB_PATH)

def save_reference_data(df: pd.DataFrame):
    conn = sqlite3.connect(DB_PATH)
    df.to_sql("reference_data", conn, if_exists="replace", index=False)
    conn.close()
    