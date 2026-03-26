import sqlite3
import pandas as pd
from pathlib import Path

DB_DIR = Path(__file__).resolve().parent.parent / "db"
DB_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = DB_DIR / "prediciton_logs.db"

conn = sqlite3.connect(DB_PATH)

df = pd.read_sql("SELECT * FROM prediction_logs", conn)
print(df.head())

conn.close()

