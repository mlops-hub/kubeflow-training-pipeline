import sqlite3
import pandas as pd
from pathlib import Path

LIVE_DIR = Path(__file__).resolve().parent.parent / "db"
LIVE_DIR.mkdir(parents=True, exist_ok=True)

LIVE_DB_PATH = LIVE_DIR / "reference_data.db"


conn = sqlite3.connect(LIVE_DB_PATH)
query = "SELECT * FROM reference_data"
df = pd.read_sql(query, conn)
conn.close()

df['event_timestamp'] = pd.to_datetime(df['event_timestamp'])
df_sampled = df.sample(n=30, random_state=42)

print(df_sampled.describe())

