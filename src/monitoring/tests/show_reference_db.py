import sqlite3
import pandas as pd
from pathlib import Path

LIVE_DIR = Path(__file__).resolve().parent.parent / "db"
LIVE_DIR.mkdir(parents=True, exist_ok=True)

LIVE_DB_PATH = LIVE_DIR / "reference_data.db"


conn = sqlite3.connect(LIVE_DB_PATH)

df = pd.read_sql("SELECT * FROM reference_data", conn)

# rename target column
df = df.rename(columns={'attrition': 'target'})

df['prediciton'] = df['target'].copy()

df.fillna(0, inplace=True)

print(df.head())
print(df.tail())

conn.close()