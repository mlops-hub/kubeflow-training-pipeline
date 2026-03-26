import sqlite3
import pandas as pd
from pathlib import Path

LIVE_DIR = Path(__file__).resolve().parent.parent / "db"
LIVE_DIR.mkdir(parents=True, exist_ok=True)

LIVE_DB_PATH = LIVE_DIR / "live_data.db"


conn = sqlite3.connect(LIVE_DB_PATH)

df = pd.read_sql("SELECT * FROM live_data", conn)

# rename true_label
df = df.rename(columns={'true_label': 'target'})
#print('missing: ', df.isnull())

print('miss columns: ', df.isnull().sum())
df.fillna(1, inplace=True)

#print(df.dtypes)
print(df["annual_income"].min(), "-", df["annual_income"].max())
print(df.head(3))
print(df.columns.tolsit())
conn.close()