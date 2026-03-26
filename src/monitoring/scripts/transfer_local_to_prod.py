import sqlite3
import pandas as pd
from sqlalchemy import create_engine
import json
import os
from dotenv import load_dotenv

load_dotenv()

# Connect SQLite
sqlite_conn = sqlite3.connect("live_data.db")
df = pd.read_sql_query("SELECT employee_id, features, prediction FROM live_data", sqlite_conn)

# Ensure features column is dict (JSON)
df['features'] = df['features'].apply(lambda x: json.loads(x) if isinstance(x, str) else x)

# Connect Postgres
POSTGRES_URI = os.environ.get(
    "POSTGRES_URI_EXTERNAL",
    "postgresql+psycopg2://feast:feast@68.183.87.245:30032/feast"
)
engine = create_engine(POSTGRES_URI)

# Write to Postgres
df.to_sql("live_data", engine, if_exists="append", index=False)
print("SQLite -> Postgres migration done!")