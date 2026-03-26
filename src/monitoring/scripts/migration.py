# migrate_sqlite_to_postgres.py
import sqlite3
import pandas as pd
from sqlalchemy import create_engine, Table, Column, Integer, String, MetaData
from sqlalchemy.dialects.postgresql import JSONB
import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

LIVE_DIR = Path(__file__).resolve().parent.parent / "db"
LIVE_DIR.mkdir(parents=True, exist_ok=True)

# SQLite local DB
LIVE_DB_PATH = LIVE_DIR / "live_data.db"

# PostgreSQL connection string (production)
POSTGRES_URI = os.environ.get(
    "POSTGRES_URI_EXTERNAL",
    "postgresql+psycopg2://feast:feast@68.183.87.245:30032/feast"
)

# Connect to SQLite
sqlite_conn = sqlite3.connect(LIVE_DB_PATH)
df = pd.read_sql_query("SELECT * FROM live_data", sqlite_conn)
df = df.rename(columns={'true_label': 'target'})
sqlite_conn.close()

# Connect to Postgres
engine = create_engine(POSTGRES_URI)
metadata = MetaData()

# Define target table (should match your live_data schema)
live_table = Table(
    "live_data",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("employee_id", String, nullable=True),
    Column("features", JSONB, nullable=False),  # features stored as JSON
    Column("prediction", Integer, nullable=True)
)

metadata.create_all(engine)

# Transform each row to match JSONB schema
insert_rows = []
feature_cols = [col for col in df.columns if col not in ["id", "employee_id", "prediction", "target"]]

for _, row in df.iterrows():
    feature_dict = {col: row[col] for col in feature_cols}
    insert_rows.append({
        "employee_id": str(row["employee_id"]),
        "features": feature_dict,
        "prediction": int(row["prediction"]) if not pd.isna(row["prediction"]) else None,
        "target": int(row["target"]) if not pd.isna(row["target"]) else None
    })

# Insert into Postgres
with engine.connect() as conn:
    conn.execute(live_table.insert(), insert_rows)
    conn.commit()

print(f"✅ Migrated {len(insert_rows)} rows from SQLite to Postgres.")