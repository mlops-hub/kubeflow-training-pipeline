from sqlalchemy import create_engine, Table, Column, Integer, String, MetaData, Float
from sqlalchemy.dialects.postgresql import JSONB
import os
from dotenv import load_dotenv

load_dotenv()

POSTGRES_URI = os.environ.get(
    "POSTGRES_URI_EXTERNAL",
    "postgresql+psycopg://feast:feast@postgres.feast.svc.cluster.local:5432/feast"
    #"postgresql+psycopg2://feast:feast@68.183.87.245:30032/feast"
)

engine = create_engine(POSTGRES_URI)
metadata = MetaData()

reference_table = Table(
    "reference_data",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("employee_id", String),
    Column("features", JSONB),
    Column("target", Integer)
)

metadata.create_all(engine)


def log_reference_data_postgres(df):
    rows = []

    feature_cols = [
        col for col in df.columns
        if col not in ["employee_id", "event_timestamp", "attrition"]
    ]

    for _, row in df.iterrows():
        features = {col: row[col] for col in feature_cols}

        rows.append({
            "employee_id": str(row["employee_id"]),
            "features": features,
            "target": int(row["attrition"])
        })

    with engine.connect() as conn:
        conn.execute(reference_table.insert(), rows)
        conn.commit()

    print(f"✅ Saved {len(rows)} reference rows to Postgres")