# save_live_db_direct.py
import os
from sqlalchemy import create_engine, Table, Column, Integer, Float, String, MetaData
from sqlalchemy.dialects.postgresql import JSONB
from dotenv import load_dotenv

load_dotenv()

# PostgreSQL connection string (external link)
POSTGRES_URI = os.environ.get(
    "POSTGRES_URI_EXTERNAL",
    "postgresql+psycopg2://feast:feast@68.183.87.245:30032/feast"
)

# Connect to DB
engine = create_engine(POSTGRES_URI)
metadata = MetaData()

# Define table for live data (adjust columns as needed)
live_table = Table(
    "live_data",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("employee_id", String, nullable=True),
    Column("features", JSONB, nullable=False),  # store features as JSON
    Column("prediction", Integer, nullable=True),
    Column("target", Integer, nullable=True) 
)

# Create table if not exists
metadata.create_all(engine)

def log_live_data_direct(feature_row: dict, prediction: int = None, employee_id: str = None):
    """Insert live data into PostgreSQL"""
    with engine.connect() as conn:
        insert_stmt = live_table.insert().values(
            employee_id=employee_id,
            features=feature_row,
            prediction=prediction
        )
        conn.execute(insert_stmt)
        conn.commit()