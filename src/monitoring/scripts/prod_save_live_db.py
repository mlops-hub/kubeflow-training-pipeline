# save_live_db_direct.py
import os
from sqlalchemy import create_engine, Table, Column, Integer, DateTime, Float, String, MetaData
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
    Column("employee_id", String),
    Column("age_group", Integer),
    Column("years_at_company", Integer),
    Column("annual_income", Integer),
    Column("overall_satisfaction", Integer),
    Column("performance_rating", Integer),
    Column("number_of_promotions", Integer),
    Column("overtime", Integer),
    Column("education_level", Integer),
    Column("number_of_dependents", Integer),
    Column("job_level", Integer),
    Column("company_size", Integer),
    Column("company_tenure", Float),
    Column("remote_work", Integer),
    Column("opportunities", Integer),
    Column("company_reputation", Integer),
    Column("role_stagnation_ratio", Float),
    Column("tenure_gap", Float),
    Column("early_company_tenure_risk", Float),
    Column("long_tenure_low_role_risk", Float),
    Column("prediction", Integer),
    Column("target", Integer),
    Column("event_timestamp", DateTime),
)

# Create table if not exists
metadata.create_all(engine)

def log_live_data(feature_row: dict, prediction: int = None, employee_id: str = None):
    from datetime import datetime 
    
    row = {
        "employee_id": employee_id,
        **feature_row,
        "prediction": prediction,
        "target": prediction,
        "event_timestamp": datetime.utcnow()
    }

    with engine.connect() as conn:
        conn.execute(live_table.insert().values(row))
        conn.commit()
    print(f"✅ Logged live data for employee_id={employee_id}")