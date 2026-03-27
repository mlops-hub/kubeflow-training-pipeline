from sqlalchemy import create_engine, Table, Column, Integer, String, MetaData, Float, DateTime, text
import os
from dotenv import load_dotenv

load_dotenv()


POSTGRES_URI = os.environ.get(
    "POSTGRES_URI_EXTERNAL",
    "postgresql+psycopg2://feast:feast@68.183.87.245:30032/feast"
)

# Connect to Postgres
engine = create_engine(POSTGRES_URI)

# Drop table
with engine.connect() as conn:
    try:
        conn.execute(text("DROP TABLE IF EXISTS reference_data;"))
        conn.commit()
        print("✅ reference_data table dropped successfully.")
    except Exception as e:
        print("❌ Failed to drop table:", e)

metadata = MetaData()

reference_table = Table(
    "reference_data",
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

metadata.create_all(engine)


def log_reference_data(df):
    ref_df = df.copy()
    ref_df['target'] = ref_df['attrition']
    ref_df['prediction'] = ref_df['attrition']

    # Ensure all table columns exist in df
    table_cols = [c.name for c in reference_table.columns if c.name != "id"]
    for col in table_cols:
        if col not in ref_df.columns:
            ref_df[col] = None  # fill missing cols with None

    # Only keep columns that exist in table
    ref_df = ref_df[table_cols]

    # Insert into Postgres
    ref_df.to_sql(
        "reference_data",
        engine,
        if_exists="append",  # or "replace" if you want to overwrite
        index=False
    )

    print(f"✅ Saved {len(ref_df)} reference rows to Postgres")