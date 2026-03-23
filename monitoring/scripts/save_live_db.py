import sqlite3
import pandas as pd
from datetime import datetime
from pathlib import Path

LIVE_DIR = Path(__file__).resolve().parent.parent / "db"
LIVE_DIR.mkdir(parents=True, exist_ok=True)

LIVE_DB_PATH = LIVE_DIR / "live_data.db"


def init_live_db():
    conn = sqlite3.connect(LIVE_DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS live_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            employee_id TEXT,
            age_group INTEGER,
            years_at_company INTEGER,
            job_role TEXT,
            annual_income REAL,
            overall_satisfaction TEXT,
            performance_rating TEXT,
            number_of_promotions INTEGER,
            overtime TEXT,
            education_level TEXT,
            number_of_dependents INTEGER,
            job_level TEXT,
            company_size TEXT,
            company_tenure INTEGER,
            remote_work TEXT,
            opportunities TEXT,
            company_reputation TEXT,
            role_stagnation_ratio REAL,
            tenure_gap REAL,
            early_company_tenure_risk REAL,
            long_tenure_low_role_risk REAL,
            prediction TEXT,
            true_label TEXT,
            event_timestamp TEXT
        )
    """)
    conn.commit()
    conn.close()


def log_live_data(feature_row: dict, prediction: str = None, employee_id: str = None):
    conn = sqlite3.connect(LIVE_DB_PATH)

    allowed_keys = [
        "employee_id", "age_group", "years_at_company", "job_role",
        "annual_income", "overall_satisfaction", "performance_rating",
        "number_of_promotions", "overtime", "education_level",
        "number_of_dependents", "job_level", "company_size",
        "company_tenure", "remote_work", "opportunities",
        "company_reputation", "role_stagnation_ratio", "tenure_gap",
        "early_company_tenure_risk", "long_tenure_low_role_risk",
    ]

    input_row = {k: feature_row.get(k) for k in allowed_keys}

    # employee id precedence: explicit param > feature_row value > None
    eid = employee_id or feature_row.get("employee_id")
    input_row["employee_id"] = str(eid) if eid is not None else None

    # determine true label if provided in feature_row
    true_label = feature_row.get("attrition") or feature_row.get("true_label")
    input_row["true_label"] = true_label

    input_row["prediction"] = prediction or feature_row.get("prediction")
    input_row["event_timestamp"] = datetime.now().isoformat()

    df = pd.DataFrame([input_row])
    df.to_sql("live_data", conn, if_exists="append", index=False)
    conn.close()
