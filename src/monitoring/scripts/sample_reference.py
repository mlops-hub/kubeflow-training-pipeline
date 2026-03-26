import sqlite3
import pandas as pd
from pathlib import Path

LIVE_DIR = Path(__file__).resolve().parent.parent / "db"
LIVE_DIR.mkdir(parents=True, exist_ok=True)

LIVE_DB_PATH = LIVE_DIR / "reference_data.db"

def get_sampled_reference_dataset():
    conn = sqlite3.connect(LIVE_DB_PATH)

    df = pd.read_sql("SELECT * FROM reference_data", conn)

    # rename target column
    df = df.rename(columns={'attrition': 'target'})

    df['prediciton'] = df['target'].copy()

    df.fillna(0, inplace=True)

    conn.close()
    
    # sampled dataset since live-dataset is small...
    df_sampled = df.sample(n=500, random_state=42)
    return df_sampled
