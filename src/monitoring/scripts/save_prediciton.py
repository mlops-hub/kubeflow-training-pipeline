import sqlite3
import datetime
import json
from pathlib import Path

DB_DIR = Path(__file__).resolve().parent.parent / "db" 
DB_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = DB_DIR / "prediciton_logs.db"

def init_prediciton_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS prediction_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            prediction INTEGER,
            target_value TEXT,
            confidence REAL       
        );
    """)
    conn.commit()
    conn.close()

def log_prediction(payload: dict = None):
    if payload is not None:
        prediction = payload.get("prediction")
        target_value = payload.get("target_value")
        confidence = payload.get("confidence")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    timestamp = datetime.datetime.now().isoformat()
    cursor.execute(
        "INSERT INTO prediction_logs (timestamp, prediction, target_value, confidence) VALUES (?, ?, ?, ?)",
        (timestamp, int(prediction) if prediction is not None else None, str(target_value), confidence)
    )
    conn.commit()
    conn.close()
