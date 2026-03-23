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
            input_data TEXT,
            prediction TEXT,
            confidence REAL       
        );
    """)
    conn.commit()
    conn.close()

def log_prediction(prediction: str = None, input_data=None, confidence: float = None, payload: dict = None):
    if payload is not None:
        prediction = payload.get("prediction", prediction)
        input_data = payload.get("input_data", payload.get("input", input_data))
        confidence = payload.get("confidence", confidence)

    input_text = None
    try:
        input_text = json.dumps(input_data)
    except Exception:
        input_text = str(input_data)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    timestamp = datetime.datetime.now().isoformat()
    cursor.execute(
        "INSERT INTO prediction_logs (timestamp, input_data, prediction, confidence) VALUES (?, ?, ?, ?)",
        (timestamp, input_text, str(prediction) if prediction is not None else None, confidence)
    )
    conn.commit()
    conn.close()
