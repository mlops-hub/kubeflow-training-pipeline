import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

ORG_ID = os.environ.get("EVIDENLTY_ORG_ID", "")
PROJECT_ID = os.environ.get("EVIDENTLY_PROJECT_ID", "")

EVIDENTLY_TOKEN = os.environ.get("EVIDENTLY_TOKEN", "")
EVIDENTLY_URL = "https://app.evidently.cloud"

PROMETHEUS_PORT = int(os.environ.get("MONITOR_PROMETHEUS_PORT", 9090))

PROJECT_ROOT = Path(__file__).resolve().parent.parent

LIVE_DB_PATH = PROJECT_ROOT / "db" / "live_data.db"
REFERENCE_DB_PATH = PROJECT_ROOT / "db" / "reference_data.db"
OUTPUT_DB_PATH =  PROJECT_ROOT / "reports"
