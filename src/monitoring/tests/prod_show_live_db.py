from sqlalchemy import create_engine, text
import os
from dotenv import load_dotenv

load_dotenv()
POSTGRES_URI = os.environ.get(
    "POSTGRES_URI_EXTERNAL",
    "postgresql+psycopg2://feast:feast@68.183.87.245:30032/feast"
)
engine = create_engine(POSTGRES_URI)

with engine.connect() as conn:
    result = conn.execute(text("SELECT * FROM live_data ORDER BY id DESC LIMIT 5"))
    for row in result:
        print(row)


