from sqlalchemy import create_engine, text
import os
from dotenv import load_dotenv
import pandas as pd

load_dotenv()
POSTGRES_URI = os.environ.get(
    "POSTGRES_URI_EXTERNAL",
    "postgresql+psycopg://feast:feast@68.183.87.245:30032/feast"
)
engine = create_engine(POSTGRES_URI)

with engine.connect() as conn:
    df = pd.read_sql("SELECT * FROM live_data", engine)
    print(df.head())
    print(df['features'].columns.tolist())


