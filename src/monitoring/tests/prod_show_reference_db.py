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
    result = pd.read_sql("SELECT * FROM reference_data", engine)
    print(result.head())
    print(result.isnull().sum())


