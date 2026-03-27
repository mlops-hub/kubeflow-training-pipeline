from sqlalchemy import create_engine
import os
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

POSTGRES_URI = os.environ.get(
    "POSTGRES_URI_EXTERNAL",
        # "postgresql+psycopg://feast:feast@postgres.feast.svc.cluster.local:5432/feast"
    "postgresql+psycopg://feast:feast@68.183.87.245:30032/feast"
)
print(POSTGRES_URI)
engine = create_engine(POSTGRES_URI)

df = pd.read_sql("SELECT * FROM live_data", engine)
print(df.head())
