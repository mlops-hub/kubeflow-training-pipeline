import os
from pathlib import Path
import pandas as pd
from pandas import DataFrame

BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = BASE_DIR / "datasets" / "employee_attrition.csv"
INGESTION_PATH = BASE_DIR / "datasets" / "data-engg" / "01_ingestion.csv"
os.makedirs(INGESTION_PATH.parent, exist_ok=True)

def ingestion() -> DataFrame:
    df = pd.read_csv(RAW_DATA_PATH)
    print(df.head(5))
    print("------")

    print(f"Shape: {df.shape}")
    print("------")

    print(f"Information: {df.info()}")
    print("------")
    
    df.to_csv(INGESTION_PATH, index=False)
    return df


if __name__ == "__main__":
    ingestion()