
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
EDA_PATH = BASE_DIR / "datasets" / "data-engg" / "03_eda_df.csv"
CLEANED_PATH = BASE_DIR / "datasets" / "data-engg" / "04_cleaned_df.csv"

def clean_data(df):
    print("--- Find duplicates ---")
    df_duplicate = df.duplicated()
    print(f"Number of duplicate rows: {df_duplicate.sum()}")

    print("--- Find Missing/Null values ---")
    missing_values = df.isnull().sum()
    print(f"Missing values: {missing_values}")

    df.to_csv(CLEANED_PATH, index=False)

    return df

if __name__ == "__main__":
    df = pd.read_csv(EDA_PATH)
    clean_data(df)
