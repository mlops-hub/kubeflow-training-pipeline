import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from dotenv import load_dotenv

load_dotenv()


def preprocess_data(df_path: str, parquet_output_path: str):
    # ------- Feast Logic ---------
    full_df = pd.read_csv(df_path)

    if 'event_timestamp' not in full_df.columns:
        full_df['event_timestamp'] = pd.Timestamp.now()

    print("Columns in full_df before saving to parquet:")
    print(full_df.columns.tolist())

    full_df.to_parquet(parquet_output_path, index=False)
    print(f"Temporarily save Feast data in: {parquet_output_path}")
    # --- END FEAST LOGIC ---
     
    # Separate features and target
    X = full_df.drop(columns=["attrition"])
    y = full_df["attrition"]

    # Preprocess
    # 1. Scale Numeric cols: Scaling is about meaning, not datatype. Does the distance between values mean something numeric?
    #  - distance matter
    #  - magnitudes matter
    NUMERIC_COLS = [
        "years_at_company",
        "company_tenure",
        "role_stagnation_ratio",
        "tenure_gap",
        "number_of_promotions",
        "number_of_dependents",
    ]

    # 2. Binary (0/1): Do not scale
    #  - 0/1 is a state, not quantity
    #  - scaling destroys interpretability
    BINARY_COLS = ["overtime", "remote_work", "early_company_tenure_risk", "long_tenure_low_role_risk"]

    # 3. Ordinal Categorical: they look numeric but NOT
    #  - Check if distance between 1 and 2 is same as 3 and 4 ?
    #  - use OneHotEncoder unless you have strong reason not to.
    CATEGORICAL_COLS = [
        "annual_income",
        "education_level",
        "job_level",
        "company_size",
        "performance_rating",
        "age_group",
        "overall_satisfaction",
        "opportunities",
        "company_reputation",
    ]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_COLS),
            ("cat", OneHotEncoder(), CATEGORICAL_COLS),
            ("bin", "passthrough", BINARY_COLS),
        ],
        remainder="passthrough"
    )

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    print("Preprocessing completed and data saved.")
    return train_df, test_df, preprocessor


if __name__ == "__main__":
    from pathlib import Path
    import joblib
    
    BASE_DIR = Path(__file__).resolve().parents[2]    
    DATASET_PATH = BASE_DIR / "datasets" / "data-pipeline"
    FEATURED_PATH = DATASET_PATH / "05_feature_engg_df.csv"
    TRAIN_PATH = DATASET_PATH / "06_preprocess_train_df.csv"
    TEST_PATH = DATASET_PATH / "06_preprocess_test_df.csv"

    ARTIFACTS_PATH = BASE_DIR / "artifacts"
    os.makedirs(ARTIFACTS_PATH, exist_ok=True)
    PREPROCESSOR_PATH = ARTIFACTS_PATH / "preprocessor.pkl"

    # New Paths for Feast
    FEAST_DATA_DIR = BASE_DIR / "_feast" / "feature_repo" / "data" 
    os.makedirs(FEAST_DATA_DIR, exist_ok=True)
    PARQUET_PATH = FEAST_DATA_DIR / "preprocessed_data.parquet"


    train_df, test_df, preprocessor = preprocess_data(
        df_path=FEATURED_PATH,
        parquet_output_path=PARQUET_PATH,
    )

    # Save preprocessed data
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    train_df.to_csv(TRAIN_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)
