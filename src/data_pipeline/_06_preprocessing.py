import os
import pandas as pd
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parents[2]
FEATURED_PATH = BASE_DIR / "datasets" / "data-engg" / "05_feature_engg_df.csv"
PREPROCESSED_PATH = BASE_DIR / "datasets" / "data-engg"
SCALER_PATH = BASE_DIR / "artifacts" / "model_v1"
os.makedirs(SCALER_PATH, exist_ok=True)

SCALER_PREPROCESSOR_PATH = SCALER_PATH / "preprocessor.pkl"

def preprocess_data(df):
    df_pp = df.copy()

    # Separate features and target
    X = df_pp.drop(columns=['Attrition'])
    y = df_pp['Attrition']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Standardize numerical features
    numeric_cols = ['Years at Company', 'Company Tenure', 'RoleStagnationRatio', 'TenureGap']

    scaler = StandardScaler()

    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    # joblib.dump(scaler, SCALER_PREPROCESSOR_PATH)

    # 🔑 RESET INDICES (CRITICAL)
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    # Save preprocessed data
    # train_df.to_csv(PREPROCESSED_PATH / "06_preprocess_train_df.csv", index=False)
    # test_df.to_csv(PREPROCESSED_PATH / "06_preprocess_test_df.csv", index=False)


    print("Preprocessing completed and data saved.")
    return train_df, test_df, scaler


if __name__ == "__main__":
    df = pd.read_csv(FEATURED_PATH)
    preprocess_data(df)