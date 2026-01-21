import os
import pandas as pd
from pathlib import Path
import joblib
from sklearn.model_selection import cross_val_score, StratifiedKFold


BASE_DIR = Path(__file__).resolve().parents[2]
PREPROCESSED_TRAIN_PATH = BASE_DIR / "datasets" / "data-engg" / "06_preprocess_train_df.csv"

MODEL_ARTIFACT = BASE_DIR / "artifacts" / "model_v1" / "model.pkl"


def cv_data(df_train, model):
    X_train = df_train.drop(columns=['Attrition'])
    y_train = df_train['Attrition']

    strat_cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=strat_cv, scoring='recall')

    print(f"Strat cv score: {cv_scores.mean() * 100}")
    return cv_scores
    


if __name__ == "__main__":
    df_train = pd.read_csv(PREPROCESSED_TRAIN_PATH)
    model = joblib.load(MODEL_ARTIFACT)

    cv_data(df_train, model)