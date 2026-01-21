import os
import pandas as pd
from pathlib import Path
import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, recall_score, precision_recall_curve
from sklearn.model_selection import cross_val_score, StratifiedKFold
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parents[2]
PREPROCESSED_TRAIN_PATH = BASE_DIR / "datasets" / "data-engg" / "06_preprocess_train_df.csv"
PREPROCESSED_TEST_PATH = BASE_DIR / "datasets" / "data-engg" / "06_preprocess_test_df.csv"

MODEL_ARTIFACT = BASE_DIR / "artifacts" / "model_v1" / "model.pkl"
BEST_MODEL_ARTIFACT = BASE_DIR / "artifacts" / "model_v1" / "best_model.pkl"
MLFLOW_MODEL_INFO = BASE_DIR / "artifacts" / "model_v1" / "model_info.pkl"
METRICS_DATA = BASE_DIR / "artifacts" / "model_v1" / "metrics.pkl"
os.makedirs(BEST_MODEL_ARTIFACT.parent, exist_ok=True)


def tuning_data(df_train, df_test, model):
    X_train = df_train.drop(columns=['Attrition'])
    y_train = df_train['Attrition']

    X_test = df_test.drop(columns=['Attrition'])
    y_test = df_test['Attrition']

    # set parameters
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'solver': ['liblinear', 'saga'],
        'l1_ratio': [0],     # equivalent to L2
        'max_iter': [1000]
    }

    # set cv
    strat_cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

    # use gridserachcv for tuning model
    grid = GridSearchCV(model, param_grid=param_grid, cv=strat_cv, scoring='recall')
    grid.fit(X_train, y_train)

    # get best model and save in models/
    tuned_model = grid.best_estimator_
    joblib.dump(tuned_model, BEST_MODEL_ARTIFACT)

    best_parameters = grid.best_params_
    print(f'best params: {best_parameters}')
        
    # predict the output with tuned_model
    y_pred_ht = tuned_model.predict(X_test)

    # tuned model evaluation
    metrics = {
        "accuracy_ht": accuracy_score(y_test, y_pred_ht),
        "recall_ht": recall_score(y_test, y_pred_ht)
    }        
    print('accuracy: ', metrics['accuracy_ht'])
    print('recall: ', metrics['recall_ht'])

    overall_parameters = {
        "best_parameters": best_parameters,
        "best_metrics": metrics
    }
    joblib.dump(overall_parameters, METRICS_DATA)

    # get trin/test score
    tuned_train_score = tuned_model.score(X_train, y_train) 
    tuned_test_score =  tuned_model.score(X_test, y_test)
    print('tuned train score: ', tuned_train_score)
    print('tuned test score: ', tuned_test_score)

    # Compare CV scores of base model and tuned model
    base_cv_scores = cross_val_score(model, X_train, y_train, cv=strat_cv)
    tuned_cv_scores = cross_val_score(tuned_model, X_train, y_train, cv=strat_cv)
    print("=== MODEL COMPARISON ===")
    print(f"Base Model CV:     {base_cv_scores.mean():.4f} (+/- {base_cv_scores.std():.4f})")
    print(f"Tuned Model CV:    {tuned_cv_scores.mean():.4f} (+/- {tuned_cv_scores.std():.4f})")

    return tuned_model



if __name__ == "__main__":
    df_train = pd.read_csv(PREPROCESSED_TRAIN_PATH)
    df_test = pd.read_csv(PREPROCESSED_TEST_PATH)

    model = joblib.load(MODEL_ARTIFACT)

    tuning_data(df_train, df_test, model)