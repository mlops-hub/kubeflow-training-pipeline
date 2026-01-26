import os
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
from dotenv import load_dotenv

load_dotenv()

def evaluate_data(df_test: pd.DataFrame, model: str):
    X_test = df_test.drop(columns=['Attrition'])
    y_test = df_test['Attrition']

    # predict
    y_pred = model.predict(X_test)
     
    # metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_pred),
    }

    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    print("Confusion Matrix:")
    print(conf_matrix)
    print("Classification Report:")
    print(class_report)

    # feature importance
    feature_names = X_test.columns
    coef = model.coef_[0]
    coef_df = pd.DataFrame({
        'Feature': feature_names, 
        'Coefficient': coef,
        'Abs_Coefficient': np.abs(coef)
    }).sort_values(by='Abs_Coefficient', ascending=False)
    print(coef_df)

    plt.figure(figsize=(10,6))
    plt.barh(coef_df['Feature'], coef_df['Coefficient'])
    plt.gca().invert_yaxis()
    plt.xlabel("Coefficient")
    plt.title("Logistic Regression Feature Importance")
    plt.show()

    return metrics



if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parents[2]
    PREPROCESSED_TEST_PATH = BASE_DIR / "datasets" / "data-engg" / "06_preprocess_test_df.csv"
    MODEL_ARTIFACT = BASE_DIR / "artifacts" / "model_v1" / "best_model.pkl"
    METRICS_PATH = BASE_DIR /"artifacts" / "model_v1" / "evaluation_metrics.json"

    model = joblib.load(MODEL_ARTIFACT)
    df_test = pd.read_csv(PREPROCESSED_TEST_PATH)

    metrics = evaluate_data(df_test, model)

    joblib.dump(metrics, METRICS_PATH)
