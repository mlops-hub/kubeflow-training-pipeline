import os
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parents[2]
PREPROCESSED_TRAIN_PATH = BASE_DIR / "datasets" / "data-engg" / "06_preprocess_train_df.csv"
PREPROCESSED_TEST_PATH = BASE_DIR / "datasets" / "data-engg" / "06_preprocess_test_df.csv"


MODEL_ARTIFACT = BASE_DIR / "artifacts" / "model_v1" / "model.pkl"
os.makedirs(MODEL_ARTIFACT.parent, exist_ok=True)


def evaluate_data(df_train, df_test, model):
    X_train = df_train.drop(columns=['Attrition'])
    y_train = df_train['Attrition']

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

    # train/test scores
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)

    print('train score %: ', train_score * 100)
    print('test score %: ', test_score * 100)

    # feature importance
    feature_names = X_train.columns
    coef = model.coef_[0]
    coef_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coef})
    coef_df['Abs_Coefficient'] = np.abs(coef_df['Coefficient'])
    coef_df = coef_df.sort_values(by='Abs_Coefficient', ascending=False)
    print(coef_df)

    plt.figure(figsize=(10,6))
    plt.barh(coef_df['Feature'], coef_df['Coefficient'])
    plt.gca().invert_yaxis()
    plt.xlabel("Coefficient")
    plt.title("Logistic Regression Feature Importance")
    plt.show()

    return metrics['recall']



if __name__ == "__main__":
    df_train = pd.read_csv(PREPROCESSED_TRAIN_PATH)
    df_test = pd.read_csv(PREPROCESSED_TEST_PATH)

    model = joblib.load(MODEL_ARTIFACT)

    evaluate_data(df_train, df_test, model)
