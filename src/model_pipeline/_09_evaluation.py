import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from _mlflow.registry import MLflowRegistry
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
from dotenv import load_dotenv

load_dotenv()

def evaluate_data(test_path: str, tracking_uri: str, experiment_name: str, artifact_name: str, mlflow_run_id: str):

    registry = MLflowRegistry(
        tracking_uri=tracking_uri,
        experiment_name=experiment_name
    )

    with registry.start_run(run_id=run_id):
        # load model form MLflow
        model = registry.load_model(run_id=mlflow_run_id, artifact_name=artifact_name)

        df_test = pd.read_csv(test_path)
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

        print('metrics: ', metrics)

        registry.log_metrics_mlflow(metrics)


    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    print("Confusion Matrix:")
    print(conf_matrix)
    print("Classification Report:")
    print(class_report)

    # feature importance
    feature_names = model.named_steps["preprocessor"].get_feature_names_out()
    logreg = model.named_steps["model"]
    coef = logreg.coef_[0]
    coef_df = pd.DataFrame({
        'Feature': feature_names, 
        'Coefficient': coef,
        'Abs_Coefficient': np.abs(coef)
    }).sort_values(by='Abs_Coefficient', ascending=False)
    # print(coef_df)

    plt.figure(figsize=(10,6))
    plt.barh(coef_df['Feature'], coef_df['Coefficient'])
    plt.gca().invert_yaxis()
    plt.xlabel("Coefficient")
    plt.title("Logistic Regression Feature Importance")
    plt.show()

    return metrics



if __name__ == "__main__":
    from pathlib import Path

    BASE_DIR = Path(__file__).resolve().parents[2]
    DATASET_PATH = BASE_DIR / "datasets" / "data-pipeline"
    TEST_PATH = DATASET_PATH / "06_preprocess_test_df.csv"

    ARTIFACTS_PATH = BASE_DIR / "artifacts" / "model_v1"
    MLFLOW_METADATA = ARTIFACTS_PATH / "mlflow_metadata.txt"

    with open(MLFLOW_METADATA, 'r') as f:
        run_id = f.read().strip()

    evaluate_data(
        test_path=TEST_PATH, 
        tracking_uri=os.environ["MLFLOW_TRACKING_URI"],
        experiment_name=os.environ["MLFLOW_EXPERIMENT_NAME"],
        artifact_name = os.environ["MLFLOW_MODEL_NAME"],
        mlflow_run_id=run_id
    )
