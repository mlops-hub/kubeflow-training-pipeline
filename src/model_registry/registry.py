from _mlflow.registry import MLflowRegistry
import pandas as pd
import joblib


def register_best_model(
    model_path, 
    train_df_path, 
    params, 
    metrics, 
    tracking_uri, 
    experiment_name, 
    registry_name, 
    recall_threshold    
):
    model = joblib.load(model_path)
    df = pd.read_csv(train_df_path)
    X_train = df.drop(columns=["Attrition"])

    registry = MLflowRegistry(
        tracking_uri=tracking_uri,
        experiment_name=experiment_name
    )

    with registry.start_run(run_name="tuned-model-run"):
        metadata = registry.log_model(
            model=model,
            X_train=X_train,
            parameters=params,
            metrics=metrics,
            artifact_name="employee-attrition-model",
        )

    registered = registry.register_model(
        metadata=metadata,
        registry_name=registry_name
    )

    stage = registry.promote_model(
        model_name=registered.name,
        version=registered.version,
        metric_value=metrics["recall"],
        threshold=recall_threshold,
    )

    print("Model promoted to:", stage)
    return stage


if __name__ == "__main__":
    import json
    from pathlib import Path

    BASE_DIR = Path(__file__).resolve().parents[2]
    PREPROCESSED_TRAIN_PATH = BASE_DIR / "datasets" / "data-engg" / "06_preprocess_train_df.csv"
    BEST_MODEL_ARTIFACT = BASE_DIR / "artifacts" / "model_v1" / "best_model.pkl"
    TUNING_METADATA = BASE_DIR / "artifacts" / "model_v1" / "tuning_metadata.json"
    METRICS_DATA = BASE_DIR /"artifacts" / "model_v1" / "evaluation_metrics.json"

    model_path = BEST_MODEL_ARTIFACT
    train_df_path = PREPROCESSED_TRAIN_PATH

    with open(TUNING_METADATA, 'r') as f:
        overall_parameters = json.load(f)
    params = { 
        key: overall_parameters[key] 
        for key in ["C", "solver", "l1_ratio", "max_iter"] 
    }

    with open(METRICS_DATA, 'r') as f:
        metrics = json.load(f)

    # User‑defined MLflow settings 
    tracking_uri = "http://localhost:5000" 
    experiment_name = "employee-attrition-v1"
    registry_name = "register-employee-attrition-model"
    recall_threshold = 0.70

    register_best_model(model_path, train_df_path, params, metrics, tracking_uri, experiment_name, registry_name, recall_threshold)
