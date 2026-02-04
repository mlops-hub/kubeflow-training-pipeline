import os
from _mlflow.registry import MLflowRegistry
from dotenv import load_dotenv

load_dotenv()


def register_model_to_mlflow(
    registry_name: str,
    tracking_uri: str,
    experiment_name: str,
    artifact_name: str,
    mlflow_run_id: str,
) -> object:

    registry = MLflowRegistry(
        tracking_uri=tracking_uri,
        experiment_name=experiment_name
    )

    registered = registry.register_model(
        run_id=mlflow_run_id,
        artifact_name=artifact_name,
        registry_name=registry_name
    )

    metrics = registry.get_metric_from_mlfow()

    return registered, metrics


def promote_to_production(
    metric: float,
    model_name: str,
    version: int,
    recall_threshold: float = 0.70,
) -> str:
    
    tracking_uri=os.environ["MLFLOW_TRACKING_URI"]
    experiment_name=os.environ["MLFLOW_EXPERIMENT_NAME"]

    registry = MLflowRegistry(
        tracking_uri=tracking_uri,
        experiment_name=experiment_name
    )

    stage = registry.promote_model(
        metric_value=metric,
        model_name=model_name,
        version=version,
        threshold=recall_threshold,
    )

    print(f"Model promoted to: {stage}")
    return stage


if __name__ == "__main__":
    from pathlib import Path

    BASE_DIR = Path(__file__).resolve().parents[2]
    ARTIFACTS_PATH = BASE_DIR / "artifacts" / "model_v1"
    MLFLOW_METADATA = ARTIFACTS_PATH / "mlflow_metadata.txt"

    with open(MLFLOW_METADATA, 'r') as f:
        run_id = f.read().strip()


    registered_model, metrics = register_model_to_mlflow(
        registry_name="register-employee-attrition-model",
        tracking_uri=os.environ["MLFLOW_TRACKING_URI"],
        experiment_name=os.environ["MLFLOW_EXPERIMENT_NAME"],
        artifact_name=os.environ["MLFLOW_MODEL_NAME"],
        mlflow_run_id=run_id
    )
    
    promote_to_production(
        metric=metrics['recall'],
        model_name=registered_model.name,
        version=registered_model.version,
        recall_threshold=0.6,
    )

