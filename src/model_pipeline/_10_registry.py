import os
from _mlflow.registry import MLflowRegistry
from dotenv import load_dotenv

load_dotenv()


def register_model_to_mlflow(
    registry_name: str
) -> object:

    tracking_uri=os.environ["MLFLOW_TRACKING_URI"]
    experiment_name=os.environ["MLFLOW_EXPERIMENT_NAME"]
        
    registry = MLflowRegistry(
        tracking_uri=tracking_uri,
        experiment_name=experiment_name
    )

    metadata = registry.get_model_uri_from_mlflow()
    print('metadata: ', metadata)

    registered = registry.register_model(
        metadata=metadata,
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
    registered_model, metrics = register_model_to_mlflow(
        registry_name="register-employee-attrition-model"
    )
    
    promote_to_production(
        metric=metrics['recall'],
        model_name=registered_model.name,
        version=registered_model.version,
        recall_threshold=0.6,
    )

