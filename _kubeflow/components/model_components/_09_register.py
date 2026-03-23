from kfp.dsl import component

@component(
    base_image="sandy345/final-kubeflow-pipeline:v1.0.0"
)
def register_model_component(
    registry_name: str, 
    recall_threshold: float,
    tracking_uri: str,
    experiment_name: str,
    artifact_name: str,
    mlflow_run_id: str
):
    from src.model_development._10_registry import register_model_to_mlflow, promote_to_production

    registered_model, metrics = register_model_to_mlflow(    
        registry_name=registry_name, 
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
        artifact_name=artifact_name,
        mlflow_run_id=mlflow_run_id
    )

    promote_to_production(
        metric=metrics['recall'],
        model_name=registered_model.name,
        version=registered_model.version,
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
        recall_threshold=recall_threshold,
    )

    print("Model registration completed.")