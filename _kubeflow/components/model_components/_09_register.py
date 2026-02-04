from kfp.dsl import component, InputPath

@component(
    base_image="sandy345/kubeflow-employee-attrition",
)
def register_model_component(
    registry_name: str, 
    recall_threshold: float,
    tracking_uri: str,
    experiment_name: str,
    artifact_name: str,
    mlflow_metadata: str
):
    import os
    from src.model_pipeline._10_registry import register_model_to_mlflow, promote_to_production

    registered_model, metrics = register_model_to_mlflow(    
        registry_name=registry_name, 
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
        artifact_name=artifact_name,
        mlflow_run_id=mlflow_metadata
    )

    promote_to_production(
        metric=metrics['recall'],
        model_name=registered_model.name,
        version=registered_model.version,
        recall_threshold=recall_threshold,
    )

    print("Model registration completed.")