from kfp.dsl import component

@component(
    base_image="sandy345/kubeflow-employee-attrition:latest",
)
def register_model_component(
    tracking_uri: str, 
    experiment_name: str, 
    registry_name: str, 
    recall_threshold: float
):
    from src.model_pipeline._10_registry import register_model_to_mlflow, promote_to_production

    registered_model, metrics = register_model_to_mlflow(    
        tracking_uri=tracking_uri, 
        experiment_name=experiment_name, 
        registry_name=registry_name, 
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