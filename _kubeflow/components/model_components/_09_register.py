from kfp.dsl import component

@component(
    base_image="aswinvj/kubeflow:latest",
)
def register_model_component(
    registry_name: str, 
    recall_threshold: float
):
    # import os
    from src.model_pipeline._10_registry import register_model_to_mlflow, promote_to_production

    # tracking_uri = os.environ["MLFLOW_TRACKING_URI", tracking_uri]
    # experiment_name = os.environ["MLFLOW_EXPERIMENT_NAME", experiment_name]

    registered_model, metrics = register_model_to_mlflow(    
        registry_name=registry_name, 
    )

    promote_to_production(
        metric=metrics['recall'],
        model_name=registered_model.name,
        version=registered_model.version,
        recall_threshold=recall_threshold,
    )

    print("Model registration completed.")