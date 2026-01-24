from kfp.dsl import component, Input, Model, Dataset

BASE_IMAGE = "python:3.10-slim"

@component(
    base_image=BASE_IMAGE,
    packages_to_install=["pandas", "mlflow", "joblib", "git+https://github.com/mlops-hub/kubeflow-training-pipeline.git"],
)
def register_model_component(
    train_data: Input[Dataset],
    tuned_model: Input[Model],
    tuning_metadata: Input[Dataset],
    tracking_uri: str, 
    experiment_name: str, 
    registry_name: str, 
    recall_threshold: float
):
    import pandas as pd
    from src.model_registry.registry import register_best_model

    model_path = tuned_model.path
    df_train = train_data.path
    metadata = pd.read_csv(tuning_metadata.path).to_dict(orient='records')[0]
    params = metadata['best_parameters']
    metrics = metadata['best_metrics']

    register_best_model(    
        model_path, 
        df_train, 
        params, 
        metrics, 
        tracking_uri, 
        experiment_name, 
        registry_name, 
        recall_threshold    
    )