from kfp.dsl import component, Input, Model, Dataset

@component(
    base_image="sandy345/kubeflow-employee-attrition:latest",
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
    import os
    import json
    import pandas as pd
    from src.model_registry.registry import register_best_model

    model_path = os.path.join(tuned_model.path, "best_model.pkl")
    train_df_path = os.path.join(train_data.path, "train.csv")
    metadata_file = os.path.join(tuning_metadata.path, "tuning_metrics.csv")

    metadata = pd.read_csv(metadata_file).to_dict(orient='records')[0]

    # Convert stringified dicts back to dicts 
    params = json.loads(metadata["best_parameters"].replace("'", '"')) 
    metrics = json.loads(metadata["best_metrics"].replace("'", '"'))

    register_best_model(    
        model_path=model_path, 
        train_df_path=train_df_path, 
        params=params, 
        metrics=metrics, 
        tracking_uri=tracking_uri, 
        experiment_name=experiment_name, 
        registry_name=registry_name, 
        recall_threshold=recall_threshold    
    )

    print("Model registration completed.")