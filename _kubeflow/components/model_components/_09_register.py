from kfp.dsl import component, Input, Model, Dataset

@component(
    base_image="sandy345/kubeflow-employee-attrition:latest",
)
def register_model_component(
    train_data: Input[Dataset],
    best_model: Input[Model],
    tuning_metadata: Input[Dataset],
    metrics: Input[Dataset],
    tracking_uri: str, 
    experiment_name: str, 
    registry_name: str, 
    recall_threshold: float
):
    import os
    import json
    from src.model_registry.registry import register_best_model

    model_path = os.path.join(best_model.path, "best_model.pkl")
    train_df_path = os.path.join(train_data.path, "train.csv")
    
    metadata_file = os.path.join(tuning_metadata.path, "tuning_metadata.json")
    with open(metadata_file, 'r') as f:
        overall_parameters = json.load(f)
    
    # Convert stringified dicts back to dicts 
    params = { 
        key: overall_parameters[key] 
        for key in ["C", "solver", "l1_ratio", "max_iter"] 
    }

    metrics_file = os.path.join(metrics.path, "evaluation_metrics.json")
    with open(metrics_file, 'r') as f:
        eval_metrics = json.load(f)

    register_best_model(    
        model_path=model_path, 
        train_df_path=train_df_path, 
        params=params, 
        metrics=eval_metrics, 
        tracking_uri=tracking_uri, 
        experiment_name=experiment_name, 
        registry_name=registry_name, 
        recall_threshold=recall_threshold    
    )

    print("Model registration completed.")