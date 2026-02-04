
from kfp.dsl import component, Input, InputPath, Dataset


@component(
    base_image="sandy345/kubeflow-employee-attrition", 
)
def evaluation_component(
    test_data: Input[Dataset],
    tracking_uri: str,
    experiment_name: str,
    artifact_name: str,
    mlflow_metadata: str,
):
    import os
    from src.model_pipeline._09_evaluation import evaluate_data

    test_path = os.path.join(test_data.path, "test.csv")

    with open(mlflow_metadata, 'r') as f:
        mlflow_run_id = f.read().strip()

    metrics = evaluate_data(
        test_path=test_path, 
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
        artifact_name=artifact_name,
        mlflow_run_id=mlflow_metadata
    )

    print(f"Evaluation is completed. Got accuracy: {metrics['recall']}")

