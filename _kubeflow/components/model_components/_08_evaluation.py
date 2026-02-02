
from kfp.dsl import component, Input, Dataset


@component(
    base_image="sandy345/kubeflow-employee-attrition:latest", 
)
def evaluation_component(
    test_data: Input[Dataset],
):
    import os
    from src.model_pipeline._09_evaluation import evaluate_data

    test_path = os.path.join(test_data.path, "test.csv")

    # tracking_uri = os.environ["MLFLOW_TRACKING_URI", tracking_uri]
    # experiment_name = os.environ["MLFLOW_EXPERIMENT_NAME", experiment_name]

    metrics = evaluate_data(test_path)

    print(f"Evaluation is completed. Got accuracy: {metrics['recall']}")

