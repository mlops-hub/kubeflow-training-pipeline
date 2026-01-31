
from kfp.dsl import component, Input, Dataset


@component(
    base_image="sandy345/kubeflow-employee-attrition:latest", 
)
def evaluation_component(
    test_data: Input[Dataset],
    tracking_uri: str,
    experiment_name: str
):
    import os
    from src.model_pipeline._09_evaluation import evaluate_data

    test_path = os.path.join(test_data.path, "test.csv")

    metrics = evaluate_data(test_path, tracking_uri, experiment_name)

    print(f"Evaluation is completed. Got accuracy: {metrics['recall']}")

