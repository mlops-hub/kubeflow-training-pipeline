from kfp import dsl
from kfp.dsl import component, Input, Output, Dataset, Model


@component(
    base_image="sandy345/kubeflow-employee-attrition:latest", 
)
def evaluation_component(
    test_data: Input[Dataset],
    best_model: Input[Model],
    evaluation_metrics: Output[Dataset]
):
    import os
    import json
    import pandas as pd
    from utils.s3_loader import load_model_from_uri
    from model_pipeline._09_evaluation import evaluate_data

    test_path = os.path.join(test_data.path, "test.csv")
    test_df = pd.read_csv(test_path)

    # read from s3 and load to model
    with open(best_model.path, 'r') as f:
        model_uri = f.read().strip()

    model = load_model_from_uri(model_uri)

    metrics = evaluate_data(test_df, model)

    os.makedirs(evaluation_metrics.path, exist_ok=True)
    metrics_file = os.path.join(evaluation_metrics.path, "evaluation_metrics.json")
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f)

    print(f"Evaluation is completed. Got accuracy: {metrics["accuracy"]}")

