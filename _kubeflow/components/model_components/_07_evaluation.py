from kfp import dsl
from kfp.dsl import component, Input, Dataset, Model


@component(
    base_image="sandy345/kubeflow-employee-attrition:latest", 
)
def evaluation_component(
    train_data: Input[Dataset],
    test_data: Input[Dataset],
    base_model: Input[Model],
):
    import os
    import pandas as pd
    import joblib
    from src.model_pipeline._08_evaluation import evaluate_data

    train_path = os.path.join(train_data.path, "train.csv")
    train_df = pd.read_csv(train_path)

    test_path = os.path.join(test_data.path, "test.csv")
    test_df = pd.read_csv(test_path)

    model_path = os.path.join(base_model.path, "model.pkl")
    model_file = joblib.load(model_path)

    recall_metric = evaluate_data(train_df, test_df, model_file)

    print(f"Evaluation is completed. Got accuracy: {recall_metric}")

