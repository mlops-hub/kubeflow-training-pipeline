from kfp import dsl
from kfp.dsl import component, Input, Output, Dataset, Metrics

BASE_IMAGE = "python:3.11-slim"

@component(
    base_image=BASE_IMAGE, 
    packages_to_install=['pandas', 'scikit-learn', 'joblib', '.'],
     source="./"
)
def evaluation_component(
    train_data: Input[Dataset],
    test_data: Input[Dataset],
    model_artifact: Input,
):
    import pandas as pd
    from pathlib import Path
    import joblib
    from src.model_pipeline._08_evaluation import evaluate_data

    train_path = Path(train_data.path) + "/train.csv"
    train_df = pd.read_csv(train_path)

    test_path = Path(test_data.path) + "/test.csv"
    test_df = pd.read_csv(test_path)

    model_path = Path(model_artifact.path) + "/model.pkl"
    model_file = joblib.load(model_path)

    recall_metric = evaluate_data(train_df, test_df, model_file)

    print(f"Evaluation is completed. Got accuracy: {recall_metric}")

