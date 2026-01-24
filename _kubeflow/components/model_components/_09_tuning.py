from kfp import dsl
from kfp.dsl import Input, Output, Dataset, Model

@dsl.component(
    base_image="sandy345/kubeflow-employee-attrition:latest",
)
def tuning_component(
    train_data: Input[Dataset],
    test_data: Input[Dataset],
    base_model: Input[Model],
    tuned_model: Output[Model],
    tuning_metadata: Output[Dataset],
):
    import os
    import pandas as pd
    import joblib
    from src.model_pipeline._10_tuning import tuning_data

    train_path = os.path.join(train_data.path, "train.csv")
    train_df = pd.read_parquet(train_path)


    test_path = os.path.join(test_data.path, "test.csv")
    test_df = pd.read_csv(test_path)

    model_path = os.path.join(base_model.path, "model.pkl")
    base_model_file = joblib.load(model_path)

    best_model, overall_parameters = tuning_data(train_df, test_df, base_model_file)

    os.makedirs(tuned_model.path, exist_ok=True)
    tuned_model_file = os.path.join(tuned_model.path, "best_model.pkl")
    joblib.dump(best_model, tuned_model_file)

    os.makedirs(tuning_metadata.path, exist_ok=True)
    metadata_file = os.path.join(tuning_metadata, "tuning_metrics.csv")

    pd.DataFrame([overall_parameters]).to_csv(metadata_file, index=False)

    print("Tuning completed successfully.")

