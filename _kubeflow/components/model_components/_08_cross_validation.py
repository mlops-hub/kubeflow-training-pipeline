from kfp.dsl import component, Input, Output, Model, Dataset

@component(
    base_image="sandy345/kubeflow-employee-attrition:latest", 
)
def cross_validation_component(
    train_data: Input[Dataset],
    model_artifact: Input[Model] ,
    cv_result: Output[Dataset]
):
    import pandas as pd
    from pathlib import Path
    import joblib
    from src.model_pipeline._09_cross_validation import cv_data

    train_path = Path(train_data.path) + "/train.csv"
    train_df = pd.read_csv(train_path)

    model_path = Path(model_artifact.path)
    model = joblib.load(model_path)

    results = cv_data(train_df, model)

    output_path = cv_result.path + "/cv_results.csv"
    results.to_csv(output_path, index=False)

    print("Cross-validation completed.")