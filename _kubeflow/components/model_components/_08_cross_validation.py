from kfp.dsl import component, Input, Output, Model, Dataset

@component(
    base_image="sandy345/kubeflow-employee-attrition:latest", 
)
def cross_validation_component(
    train_data: Input[Dataset],
    base_model: Input[Model] ,
):
    import os
    import pandas as pd
    import joblib
    from src.model_pipeline._09_cross_validation import cv_data

    train_path = os.path.join(train_data.path, "train.csv")
    train_df = pd.read_csv(train_path)

    model_path = os.path.join(base_model.path, "model.pkl")
    model_file = joblib.load(model_path)

    results = cv_data(train_df, model_file)
    print(results)

    print("Cross-validation completed.")