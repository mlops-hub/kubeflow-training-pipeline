from kfp import dsl
from kfp.dsl import component, Input, Output, Dataset, Model

@component(
    base_image="sandy345/kubeflow-employee-attrition:latest", 
)
def preprocessed_component(
    input_data: Input[Dataset],
    train_data: Output[Dataset],
    test_data: Output[Dataset],
    scaler_model: Output[Model],
):
    import os
    import pandas as pd
    import joblib
    from src.data_pipeline._06_preprocessing import preprocess_data

    input_path = os.path.join(input_data.path, "transformed_data.csv")
    df = pd.read_csv(input_path)

    train_df, test_df, scaler = preprocess_data(df)

    os.makedirs(train_data.path, exist_ok=True)
    os.makedirs(test_data.path, exist_ok=True)
    os.makedirs(scaler_model.path, exist_ok=True)

    train_output_path = os.path.join(train_data.path, "train.csv")
    train_df.to_csv(train_output_path, index=False)

    test_output_path = os.path.join(test_data.path, "test.csv")
    test_df.to_csv(test_output_path, index=False)

    scaler_output_path = os.path.join(scaler_model.path, "scaler.pkl")
    joblib.dump(scaler, scaler_output_path)

    print("Preprocessing completed.")
    print(f"Train data saved to: {train_output_path}")
    print(f"Test data saved to: {test_output_path}")
    print(f"Scaler saved to: {scaler_output_path}")
