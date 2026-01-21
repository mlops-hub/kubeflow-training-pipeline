from kfp import dsl
from kfp.dsl import component, Input, Output, Dataset, Model

BASE_IMAGE = "python:3.11-slim"

@component(
    base_image=BASE_IMAGE, 
    packages_to_install=['pandas', 'scikit-learn', 'joblib', '.'],
    source="./"
)
def preprocessed_component(
    input_data: Input[Dataset],
    train_data: Output[Dataset],
    test_data: Output[Dataset],
    scaler_model: Output[Model],
):
    import pandas as pd
    import joblib
    from pathlib import Path
    from src.data_pipeline._06_preprocessing import preprocess_data

    input_path = Path(input_data.path) + "/transformed_data.csv"
    df = pd.read_csv(input_path)

    train_df, test_df, scaler = preprocess_data(df)

    train_output_path = Path(train_data.path) / "train.csv"
    train_df.to_csv(train_output_path, index=False)

    test_output_path = Path(test_data.path) / "test.csv"
    test_df.to_csv(test_output_path, index=False)

    scaler_output_path = Path(scaler_model.path) / "preprocessor.pkl"
    joblib.dump(scaler, scaler_output_path)

    print("Preprocessing completed.")
    print(f"Train data saved to: {train_output_path}")
    print(f"Test data saved to: {test_output_path}")
    print(f"Scaler saved to: {scaler_output_path}")
