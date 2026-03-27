from kfp import dsl
from kfp.dsl import component, Input, Output, Dataset, Model
from _kubeflow.config import BASE_IMAGE

@component(
    base_image=BASE_IMAGE
)
def preprocessed_component(
    input_data: Input[Dataset],
    train_data: Output[Dataset],
    test_data: Output[Dataset],
    preprocessor_model: Output[Model],
    feast_data: Output[Dataset],
):
    import os
    import joblib
    from src.data_preparation._06_preprocessing import preprocess_data

    input_path = os.path.join(input_data.path, "feature_engg.csv")
    
    os.makedirs(train_data.path, exist_ok=True)
    os.makedirs(test_data.path, exist_ok=True)
    os.makedirs(preprocessor_model.path, exist_ok=True)
    os.makedirs(feast_data.path, exist_ok=True)

    train_output_path = os.path.join(train_data.path, "train.csv")
    test_output_path = os.path.join(test_data.path, "test.csv")
    preprocessor_output_path = os.path.join(preprocessor_model.path, "preprocessor.pkl")
    feast_dataset_path = os.path.join(feast_data.path, "preprocessed_data.parquet")


    train_df, test_df, preprocessor_obj = preprocess_data(
        df_path=input_path,
        parquet_output_path=feast_dataset_path
    )

    
    train_df.to_csv(train_output_path, index=False)
    test_df.to_csv(test_output_path, index=False)
    joblib.dump(preprocessor_obj, preprocessor_output_path)

    print("Preprocessing completed.")
    print(f"Train data saved to: {train_output_path}")
    print(f"Test data saved to: {test_output_path}")
    print(f"Scaler saved to: {preprocessor_output_path}")
    print(f"Feast Parquet saved at: {feast_dataset_path}")
