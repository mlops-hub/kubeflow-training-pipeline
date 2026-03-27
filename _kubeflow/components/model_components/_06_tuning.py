from kfp import dsl
from kfp.dsl import Input, Output, OutputPath, Model, Dataset
from _kubeflow.config import BASE_IMAGE

@dsl.component(
    base_image=BASE_IMAGE
)
def tuning_component(
    feast_repo_path: str,
    train_data: Input[Dataset],
    test_data: Input[Dataset],
    preprocessor_model: Input[Model],
    best_parameters: Output[Dataset],
    tracking_uri: str,
    experiment_name: str,
    mlflow_run_id: OutputPath(str),
    minio_endpoint: str,
    minio_access_key: str,
    minio_secret_key: str,
):
    import os
    import json

    # Must be set BEFORE any feast/pyarrow/boto3 imports
    os.environ["AWS_ACCESS_KEY_ID"] = minio_access_key
    os.environ["AWS_SECRET_ACCESS_KEY"] = minio_secret_key
    os.environ["AWS_DEFAULT_REGION"] = "us-east-1"          # dummy, but required
    os.environ["AWS_S3_ENDPOINT"] = minio_endpoint
    os.environ["FEAST_S3_ENDPOINT_URL"] = minio_endpoint

    from src.model_development._07_tuning import tuning_data

    os.makedirs(best_parameters.path, exist_ok=True)

    train_path = os.path.join(train_data.path, "train.csv")
    test_path = os.path.join(test_data.path, "test.csv")
    preprocessor_output = os.path.join(preprocessor_model.path, "preprocessor.pkl")
    best_params_path = os.path.join(best_parameters.path, "best_parameters.json")

    run_id, best_parameters_output = tuning_data(
        feast_repo_path=feast_repo_path,
        train_path=train_path, 
        test_path=test_path, 
        preprocess_path=preprocessor_output,
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
    )

    with open(best_params_path, 'w') as f:
        json.dump(best_parameters_output, f)


    with open(mlflow_run_id, "w") as f:
        f.write(run_id)
            
    print("Tuning completed successfully.")

