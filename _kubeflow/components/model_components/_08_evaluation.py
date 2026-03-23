
from kfp.dsl import component, Input, InputPath, Dataset


@component(
    base_image="sandy345/final-kubeflow-pipeline:v1.0.0"
)
def evaluation_component(
    feast_repo_path: str,
    test_data: Input[Dataset],
    tracking_uri: str,
    experiment_name: str,
    artifact_name: str,
    mlflow_run_id: str,
    minio_endpoint: str,
    minio_access_key: str,
    minio_secret_key: str,
):
    import os
    from src.model_development._09_evaluation import evaluate_data

    # Must be set BEFORE any feast/pyarrow/boto3 imports
    os.environ["AWS_ACCESS_KEY_ID"] = minio_access_key
    os.environ["AWS_SECRET_ACCESS_KEY"] = minio_secret_key
    os.environ["AWS_DEFAULT_REGION"] = "us-east-1"          # dummy, but required
    os.environ["AWS_S3_ENDPOINT"] = minio_endpoint
    os.environ["FEAST_S3_ENDPOINT_URL"] = minio_endpoint

    test_path = os.path.join(test_data.path, "test.csv")

    metrics = evaluate_data(
        feast_repo_path=feast_repo_path,
        test_path=test_path, 
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
        artifact_name=artifact_name,
        mlflow_run_id=mlflow_run_id
    )

    print(f"Evaluation is completed. Got accuracy: {metrics['recall']}")

