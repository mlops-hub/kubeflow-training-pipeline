from kfp import dsl
from kfp.dsl import Input, Artifact, OutputPath, Dataset


@dsl.component(
    base_image="sandy345/final-kubeflow-pipeline:v1.0.0",
)
def trainer_model_component(
    feast_repo_path: str,
    train_path: Input[Dataset],
    preprocessor_model: Input[Artifact],
    best_parameters: Input[Artifact],
    mlflow_run_id: str,
    job_output: OutputPath(str),
    tracking_uri: str,
    experiment_name: str,
    artifact_name: str,
    minio_endpoint: str,
    minio_access_key: str,
    minio_secret_key: str,
):

    from kubeflow.trainer import TrainerClient, CustomTrainer

    def run_training(
        feast_repo_path: str,
        train_uri: str,
        preprocess_uri: str,
        best_params_uri: str,
        mlflow_run_id: str,
    ):
        import os
        from src.model_development._08_training import training_data

        training_data(
            train_path=train_uri,
            preprocessor_path=preprocess_uri,
            best_params_path=best_params_uri,
            mlflow_run_id=mlflow_run_id,
            tracking_uri=os.environ.get("MLFLOW_TRACKING_INTERNAL_URI", "http://mlflow.mlflow:80"),
            experiment_name=os.environ.get("MLFLOW_EXPERIMENT_NAME", "employee-attrition-v1"),
            artifact_name=os.environ.get("MLFLOW_MODEL_NAME", "model-name"),
            feast_repo_path=feast_repo_path,
        )


    client = TrainerClient()

    # Log available runtimes for debugging
    for r in client.list_runtimes():
        print(f"Available runtime: {r.name}")

    job_id = client.train(
        # runtime=client.get_runtime("torch-distributed"),
        trainer=CustomTrainer(
            image="sandy345/final-kubeflow-pipeline:v1.0.0",
            func=run_training,
            func_args={
                "feast_repo_path": feast_repo_path,
                "train_uri":        train_path.uri,
                "preprocess_uri":   preprocessor_model.uri,
                "best_params_uri":  best_parameters.uri,
                "mlflow_run_id":    mlflow_run_id,
            },
            num_nodes=1,
            resources_per_node={
                "cpu":    "1",
                "memory": "1Gi",
            },
            env={
                "MLFLOW_TRACKING_URI":   tracking_uri,
                "MLFLOW_EXPERIMENT_NAME": experiment_name,
                "MLFLOW_MODEL_NAME":     artifact_name,
                "MLFLOW_S3_ENDPOINT_URL": minio_endpoint,
                "MINIO_ENDPOINT":        minio_endpoint,
                "AWS_ACCESS_KEY_ID":     minio_access_key,
                "AWS_SECRET_ACCESS_KEY": minio_secret_key,
                "FEAST_REGISTRY_URL": "postgresql+psycopg://feast:feast@postgres.feast.svc.cluster.local:5432/feast",
                "FEAST_REDIS_URL": "redis.feast.svc.cluster.local:6379,username=default,password=changeMeVeryStrong,db=0",
                "FEAST_S3_ENDPOINT_URL": minio_endpoint,
                "S3_ENDPOINT_URL": minio_endpoint,
            }
        ),
    )

    print(f"Created TrainJob: {job_id}")
    
    with open(job_output, "w") as f:
        f.write(job_id)
