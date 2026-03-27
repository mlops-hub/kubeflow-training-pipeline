import argparse
import json
import os
import joblib
import pandas as pd
import boto3
from botocore.client import Config
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from _mlflow.registry import MLflowRegistry
from dotenv import load_dotenv
from feast import FeatureStore

load_dotenv()


def download_local_or_minio(file_path: str):
    # Handle minio:// URL format
    if file_path.startswith("minio://"):
        file_path = file_path[8:]  # Remove 'minio://' prefix
        result_path = download_from_minio(file_path)
        return result_path
    # Handle /minio/ path format
    elif file_path.startswith("/minio/"):
        file_path = file_path[7:]  # Remove '/minio/' prefix
        result_path = download_from_minio(file_path)
        return result_path
    else:
        print("\nDownloading from local.....")
        return file_path


def download_from_minio(minio_path: str) -> str:
    """
    Download file from MinIO and return local path.
    Handles paths like: 
      - minio://mlpipeline/v2/artifacts/.../file.csv
      - /minio/mlpipeline/v2/artifacts/.../file.csv
    """

    print("\nDownloading files from MinIO...")
    # minio_path: mlpipeline/v2/artifacts/...
    parts = minio_path.split("/", 1)
    bucket = parts[0]  # mlpipeline
    key = parts[1] if len(parts) > 1 else ""  # v2/artifacts/...
    
    print(f"Downloading from MinIO: bucket={bucket}, key={key}")
    
    # MinIO client using S3 API
    s3 = boto3.client(
        's3',
        endpoint_url=os.environ.get('MINIO_ENDPOINT', 'http://minio-service.kubeflow:9000'),
        aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID', 'minio'),
        aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY', 'minio123'),
        config=Config(signature_version='s3v4'),
        region_name='us-east-1'
    )
    
    # Create temp directory if not exists
    os.makedirs('/tmp/minio_downloads', exist_ok=True)
    
    # Download to local file
    filename = os.path.basename(key)
    local_path = f"/tmp/minio_downloads/{filename}"
    
    s3.download_file(bucket, key, local_path)
    print(f"Downloaded to: {local_path}")
    
    return local_path



def training_data(
    train_path: str, 
    preprocessor_path: str, 
    best_params_path: str,
    mlflow_run_id: str,
    tracking_uri: str, 
    experiment_name: str, 
    artifact_name: str,
    feast_repo_path: str,
):
    print("=" * 50)
    print("Starting training job...")
    print(f"Train path: {train_path}")
    print(f"Preprocessor path: {preprocessor_path}")
    print(f"Best params path: {best_params_path}")
    print(f"MLflow Run ID: {mlflow_run_id}")
    print(f"Tracking URI: {tracking_uri}")
    print(f"Experiment: {experiment_name}")
    print("=" * 50)

    
    # Download files from MinIO
    # Note: Artifact URIs are folders, actual files are inside
    print(f"Checking file path to download from local or minio....")
    local_train = download_local_or_minio(train_path + "/06_preprocess_train_df.csv") # local use "06_preprocess_train_df.csv"
    local_preprocessor = download_local_or_minio(preprocessor_path + "/preprocessor.pkl")
    local_params = download_local_or_minio(best_params_path + "/best_parameters.json")
                                 
    # Load data
    print("\nLoading training data...")
    entity_df_train = pd.read_csv(local_train)

    # ---------- feast -------------
    entity_cols = ['employee_id', 'event_timestamp']
    entity_df_train = entity_df_train[entity_cols].copy()
    entity_df_train['event_timestamp'] = pd.to_datetime(entity_df_train['event_timestamp'])

    store = FeatureStore(repo_path=feast_repo_path, fs_yaml_file=f"{feast_repo_path}/feature_store.local.yaml")

    df_train = store.get_historical_features(
        entity_df=entity_df_train,
        features=store.get_feature_service("employee_attrition_features")
    ).to_df()

    print(df_train.head(3))

    # # save reference data
    # from src.monitoring.scripts.prod_save_reference_data import log_reference_data_postgres
    # log_reference_data_postgres(df_train)

    
    # ---------- end feast ----------

    X_train = df_train.drop(columns=['employee_id', 'event_timestamp', 'attrition'])
    y_train = df_train['attrition']
    
    # Load parameters
    print("\nLoading best parameters...")
    with open(local_params, 'r') as f:
        params = json.load(f)
    
    best_params = {
        key.replace("model__", ""): value
        for key, value in params.items()
        if key.startswith("model__")
    }
    print(f"Best params: {best_params}")
    
    # Load preprocessor
    print("\nLoading preprocessor...")
    preprocessor = joblib.load(local_preprocessor)


    # Create pipeline
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(**best_params))
    ])


    # Initialize MLflow registry
    registry = MLflowRegistry(
        tracking_uri=tracking_uri,
        experiment_name=experiment_name
    )


    # Train and log to MLflow
    with registry.start_run(run_name='model-training-run', run_id=mlflow_run_id):
        print("\nTraining the model...")
        pipeline.fit(X_train, y_train)

        print(f"Pipeline: {pipeline}")
        print("Training completed!")
        
        registry.log_model(
            model=pipeline,
            X_train=X_train,
            parameters=best_params,
            artifact_name=artifact_name,
            stage='training'
        )

        print("Model logged to MLflow!")
    
    print(f"\n{'=' * 50}")
    print("✓ Training complete")
    print(f"{'=' * 50}")
    return True


# Support both direct execution and module execution
if __name__ == "__main__":
    from pathlib import Path 

    parser = argparse.ArgumentParser()
    parser.add_argument("--feast_repo_path", required=True)
    parser.add_argument("--train_path", required=True)
    parser.add_argument("--preprocessor_path", required=True)
    parser.add_argument("--best_params_path", required=True)
    parser.add_argument("--mlflow_run_id", required=True)
    args = parser.parse_args()
    
    training_data(
        feast_repo_path=args.feast_repo_path,
        train_path=args.train_path,
        preprocessor_path=args.preprocessor_path,
        best_params_path=args.best_params_path,
        mlflow_run_id=args.mlflow_run_id,
        tracking_uri=os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow.mlflow:80"),
        experiment_name=os.environ.get("MLFLOW_EXPERIMENT_NAME", "employee-attrition-v1"),
        artifact_name=os.environ.get("MLFLOW_MODEL_NAME", "model-name"),
    )

# python -m src.model_development._08_training --feast_repo_path "_feast/feature_repo" --train_path "datasets/data-pipeline" --preprocessor_path "artifacts" --best_params_path "artifacts" --mlflow_run_id "c8c5f3c57ebf4a98bcccec085f0c578f"