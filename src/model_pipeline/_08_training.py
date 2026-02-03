import argparse
import json
import os
import tempfile
import joblib
import pandas as pd
import boto3
from botocore.client import Config
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from _mlflow.registry import MLflowRegistry


def download_from_minio(minio_path: str) -> str:
    """
    Download file from MinIO and return local path.
    Handles paths like: 
      - minio://mlpipeline/v2/artifacts/.../file.csv
      - /minio/mlpipeline/v2/artifacts/.../file.csv
    """
    # Parse minio path
    path_clean = minio_path
    
    # Handle minio:// URL format
    if path_clean.startswith("minio://"):
        path_clean = path_clean[8:]  # Remove 'minio://' prefix
    # Handle /minio/ path format
    elif path_clean.startswith("/minio/"):
        path_clean = path_clean[7:]  # Remove '/minio/' prefix
    
    # Now path_clean is: mlpipeline/v2/artifacts/...
    parts = path_clean.split("/", 1)
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
    tracking_uri: str, 
    experiment_name: str, 
    artifact_name: str
):
    print("=" * 50)
    print("Starting training job...")
    print(f"Train path: {train_path}")
    print(f"Preprocessor path: {preprocessor_path}")
    print(f"Best params path: {best_params_path}")
    print(f"Tracking URI: {tracking_uri}")
    print(f"Experiment: {experiment_name}")
    print("=" * 50)
    
    # Download files from MinIO
    # Note: Artifact URIs are folders, actual files are inside
    print("\nDownloading files from MinIO...")
    local_train = download_from_minio(train_path + "/train.csv")
    local_preprocessor = download_from_minio(preprocessor_path + "/preprocessor.pkl")
    local_params = download_from_minio(best_params_path + "/tuning_metadata.json")
    
    # Initialize MLflow registry
    registry = MLflowRegistry(
        tracking_uri=tracking_uri,
        experiment_name=experiment_name
    )
    
    # Load data
    print("\nLoading training data...")
    df = pd.read_csv(local_train)
    X_train = df.drop(columns=['Attrition'])
    y_train = df['Attrition']
    print(f"Training data shape: {X_train.shape}")
    
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
    
    # Train and log to MLflow
    with registry.start_run(run_name='model-training-run'):
        print("\nTraining the model...")
        pipeline.fit(X_train, y_train)
        print(f"Pipeline: {pipeline}")
        print("Training completed!")
        
        registry.log_model(
            model=pipeline,
            X_train=X_train,
            parameters=best_params,
            artifact_name=artifact_name,
        )
        print("Model logged to MLflow!")
    
    print(f"\n{'=' * 50}")
    print("✓ Training complete")
    print(f"{'=' * 50}")
    return True


# Support both direct execution and module execution
if __name__ == "__main__" or __name__ == "src.model_pipeline._08_training":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", required=True)
    parser.add_argument("--preprocessor_path", required=True)
    parser.add_argument("--best_params_path", required=True)
    args = parser.parse_args()
    
    training_data(
        train_path=args.train_path,
        preprocessor_path=args.preprocessor_path,
        best_params_path=args.best_params_path,
        tracking_uri=os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow.mlflow:80"),
        experiment_name=os.environ.get("MLFLOW_EXPERIMENT_NAME", "employee-attrition-v1"),
        artifact_name="employee-attrition-model"
    )