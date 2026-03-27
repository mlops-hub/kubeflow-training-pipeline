from kfp import dsl
from kfp.dsl import component, Input, Dataset
from _kubeflow.config import BASE_IMAGE

@component(
    base_image=BASE_IMAGE
)
def feast_sync_component(
    feast_data: Input[Dataset],
    feast_repo_path: str,
    minio_endpoint: str,
    minio_access_key: str,
    minio_secret_key: str,
):
    import os

    # Must be set BEFORE any feast/pyarrow/boto3 imports
    os.environ["AWS_ACCESS_KEY_ID"] = minio_access_key
    os.environ["AWS_SECRET_ACCESS_KEY"] = minio_secret_key
    os.environ["AWS_DEFAULT_REGION"] = "us-east-1"          # dummy, but required
    os.environ["AWS_S3_ENDPOINT"] = minio_endpoint
    os.environ["FEAST_S3_ENDPOINT_URL"] = minio_endpoint

    from src.model_development.feast_sync import sync_to_feast

    
    print(f"MinIO Configuration:")
    print(f"  Endpoint: {minio_endpoint}")
    print(f"  Access Key: {minio_access_key[:4]}...")
    print(f"  Region: us-east-1")
    
    # KFP usually gives URIs like:
    #   minio://mlpipeline/v2/artifacts/<wf>/<op>/feast_data/
    uri = feast_data.uri.rstrip('/')

    if uri.startswith("minio://"):
        bucket_and_prefix = uri[len("minio://"):] # "mlpipeline/v2/artifacts/..."
        parquet_uri = f"s3://{bucket_and_prefix}/preprocessed_data.parquet"
    elif uri.startswith("s3://"):
        parquet_uri = f"{uri}/preprocessed_data.parquet"
    else:
        # Local/dev fallback
        parquet_uri = os.path.join(feast_data.path, "preprocessed_data.parquet")
    

    print(f"Feast offline path (MinIO): {parquet_uri}")

    sync_to_feast(
        parquet_path=parquet_uri,
        feast_repo_path=feast_repo_path,
    )