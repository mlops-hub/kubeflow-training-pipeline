from kfp.dsl import component, Input, Dataset, Output
from _kubeflow.config import BASE_IMAGE


# component-1
@component(
    base_image=BASE_IMAGE,
    # packages_to_install=['sqlalchemy', 'psycopg2-binary', 'python-dotenv']
)
def get_live_data(
    live_data: Output[Dataset]
):
    from sqlalchemy import create_engine
    import os
    from dotenv import load_dotenv
    import pandas as pd

    load_dotenv()

    POSTGRES_URI = os.environ.get(
        "POSTGRES_URI_EXTERNAL",
        "postgresql+psycopg://feast:feast@postgres.feast.svc.cluster.local:5432/feast"
        # "postgresql+psycopg2://feast:feast@postgres.feast.svc.cluster.local:5432/feast"
    )
    print(POSTGRES_URI)
    engine = create_engine(POSTGRES_URI)

    df = pd.read_sql("SELECT * FROM live_data", engine)
    print(df.head())

    df.to_csv(live_data.path, index=False)


@component(
    base_image=BASE_IMAGE,
)
def get_reference_data(
    ref_data: Output[Dataset]
):
    import pandas as pd
    import boto3
    from botocore.client import Config
    from src.monitoring.scripts.prod_save_reference_data import log_reference_data_postgres

    bucket = "mlpipeline"
    key = "v2/artifacts/employee-attrition-full-pipeline/90db58cc-4685-4029-9975-963032d1a9c7/preprocessed-component/6cf2fedf-676a-4838-ba9c-27bff34b1893/train_data/train.csv"

    s3 = boto3.client(
        "s3",
        endpoint_url="http://minio-service.kubeflow:9000",
        aws_access_key_id="minio",
        aws_secret_access_key="minio123",
        config=Config(signature_version="s3v4"),
        region_name="us-east-1"
    )

    local_path = "/tmp/train.csv"
    s3.download_file(bucket, key, local_path)

    df = pd.read_csv(local_path)
    print(df.head())

    # df_postgres = df.copy()
    # log_reference_data_postgres(df_postgres)
    # print('✅ got data from postgres...')

    df.to_csv(ref_data.path, index=False)


# component-3
@component(
    base_image=BASE_IMAGE
)
def run_monitor_comp():
    import os
    from src.monitoring.evidently_monitor.index import MonitorPipeline

    POSTGRES_URI = os.environ.get(
        "POSTGRES_URI_EXTERNAL",
        "postgresql+psycopg2://feast:feast@68.183.87.245:30032/feast"
    )

    pipeline = MonitorPipeline(
        ev_token=os.environ.get("EVIDENTLY_TOKEN", ""),
        ev_url=os.environ.get("EVIDENTLY_URL", "https://app.evidently.cloud"),
        org_id=os.environ.get("EVIDENLTY_ORG_ID", ""),
        project_id=os.environ.get("EVIDENTLY_PROJECT_ID", ""),
        postgres_uri=POSTGRES_URI,
        ref_path=Input[Dataset],
        live_path=Input[Dataset],
        output_path=Output[Dataset],
    )
    pipeline.run_daily()
