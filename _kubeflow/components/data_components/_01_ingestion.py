from kfp.dsl import component, Output, Dataset
from _kubeflow.config import BASE_IMAGE

@component(
    base_image=BASE_IMAGE
)
def ingestion_component(
    output_data: Output[Dataset],
    is_retrain: bool,
):
    import os
    import boto3
    import pandas as pd
    from src.data_preparation._01_ingestion import ingestion


    s3 = boto3.client(
        "s3",
        endpoint_url="http://minio-service.kubeflow:9000",
        aws_access_key_id="minio",
        aws_secret_access_key="minio123",
    )

    tmp_path = "/tmp/employee_attrition.csv"

    # 
    if is_retrain:
        print("Running in retrain mode: Load live data from PostgreSQL")
        from sqlalchemy import create_engine

        # live data from PostgreSQL
        POSTGRES_URI="postgresql+psycopg://feast:feast@postgres.feast.svc.cluster.local:5432/feast"
        engine = create_engine(POSTGRES_URI)
        new_df = pd.read_sql("SELECT * FROM live_data", engine)

        # ── Fix 1: rename 'prediction' → 'attrition' in live data ────────────
        if "prediction" in new_df.columns:
            new_df = new_df.rename(columns={"prediction": "attrition"})

        # ── Fix 2: drop columns that don't belong in training data ────────────
        drop_cols = ["id", "target", "event_timestamp"]
        new_df = new_df.drop(columns=[col for c in drop_cols if c in new_df.columns], errors='ignore')

        # ── Fix 3: fix employee_id — generate synthetic IDs if None ──────────
        if new_df["employee_id"].isna().all():
            print("WARNING: employee_id is all None in live data, generating synthetic IDs")
            new_df["employee_id"] = range(900001, 900001 + len(new_df))
        else:
            new_df = new_df.dropna(subset=["employee_id"])
            new_df["employee_id"] = new_df["employee_id"].astype(int)
        
        print(f"Live data after cleanup: {new_df.shape}")
        print(f"Live data columns after cleanup: {new_df.columns.tolist()}")

        s3.download_file("mlpipeline", "datasets/feature_employee_attrition.csv", tmp_path)
        ref_df = pd.read_csv(tmp_path)

        # ── Fix 4: align columns — keep only shared columns + attrition ───────
        shared_cols = [c for c in ref_df.columns if c in new_df.columns]
        print(f"Shared columns: {shared_cols}")

        ref_df  = ref_df[shared_cols]
        new_df = new_df[shared_cols]

        combined_df = pd.concat([ref_df, new_df], ignore_index=True)
        combined_df.drop_duplicates(inplace=True)

        # ── Drop rows where label is missing ──────────────────────────────────
        before = len(combined_df)
        combined_df = combined_df.dropna(subset=["attrition"])
        print(f"Dropped {before - len(combined_df)} rows with null attrition.")
        print(f"Final combined shape: {combined_df.shape}")

        combined_df.to_csv(tmp_path, index=False)
    else:
        s3.download_file("mlpipeline", "datasets/employee_attrition.csv", tmp_path)

    # call ingestion ML logic
    df = ingestion(tmp_path)

    print(f"Post-ingestion shape: {df.shape}")
    print(f"Post-ingestion columns: {df.columns.tolist()}")

    # KFP v2: output_data.path is a DIRECTORY, not a file
    os.makedirs(output_data.path, exist_ok=True)
    output_path = os.path.join(output_data.path, "ingestion.csv")

    df.to_csv(output_path, index=False)

    print(f"Ingestion completed. Saved to {output_path}")
