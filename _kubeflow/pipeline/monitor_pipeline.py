from kfp import dsl
from kfp.compiler import Compiler

# util
from _kubeflow.components.monitor_component.run_monitor import run_monitoring
from _kubeflow.components.monitor_component.retrain import run_retrain

@dsl.pipeline( 
    name="Employee Attrition Monitoring Pipeline", 
    description="Monitoring the model performance and drift using Evidently"
)
def monitor_pipeline(
    pipeline_endpoint: str    = "http://ml-pipeline.kubeflow.svc.cluster.local:8888",
    experiment_id: str = "0758c94d-2635-4d90-bc52-2ad6bb44aa7d",
    reference_run_id: str = "c8b51e32-8371-4d37-b3e0-15e812d1e5ac", 
    retrain_threshold: float  = 0.25, # 0.5 (False), 0.25 (Autoretrain)
    drift_threshold: float    = 0.25,
    prediction_drift_threshold: float = 0.05,
    # ── pass-through params for full_pipeline ────────────
    namespace: str            = "kubeflow",
    tracking_uri: str         = "http://mlflow.mlflow.svc.cluster.local:80",
    experiment_name: str      = "employee-attrition",
    artifact_name: str        = "employee-attrition-model",
    registry_name: str        = "register-employee-attrition-model",
    recall_threshold: float   = 0.65,
    feast_repo_path: str      = "_feast/feature_repo",
    minio_endpoint: str       = "http://minio-service.kubeflow:9000",
    minio_access_key: str     = "minio",
    minio_secret_key: str     = "minio123",
):
    import os
    from dotenv import load_dotenv

    load_dotenv()

    POSTGRES_URI = os.environ.get(
        "POSTGRES_URI_INTERNAL",
        "postgresql+psycopg://feast:feast@postgres.feast.svc.cluster.local:5432/feast"
    )

    monitoring = run_monitoring(
        postgres_uri=POSTGRES_URI,
        evidently_token="sk_prod.019d410b-60f2-7920-ba94-aa41f9cd4100.ENVytK1EW8gPB3BqUdsC3f3WlpNaddAXWEYUqdKdTHCVK0YK5JAReFCrB-ANtLytCUdBN-wQRa1VRs-grtAvtw-qK5RrLmqXiGwhHaeZ9qZhjMl67lR4soaTmXrxbhUM",
        evidently_url="https://app.evidently.cloud",
        evidently_org_id="019d2071-06a2-79b9-b627-2c23f3cdb8c5",
        evidently_project_id="019d212d-30ad-7f4b-bb92-37b7049ab60e",  # optional
        retrain_threshold=retrain_threshold,
    )


    trigger = run_retrain(
        drift_score=monitoring.outputs['drift_score'],
        prediction_drift_score=monitoring.outputs['prediction_drift_score'],
        should_retrain=monitoring.outputs['should_retrain'],
        pipeline_endpoint=pipeline_endpoint,
        experiment_id=experiment_id,
        reference_run_id=reference_run_id,
        drift_threshold=drift_threshold,
        prediction_drift_threshold=prediction_drift_threshold,
        # training parameters
        namespace=namespace,
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
        artifact_name=artifact_name,
        registry_name=registry_name,
        recall_threshold=recall_threshold,
        feast_repo_path=feast_repo_path,
        minio_endpoint=minio_endpoint,
        minio_access_key=minio_access_key,
        minio_secret_key=minio_secret_key,
    ).after(monitoring).set_caching_options(False)


# Compile pipeline 
# if __name__ == "__main__": 
#     Compiler().compile( 
#         pipeline_func=monitor_pipeline, 
#         package_path="monitor_pipeline.yaml" 
#     )
