from kfp.dsl import component
from _kubeflow.config import BASE_IMAGE

@component(
    base_image=BASE_IMAGE,
)
def run_retrain(
    drift_score: float,
    prediction_drift_score: float,
    should_retrain: bool,
    pipeline_endpoint: str,
    experiment_id: str,
    reference_run_id: str,
    drift_threshold: float,
    prediction_drift_threshold: float,
    # training pipeline parameters
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
    import requests
    from datetime import datetime

    # slack_webhook = None  # set via env if needed

    def send_alert(msg):
        print(f"Alert: {msg}")
        # you can use slack, email webhooks if you want alerts outside of Kubeflow UI
        

    if float(prediction_drift_score) < prediction_drift_threshold:
        send_alert(f"Prediction drift detected! p-value: {prediction_drift_score:.6f} (below threshold {prediction_drift_threshold})")
    else:
        print(f"No significant prediction drift (p-value: {prediction_drift_score:.4f})")


    if bool(should_retrain):
        send_alert(f"Alert: Severe drift detected, score: {drift_score*100:.1f}% of columns have drfited. Retraining recommended.")

        # trigger retraining
        try:
            # 1. Fetch the existing run to extract its pipeline spec
            existing_run_resp = requests.get(
                f"{pipeline_endpoint}/apis/v2beta1/runs/{reference_run_id}",
                timeout=10,
            )
            existing_run_resp.raise_for_status()
            existing_run = existing_run_resp.json()

            print("Existing run:", existing_run)  # inspect this first

            # 2. Extract the pipeline spec from the existing run
            pipeline_spec = existing_run.get("pipeline_spec")  # or "pipeline_version_reference"

            # 3. Submit a new run cloning the same spec
            payload = {
                "display_name": f"retrain-drift-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                "experiment_id": experiment_id,
                "pipeline_spec": pipeline_spec,   # reuse same spec from existing run
                "runtime_config": {
                    "parameters": {
                        "namespace":        namespace,
                        "tracking_uri":     tracking_uri,
                        "experiment_name":  experiment_name,
                        "artifact_name":    artifact_name,
                        "registry_name":    registry_name,
                        "recall_threshold": recall_threshold,
                        "feast_repo_path":  feast_repo_path,
                        "minio_endpoint":   minio_endpoint,
                        "minio_access_key": minio_access_key,
                        "minio_secret_key": minio_secret_key,
                        "is_retrain":       True,
                    }
                },
            }

            resp = requests.post(
                f"{pipeline_endpoint}/apis/v2beta1/runs",
                json=payload,
                timeout=30,
            )
            print(f"Response [{resp.status_code}]: {resp.text}")
            resp.raise_for_status()
            print("New run ID:", resp.json().get("run_id"))

        except Exception as e:
            print(f"Failed to trigger retraining: {e}")

    elif float(drift_score) > drift_threshold:
        send_alert(f"Warning: Drift detected in {drift_score*100:.1f}% of columns.")
    else:
        print("No significant drift.")

