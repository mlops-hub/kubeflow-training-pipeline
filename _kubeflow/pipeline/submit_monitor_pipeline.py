"""
Utility to submit the monitoring pipeline as a recurring Kubeflow Pipelines run.
Creates a recurring run via the KFP SDK and also includes a CronWorkflow manifest example.
"""
from kfp import Client
from kfp_server_api.rest import ApiException
import os

# Adjust these to your environment
KFP_HOST = os.environ.get("KFP_HOST", "http://ml-pipeline.kubeflow.svc.cluster.local:8888")
NAMESPACE = os.environ.get("KFP_NAMESPACE", "kubeflow")
PIPELINE_PACKAGE = os.environ.get("MONITOR_PIPELINE_PACKAGE", "monitor_pipeline.yaml")
PIPELINE_NAME = os.environ.get("MONITOR_PIPELINE_NAME", "Employee Attrition Monitor")


def create_recurring_run():
    client = Client(host=KFP_HOST, namespace=NAMESPACE)
    try:
        pipeline = client.list_pipelines(page_size=100).pipelines
    except Exception:
        pipeline = []

    # If pipeline package exists, upload and create a pipeline
    if os.path.exists(PIPELINE_PACKAGE):
        pipeline = client.upload_pipeline(pipeline_package_path=PIPELINE_PACKAGE, pipeline_name=PIPELINE_NAME)
        pipeline_id = pipeline.id
    else:
        # try to find by name
        found = [p for p in (pipeline or []) if p.name == PIPELINE_NAME]
        if not found:
            raise RuntimeError("Pipeline package not found and pipeline not already present. Set MONITOR_PIPELINE_PACKAGE or upload manually.")
        pipeline_id = found[0].id

    # Create a recurring run (daily)
    experiment = client.create_experiment(name="monitoring-experiments")
    recurring = client.create_recurring_run(
        experiment_id=experiment.id,
        job_name=f"{PIPELINE_NAME}-daily",
        description="Daily Evidently monitoring run",
        pipeline_id=pipeline_id,
        cron_expression="0 6 * * *",  # every day at 06:00 UTC
        max_concurrency=1,
    )
    print("Created recurring run:", recurring)


if __name__ == '__main__':
    create_recurring_run()
