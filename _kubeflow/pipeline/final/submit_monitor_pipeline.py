import os
from dotenv import load_dotenv
import kfp
from _kubeflow.pipeline.monitor_pipeline import monitor_pipeline

load_dotenv()

PIPELINE_ENDPOINT = os.environ.get("PIPELINE_ENDPOINT", "http://localhost:4040")
print('Pipeline Endpoint: ', PIPELINE_ENDPOINT)

EXPERIMENT_NAME = "monitoring-pipeline-v1.0.2"

def submit_monitor_pipeline():
    client = kfp.Client(host=PIPELINE_ENDPOINT)

    try:
        experiment = client.get_experiment(experiment_name=EXPERIMENT_NAME)
        print('experiment: ', experiment)
    except Exception:
        experiment = client.create_experiment(name=EXPERIMENT_NAME)
        print('experiment: ', experiment)
    
    run = client.create_run_from_pipeline_func(
        pipeline_func=monitor_pipeline,
        arguments={},
        experiment_name=EXPERIMENT_NAME,
        run_name="monitoring_pipeline1",
        enable_caching=False,
    )

    print('run-id: ', run.run_id)
    print(f'run url: {PIPELINE_ENDPOINT}/#/runs/details/{run.run_id}')

if __name__ == "__main__":
    submit_monitor_pipeline()


# python -m _kubeflow.pipeline.final.submit_monitor_pipeline