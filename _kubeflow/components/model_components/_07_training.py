from kfp import dsl
from kfp.dsl import Input, Artifact, InputPath, OutputPath
from kubernetes import client as k8s_client
from kubernetes.client import V1EnvVar, V1EnvVarSource, V1SecretKeySelector


@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=["kubernetes"]
)
def trainer_model_component(
    job_name: str,
    namespace: str,
    image: str,
    cpu: str,
    memory: str,
    train_path: Input[Artifact],
    preprocessor_model: Input[Artifact],
    tuning_metadata: Input[Artifact],
    mlflow_metadata: str,
    job_output: OutputPath(str),
    tracking_uri: str,
    experiment_name: str,
    artifact_name: str,
):
    """Creates a Kubeflow TrainJob for model training with MinIO access."""
    
    from kubernetes import client, config
    
    # Load in-cluster config
    config.load_incluster_config()
    
    # Create custom objects API
    api = client.CustomObjectsApi()
    
    # TrainJob specification with MinIO credentials
    train_job = {
        "apiVersion": "trainer.kubeflow.org/v1alpha1",
        "kind": "TrainJob",
        "metadata": {
            "name": job_name,
            "namespace": namespace
        },
        "spec": {
            "suspend": False,
            "runtimeRef": {
                "apiGroup": "trainer.kubeflow.org",
                "kind": "ClusterTrainingRuntime",
                "name": "torch-distributed"
            },
            "trainer": {
                "image": image,
                "command": ["python", "-m", "src.model_pipeline._08_training"],
                "args": [
                    "--train_path", train_path.uri,
                    "--preprocessor_path", preprocessor_model.uri,
                    "--best_params_path", tuning_metadata.uri,
                    "--mlflow_run_id", mlflow_metadata,
                ],
                "numNodes": 1,
                "resourcesPerNode": {
                    "requests": {
                        "cpu": cpu,
                        "memory": memory
                    },
                    "limits": {
                        "cpu": cpu,
                        "memory": memory
                    }
                },
                "env": [
                    # MLflow settings
                    {"name": "MLFLOW_TRACKING_URI", "value": str(tracking_uri)},
                    {"name": "MLFLOW_EXPERIMENT_NAME", "value": str(experiment_name)},
                    {"name": "MLFLOW_MODEL_NAME", "value": str(artifact_name)},
                    # MinIO settings
                    {"name": "MINIO_ENDPOINT", "value": "http://minio-service.kubeflow:9000"},
                    # MinIO credentials from secret
                    {
                        "name": "AWS_ACCESS_KEY_ID",
                        "valueFrom": {
                            "secretKeyRef": {
                                "name": "mlpipeline-minio-artifact",
                                "key": "accesskey"
                            }
                        }
                    },
                    {
                        "name": "AWS_SECRET_ACCESS_KEY",
                        "valueFrom": {
                            "secretKeyRef": {
                                "name": "mlpipeline-minio-artifact",
                                "key": "secretkey"
                            }
                        }
                    }
                ]
            }
        }
    }
    
    # Create the TrainJob
    api.create_namespaced_custom_object(
        group="trainer.kubeflow.org",
        version="v1alpha1",
        namespace=namespace,
        plural="trainjobs",
        body=train_job
    )
    
    print(f"Created TrainJob: {job_name}")
    
    # Write job name to output
    with open(job_output, "w") as f:
        f.write(job_name)