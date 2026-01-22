from kfp.dsl import component, Dataset, Input, Output, Model

@component(
    base_image="python:3.10-slim",
    packages_to_install=['kubernetes', 'scikit-learn', "git+https://github.com/mlops-hub/kubeflow-training-pipeline.git"]
)
def trainer_model_component(
    job_name: str,
    namespace: str,
    image: str,
    cpu: str,
    memory: str,
    train_path: Input[Dataset],
    model_path: Output[Model],
    feature_store_path: Output[Model]
):
    from kubernetes import client, config
    
    config.load_incluster_config()
    api = client.CustomObjectsApi()

    # Access the actual file path in container
    train_df_path = train_path.path
    model_output_uri = f"s3://mlflow-artifacts/{job_name}/model.pkl" 
    feature_output_uri = f"s3://mlflow-artifacts/{job_name}/features.pkl"

    command = ["python", "-m", "src.model_pipeline._07_training"]

    arguments = [
        "--train_path", train_df_path,
        "--model_path", model_output_uri,
        "--feature_store_path", feature_output_uri
    ]

    env = []

    trainjob_manifest = {
        "apiVersion": "trainer.kubeflow.org/v1alpha1",
        "kind": "TrainerJob",
        "metadata": {"name": job_name, "namespace": namespace},
        "spec": {
            "trainer": {
                "image": image,
                "command": command,
                "args": arguments,
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
                "env": env,
            }
        }
    }

    api.create_namespaced_custom_object(
        group="trainer.kubeflow.org",
        version="v1alpha1",
        namespace=namespace,
        plural="trainjobs",
        body=trainjob_manifest,
    )

    with open(model_path.path, "w") as f:
        f.write(model_output_uri)

    with open(feature_store_path.path, "w") as f:
        f.write(feature_output_uri)

    print(f"✅ Trainer Job {job_name} created in namespace {namespace}")
