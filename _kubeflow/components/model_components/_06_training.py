from kfp.dsl import component, Dataset, Input, Output, Model

@component(
    base_image="sandy345/kubeflow-employee-attrition:latest",
    packages_to_install=['kubernetes', 'scikit-learn']
)
def trainer_model_component(
    job_name: str,
    namespace: str,
    image: str,
    cpu: str,
    memory: str,
    train_path: Input[Dataset],
    base_model: Output[Model],
    feature_store_path: Output[Model]
):
    import os
    from kubernetes import client, config
    
    config.load_incluster_config()
    api = client.CustomObjectsApi()

    # Access the actual file path in container
    train_df_path = os.path.join(train_path.path, "train.csv")

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
        "kind": "TrainJob",
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

    with open(base_model.path, "w") as f:
        f.write(model_output_uri)

    with open(feature_store_path.path, "w") as f:
        f.write(feature_output_uri)

    print(f"✅ Trainer Job {job_name} created in namespace {namespace}")
