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
    preprocessor_model: Input[Model],
    tuning_metadata: Input[Dataset],
    job_output: Output[str]
):
    import os
    from kubernetes import client, config
    
    config.load_incluster_config()
    api = client.CustomObjectsApi()

    # Access the actual file path in container
    train_path = os.path.join(train_path.path, "train.csv")
    preprocessor_model_path = os.path.join(preprocessor_model.path, "preprocessor.pkl")
    best_params_path = os.path.join(tuning_metadata.path, "tuning_metadata.json")

    # model_output_uri = f"s3://mlflow-artifacts/{job_name}/model.pkl" 

    command = ["python", "-m", "src.model_pipeline._08_training"]

    arguments = [
        "--train_path", train_path,
        "--preprocessor_path", preprocessor_model_path,
        "--best_params_path", best_params_path,
    ]

    env = [
        {"name": "MLFLOW_TRACKING_URI", "value": "http://mlflow.kubeflow.svc.cluster.local:80"},
        {"name": "MLFLOW_EXPERIMENT_NAME", "value": "employee-attrition-v1"},
    ]

    trainjob_manifest = {
        "apiVersion": "trainer.kubeflow.org/v1alpha1",
        "kind": "TrainJob",
        "metadata": {
            "name": job_name, 
            "namespace": namespace
        },
        "spec": {
            "runtimeRef": {
                "name": "torch-distributed"
            },
            "trainSpec": {
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

    print(f"✅ Trainer Job {job_name} created in namespace {namespace}")

    with open(job_output.path, 'w') as f:
        f.write(job_name)