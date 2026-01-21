from kfp.dsl import component

@component(
    base_image="python:3.11-slim",
    packages_to_install=['kubernetes']
)
def trainer_model_component(
    job_name: str,
    namespace: str,
    image: str,
    cpu: str,
    memory: str,
    train_path: str,
    model_path: str,
    feature_store_path: str
):
    from kubernetes import client, config
    
    config.load_incluster_config()
    api = client.CustomObjectsApi()

    command = ["python", "-m", "src.model_pipeline._07_training"]

    arguments = [
        "--train_path", train_path,
        "--model_path", model_path,
        "--feature_store_path", feature_store_path
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

    print(f"✅ Trainer Job {job_name} created in namespace {namespace}")
    return job_name