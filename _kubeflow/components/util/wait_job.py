from kfp.dsl import component

@component
def wait_for_training(job_name: str, namespace: str):
    from kubernetes import client, config
    import time

    config.load_incluster_config()
    api = client.CustomObjectsApi()

    time.sleep(10)

    while True:
        job = api.get_namespaced_custom_object(
            group="trainer.kubeflow.org",
            version="v1alpha1",
            namespace=namespace,
            plural="trainjobs",
            name=job_name,
        )

        status = job.get("status", {}).get("phase")
        print('status: ', status)

        if status == "Succeeded":
            print("Trianing completed successfully")
            break
        elif status == "Failed":
            raise RuntimeError("Training failed")
        else:
            print("Training still running...")
            time.sleep(60)