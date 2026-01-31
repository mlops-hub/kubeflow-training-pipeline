from kfp.dsl import component

@component(
    base_image="sandy345/kubeflow-employee-attrition:latest",
    packages_to_install=['kubernetes']
)
def wait_for_training(job_name: str, namespace: str):
    from kubernetes import client, config
    import time

    config.load_incluster_config()
    api = client.CustomObjectsApi()

    time.sleep(5)
    poll_interval = 30
    elapsed = 0

    while elapsed < 1800:
        job = api.get_namespaced_custom_object(
            group="trainer.kubeflow.org",
            version="v1alpha1",
            namespace=namespace,
            plural="trainjobs",
            name=job_name,
        )

        status = job.get("status", {})
        print('status: ', status)

        conditions = status.get("conditions", [])
        print('conditions: ', conditions)

        for condition in conditions:
            cond_type = condition.get("type", "")
            cond_status = condition.get("status", "")

            if cond_type == "Succeeded" and cond_status == "True":
                print(f"TrainJob '{job_name}' succeeded!")
                break

            elif cond_type == "Failed" and cond_status == "True":
                msg = condition.get("message", "Unknown error")
                raise Exception(f"TrainJob '{job_name}' failed: {msg}")

            elif cond_type == "Complete" and cond_status == "True":
                print(f"TrainJob '{job_name}' completed!")
                break
            
        print(f"Status: Running... ({elapsed}s elapsed)")
                
        time.sleep(poll_interval)
        elapsed += poll_interval

    raise Exception(f"TrainJob '{job_name}' timed out after 1800s")

