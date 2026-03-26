from kfp import dsl

# util
from _kubeflow.components.monitor_component.monitor_components import get_live_data, get_reference_data


@dsl.pipeline( 
    name="Employee Attrition Monitoring Pipeline", 
    description="Monitoring the model performance and drift using Evidently"
)
def monitor_pipeline(
    namespace: str = "kubeflow",
    tracking_uri: str = "http://mlflow.mlflow.svc.cluster.local:80",
    experiment_name: str = "employee-attrition",
    artifact_name: str = "employee-attrition-model",
    registry_name: str = "register-employee-attrition-model",
    recall_threshold: float = 0.65,
    feast_repo_path: str = "_feast/feature_repo",
    minio_endpoint: str = "http://minio-service.kubeflow:9000",
    minio_access_key: str = "minio",
    minio_secret_key: str = "minio123",
):
    
    live_data = get_live_data()

   



# Compile pipeline 
# if __name__ == "__main__": 
#     Compiler().compile( 
#         pipeline_func=monitor_pipeline, 
#         package_path="monitor_pipeline.yaml" 
#     )
