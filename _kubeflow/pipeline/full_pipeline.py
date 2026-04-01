from kfp import dsl, kubernetes
from kfp.compiler import Compiler
import uuid

# data
from _kubeflow.components.data_components._01_ingestion import ingestion_component
from _kubeflow.components.data_components._02_validation import validation_component
from _kubeflow.components.data_components._03_cleaning import cleaned_component
from _kubeflow.components.data_components._04_feature_engg import feature_engg_component
from _kubeflow.components.data_components._05_preprocessing import preprocessed_component

# model
from _kubeflow.components.model_components.feast_sync import feast_sync_component
from _kubeflow.components.model_components._07_training import trainer_model_component
from _kubeflow.components.model_components._08_evaluation import evaluation_component
from _kubeflow.components.model_components._06_tuning import tuning_component
from _kubeflow.components.model_components._09_register import register_model_component
# from _kubeflow.components.monitor_component.monitor_components import run_monitor_comp

# util
from _kubeflow.components.util.wait_job import wait_for_training


@dsl.pipeline( 
    name="Employee Attrition Full Pipeline", 
    description="Data  Feast_Registry  Training  Tuning  Evaluation  MLflow Registry"
)
def full_pipeline(
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
    is_retrain: bool = False,
):
    # data pipeline
    # -----------------------------------------------------
    ingest = ingestion_component(
        is_retrain=is_retrain
    )
    
    validate = validation_component(
        is_retrain=is_retrain,
        input_data=ingest.outputs['output_data']
    )

    cleaned = cleaned_component(
        input_data=validate.outputs['output_data']
    )

    feature_engg = feature_engg_component(
        input_data=cleaned.outputs['output_data']
    )
    
    preprocess = preprocessed_component(
        input_data=feature_engg.outputs['output_data']
    )

    # preprocess outputs: 
    # - train_data 
    # - test_data 
    # - preprocessor_model
    # - feast_data => output_parquet

    # Feast Setup
    # ----------------------------------------------------
    feast_sync = feast_sync_component(
        feast_data=preprocess.outputs['feast_data'],
        feast_repo_path=feast_repo_path,
        minio_endpoint=minio_endpoint,
        minio_access_key=minio_access_key,
        minio_secret_key=minio_secret_key,
    )
    feast_sync.set_caching_options(False)
    
    # model pipeline
    # ----------------------------------------------------
    tuning = tuning_component(
        feast_repo_path=feast_repo_path,
        train_data=preprocess.outputs['train_data'],
        test_data=preprocess.outputs['test_data'],
        preprocessor_model=preprocess.outputs['preprocessor_model'],
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
        minio_endpoint=minio_endpoint,
        minio_access_key=minio_access_key,
        minio_secret_key=minio_secret_key,
    ).after(feast_sync)
    # tune outputs: 
    # - tuning_metadata, 
    # - mlflow_metadata [run_id]


    # trainer job - kubeflow trainer
    train_job = trainer_model_component(
        feast_repo_path=feast_repo_path,
        train_path=preprocess.outputs['train_data'],
        preprocessor_model=preprocess.outputs['preprocessor_model'],
        best_parameters=tuning.outputs['best_parameters'],
        mlflow_run_id=tuning.outputs["mlflow_run_id"],
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
        artifact_name=artifact_name,
        minio_endpoint=minio_endpoint,
        minio_access_key=minio_access_key,
        minio_secret_key=minio_secret_key,
    ).after(tuning)
    train_job.set_caching_options(False)
    kubernetes.set_image_pull_policy(train_job, "Always")

    
    wait = wait_for_training(
        job_name=train_job.outputs['job_output'],
        namespace=namespace
    ).after(train_job)

    
    eval = evaluation_component(
        feast_repo_path=feast_repo_path,
        test_data=preprocess.outputs['test_data'],
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
        artifact_name=artifact_name,
        mlflow_run_id=tuning.outputs["mlflow_run_id"],
        minio_endpoint=minio_endpoint,
        minio_access_key=minio_access_key,
        minio_secret_key=minio_secret_key,
    ).after(wait)
    eval.set_caching_options(False)
    kubernetes.set_image_pull_policy(eval, "Always")

    
    reg = register_model_component(
        registry_name=registry_name,
        recall_threshold=recall_threshold,
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
        artifact_name=artifact_name,
        mlflow_run_id=tuning.outputs["mlflow_run_id"],
    ).after(eval)

    # # Monitoring step: run Evidently monitor after model registration
    # monitor = run_monitor_comp().after(reg)
    # monitor.set_caching_options(False)
    # kubernetes.set_image_pull_policy(monitor, "IfNotPresent")


# Compile pipeline 
# if __name__ == "__main__": 
#     Compiler().compile( 
#         pipeline_func=full_pipeline, 
#         package_path="full_pipeline.yaml" 
#     )
