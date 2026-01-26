from kfp import dsl
from kfp.compiler import Compiler

# data
from _kubeflow.components.data_components._01_ingestion import ingestion_component
from _kubeflow.components.data_components._02_validation import validation_component
from _kubeflow.components.data_components._03_cleaning import cleaned_component
from _kubeflow.components.data_components._04_feature_engg import feature_engg_component
from _kubeflow.components.data_components._05_preprocessing import preprocessed_component

# model
from _kubeflow.components.model_components._06_training import trainer_model_component
from _kubeflow.components.model_components._07_evaluation import evaluation_component
from _kubeflow.components.model_components._08_cross_validation import cross_validation_component
from _kubeflow.components.model_components._09_tuning import tuning_component
from _kubeflow.components.model_components._10_register import register_model_component


@dsl.pipeline( 
    name="Employee Attrition Full Pipeline", 
    description="Data -> Training -> Tuning -> Evaluation -> MLflow Registry"
)
def full_pipeline(
    namespace: str = "_kubeflow-employee-attrition",
    trainer_image: str = "sandy345/kubeflow-employee-attrition:latest",
    cpu: str = "500m",
    memory: str = "1Gi",
    tracking_uri: str = "http://mlflow.kubeflow.svc.cluster.local:80",
    experiment_name: str = "employee-attrition-v1",
    registry_name: str = "register-employee-attrition-model",
    recall_threshold: float = 0.70,
):
    # data pipeline
    # -----------------------------------------------------
    ingest = ingestion_component()
    validate = validation_component(
        input_data=ingest.outputs['output_data']
    )
    cleaned = cleaned_component(
        input_data=validate.outputs['output_data']
    )
    transform = feature_engg_component(
        input_data=cleaned.outputs['output_data']
    )
    preprocess = preprocessed_component(
        input_data=transform.outputs['output_data']
    )

    # preprocess outputs: 
    # - train_data 
    # - test_data 
    # - scaler_model

    # model pipeline
    # ----------------------------------------------------

    # trainer job - kubeflow trainer
    train_job = trainer_model_component(
        job_name="attrition-trainer-job",
        namespace=namespace,
        image=trainer_image,
        cpu=cpu,
        memory=memory,
        train_path=preprocess.outputs['train_data'],
    )

    print(train_job)

    evaluation_component(
        train_data=preprocess.outputs['train_data'],
        test_data=preprocess.outputs['test_data'],
        base_model=train_job.outputs['base_model']
    )

    cross_validation_component(
        train_data=preprocess.outputs['train_data'],
        base_model=train_job.outputs['base_model']
    )

    tune = tuning_component(
        train_data=preprocess.outputs['train_data'],
        test_data=preprocess.outputs['test_data'],
        base_model=train_job.outputs['base_model']
    )

    # tune outputs: 
    # - tuned_model
    # - tuning_metadata

    # model register
    # --------------------------------------------------
    register_model_component(
        train_data=preprocess.outputs['train_data'],
        tuned_model=tune.outputs['tuned_model'],
        tuning_metadata=tune.outputs['tuning_metadata'],
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
        registry_name=registry_name,
        recall_threshold=recall_threshold,
    )


# Compile pipeline 
# if __name__ == "__main__": 
#     Compiler().compile( 
#         pipeline_func=full_pipeline, 
#         package_path="full_pipeline.yaml" 
#     )

# model_path="/outputs/model.pkl",
# feature_store_path="/outputs/features.pkl"