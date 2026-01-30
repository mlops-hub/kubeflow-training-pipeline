from kfp import dsl
from kfp.compiler import Compiler

# data
from _kubeflow.components.data_components._01_ingestion import ingestion_component
from _kubeflow.components.data_components._02_validation import validation_component
from _kubeflow.components.data_components._03_cleaning import cleaned_component
from _kubeflow.components.data_components._04_feature_engg import feature_engg_component
from _kubeflow.components.data_components._05_preprocessing import preprocessed_component

# model
from _kubeflow.components.model_components._07_training import trainer_model_component
from _kubeflow.components.model_components._08_evaluation import evaluation_component
from _kubeflow.components.model_components._06_tuning import tuning_component
from _kubeflow.components.model_components._09_register import register_model_component


@dsl.pipeline( 
    name="Employee Attrition Full Pipeline", 
    description="Data -> Training -> Tuning -> Evaluation -> MLflow Registry"
)
def full_pipeline(
    namespace: str = "kubeflow",
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
    tuning = tuning_component(
        train_data=preprocess.outputs['train_data'],
        test_data=preprocess.outputs['test_data'],
    )
    # tune outputs: tuning_metadata


    # trainer job - kubeflow trainer
    trainer_model_component(
        job_name="attrition-trainer-job",
        namespace=namespace,
        image=trainer_image,
        cpu=cpu,
        memory=memory,
        train_path=preprocess.outputs['train_data'],
        scaler_path=preprocess.outputs['scaler_model'],
        tuning_metadata=tuning.outputs['tuning_metadata']
    )

    evaluation_component(
        test_data=preprocess.outputs['test_data'],
    )

    register_model_component(
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