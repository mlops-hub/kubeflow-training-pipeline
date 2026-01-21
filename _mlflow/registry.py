from pathlib import Path
import mlflow
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient


BASE_DIR = Path(__file__).resolve().parents[2]
PREPROCESSED_TRAIN_PATH = BASE_DIR / "datasets" / "data-engg" / "06_preprocess_train_df.csv"

BEST_MODEL_ARTIFACT = BASE_DIR / "artifacts" / "model_v1" / "best_model.pkl"
MLFLOW_MODEL_INFO = BASE_DIR / "artifacts" / "model_v1" / "model_info.pkl"
THRESHOLD = 0.70


class MLflowRegistry:
    def __init__(self, tracking_uri: str, experiment_name: str):
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name

        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        self.client = MlflowClient()

    # start run
    def start_run(self, run_name: str = None):
        return mlflow.start_run(run_name=run_name)


    def log_model(self, model, X_train, parameters: dict, metrics: dict[str, float], artifact_name: str):
        signature = infer_signature(X_train, model.predict(X_train))

        mlflow.log_params(parameters)
        mlflow.log_metrics(metrics)

        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            name=artifact_name,
            signature=signature,
            input_example=X_train.iloc[:5],
        )

        return {
            "run_id": mlflow.active_run().info.run_id,
            "model_uri": model_info.model_uri,
            "recall": metrics['recall_ht'],
            "artifact_name": artifact_name
        }


    def register_model(self, metadata, registry_name):
        model_uri = f"runs:/{metadata['run_id']}/{metadata['artifact_name']}"
        print('model-uri: ', model_uri)

        register_model = mlflow.register_model(
            model_uri=model_uri,
            name=registry_name
        )
        version = register_model.version

        # set tags
        self.client.set_model_version_tag(
            name=register_model.name,
            version=version,
            key="source_run_id",
            value=metadata['run_id']
        )

        self.client.set_model_version_tag(
            name=register_model.name,
            version=version,
            key="production_ready",
            value="pending_approval",
        )

        # Transition to stage
        self.client.transition_model_version_stage(
            name=register_model.name,
            version=version,
            stage='Staging',
        )

        print("\nModel registered successfully!")
        print(f"\nVersion: {version}")
        print(f"\nStage: {register_model.current_stage}")

        return register_model


    def promote_model(self, model_name: str, version: int, metric_value: float, threshold: float):
        stage = "Production" if metric_value >= threshold else "Staging"
        self.client.transition_model_version_stage( 
            name=model_name, 
            version=version, 
            stage=stage, 
        ) 
        self.client.set_model_version_tag( 
            name=model_name, 
            version=version, 
            key="promotion_decision", 
            value=f"{stage} (metric={metric_value}, threshold={threshold})", 
        ) 
        return stage
