import argparse
import json
import os
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from _mlflow.registry import MLflowRegistry
from dotenv import load_dotenv

load_dotenv()


def training_data(train_path: str, preprocessor_path: str, best_params_path: str, tracking_uri: str, experiment_name: str, artifact_name: str):
    registry = MLflowRegistry(
        tracking_uri=tracking_uri,
        experiment_name=experiment_name
    )

    # load data
    df = pd.read_csv(train_path)
    X_train = df.drop(columns=['Attrition'])
    y_train = df['Attrition']

    # load parameters
    with open(best_params_path, 'r') as f:
        params = json.load(f)

    best_params = {
        key.replace("model__", ""): value
        for key, value in params.items()
        if key.startswith("model__")
    }

    # load scaler
    preprocessor = joblib.load(preprocessor_path)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(**best_params))
    ])

    with registry.start_run(run_name='model-training-run'):
        print("Training the model....")
        pipeline.fit(X_train, y_train)

        print(pipeline)
        print("training completed...")

        registry.log_model(
            model=pipeline,
            X_train=X_train,
            parameters=best_params,
            artifact_name=artifact_name,
        )

    print(f"✓ Training complete")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", required=True)
    parser.add_argument("--preprocessor_path", required=True)
    parser.add_argument("--best_params_path", required=True)
    args = parser.parse_args()
    

    training_data(
        train_path=args.train_path,
        preprocessor_path=args.preprocessor_path,
        best_params_path=args.best_params_path,
        tracking_uri=os.environ["MLFLOW_TRACKING_URI"], 
        experiment_name=os.environ["MLFLOW_EXPERIMENT_NAME"], 
        artifact_name="employee-attrition-model"
    )



# run command
# python -m src.model_pipeline._08_training --train_path datasets/data-pipeline/06_preprocess_train_df.csv --preprocessor_path artifacts/model_v1/preprocessor.pkl --best_params_path artifacts/model_v1/tuning_metadata.json

# model_path=args.model_path, 
# feature_store_path=args.feature_store_path,