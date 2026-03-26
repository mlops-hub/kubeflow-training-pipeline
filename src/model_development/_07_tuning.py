import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, recall_score
from _mlflow.registry import MLflowRegistry
import os
from dotenv import load_dotenv
from feast import FeatureStore

load_dotenv()


def tuning_data(feast_repo_path: str, train_path: str, test_path: str, preprocess_path: str, tracking_uri: str, experiment_name: str) -> dict:
    # ------- feast setup ------------
    # 1. Initialize Feast
    # repo_path should point to where your feature_store.yaml is
    store = FeatureStore(repo_path=feast_repo_path)
        
    entity_df_train = pd.read_csv(train_path)
    entity_df_test = pd.read_csv(test_path)

    entity_cols = ['employee_id', 'event_timestamp']
    entity_df_train = entity_df_train[entity_cols].copy()
    entity_df_test = entity_df_test[entity_cols].copy()


    # 2. Ensure timestamps are actual datetime objects
    entity_df_train['event_timestamp'] = pd.to_datetime(entity_df_train['event_timestamp'])
    entity_df_test['event_timestamp'] = pd.to_datetime(entity_df_test['event_timestamp'])

    # 3. Pull Features from Feast
    print("Fetching training features from Feast...")
    df_train = store.get_historical_features(
        entity_df=entity_df_train,
        features=store.get_feature_service("employee_attrition_features")
    ).to_df()
    # to_arrow().to_pandas(types_mapper=pd.ArrowDtype)

    print("Fetching testing features from Feast...")
    df_test = store.get_historical_features(
        entity_df=entity_df_test,
        features=store.get_feature_service("employee_attrition_features")
    ).to_df()

    print(df_train.head(2))
    print(df_train.isnull().sum())
    print(df_test.isnull().sum())
    # ------- end feast ------------------

    # Prepare X and y
    # Note: Feast returns entity columns + features. 
    # Drop the keys that the model shouldn't see.
    cols_to_drop = ['attrition', 'employee_id', 'event_timestamp']

    X_train = df_train.drop(columns=cols_to_drop)
    y_train = df_train['attrition']

    X_test = df_test.drop(columns=cols_to_drop)
    y_test = df_test['attrition']


    # load preprocessor
    preprocessor = joblib.load(preprocess_path)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(max_iter=1000, class_weight='balanced'))
    ])

    # set parameters
    param_grid = {
        'model__C': [0.01, 0.1, 1, 10, 100],
        'model__solver': ['liblinear', 'saga'],
        'model__l1_ratio': [0],     # equivalent to L2
        'model__max_iter': [1000]
    }

    # set cv
    strat_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    # use gridserachcv for tuning model
    grid = GridSearchCV(
        estimator=pipeline, 
        param_grid=param_grid, 
        cv=strat_cv, 
        scoring='recall',
        n_jobs=-1
    )
    grid.fit(X_train, y_train)

    # get best model
    tuned_model = grid.best_estimator_

    best_parameters = grid.best_params_
        
    # predict the output with tuned_model
    y_pred = tuned_model.predict(X_test)

    # tuned model evaluation
    metrics = {
        "tuned_accuracy": accuracy_score(y_test, y_pred),
        "tuned_recall": recall_score(y_test, y_pred),
        "tuned_train_score": tuned_model.score(X_train, y_train),
        "tuned_test_score": tuned_model.score(X_test, y_test)
    }        

    # log in mlflow
    registry = MLflowRegistry(
        tracking_uri=tracking_uri,
        experiment_name=experiment_name
    )

    with registry.start_run(run_name="model-training-run"):
        mlflow_run_id = registry.get_run_id()
        print('run_id: ', mlflow_run_id)

        overall_parameters = {
            **best_parameters,
            **metrics,
        }

        registry.log_params_mlflow(params=overall_parameters, stage="tuning")
        
        print("Logged tuning values in MLflow!")
        

    return mlflow_run_id, overall_parameters



if __name__ == "__main__":
    from pathlib import Path
    import json

    BASE_DIR = Path(__file__).resolve().parents[2]
    DATASET_PATH = BASE_DIR / "datasets" / "data-pipeline"
    TRAIN_PATH =  DATASET_PATH / "06_preprocess_train_df.csv"
    TEST_PATH = DATASET_PATH / "06_preprocess_test_df.csv"

    ARTIFACTS_PATH = BASE_DIR / "artifacts"
    PREPROCESSOR_PATH = ARTIFACTS_PATH / "preprocessor.pkl"
    BEST_PARAMETERS = ARTIFACTS_PATH / "best_parameters.json"
    MLFLOW_RUN_ID = ARTIFACTS_PATH / "mlflow_run_id.txt"  

    FEAST_DATA_DIR = BASE_DIR / "_feast" / "feature_repo"

    run_id, overall_parameters = tuning_data(
        feast_repo_path=FEAST_DATA_DIR,
        train_path=TRAIN_PATH, 
        test_path=TEST_PATH, 
        preprocess_path=PREPROCESSOR_PATH,
        tracking_uri = os.environ["MLFLOW_TRACKING_INTERNAL_URI"],
        experiment_name = os.environ["MLFLOW_EXPERIMENT_NAME"],
    )

    with open(BEST_PARAMETERS, 'w') as f:
        json.dump(overall_parameters, f)

    with open(MLFLOW_RUN_ID, "w") as f:
        f.write(run_id)
