from kfp import dsl
from kfp.dsl import Input, Output, Dataset, Model

@dsl.component(
    base_image="python:3.10-slim",
    packages_to_install=[
        "pandas",
        "scikit-learn",
        "joblib",
        "git+https://github.com/mlops-hub/kubeflow-training-pipeline.git"
    ],
)
def tuning_component(
    train_df: Input[Dataset],
    test_df: Input[Dataset],
    base_model: Input[Model],
    tuned_model: Output[Model],
    tuning_metadata: Output[Dataset],
):
    import pandas as pd
    import joblib
    from src.model_pipeline._10_tuning import tuning_data

    df_train = pd.read_parquet(train_df.path)
    df_test = pd.read_parquet(test_df.path)
    model = joblib.load(base_model.path)

    best_model = tuning_data(df_train, df_test, model)

    joblib.dump(best_model, tuned_model.path)
    metrics = joblib.load("artifacts/model_v1/metrics.pkl")

    pd.DataFrame([metrics]).to_csv(tuning_metadata.path)
