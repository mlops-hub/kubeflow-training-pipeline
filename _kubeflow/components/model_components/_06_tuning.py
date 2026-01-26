from kfp import dsl
from kfp.dsl import Input, Output, Dataset, Model

@dsl.component(
    base_image="sandy345/kubeflow-employee-attrition:latest",
)
def tuning_component(
    train_data: Input[Dataset],
    test_data: Input[Dataset],
    tuned_model: Output[Model],
    tuning_metadata: Output[Dataset],
):
    import os
    import pandas as pd
    import joblib
    import json
    from sklearn.linear_model import LogisticRegression
    from model_pipeline._07_tuning import tuning_data

    train_path = os.path.join(train_data.path, "train.csv")
    train_df = pd.read_csv(train_path)

    test_path = os.path.join(test_data.path, "test.csv")
    test_df = pd.read_csv(test_path)

    base_model = LogisticRegression(max_iter=1000, class_weight='balanced')

    best_model, overall_parameters = tuning_data(train_df, test_df, base_model)

    os.makedirs(tuned_model.path, exist_ok=True)
    tuned_model_file = os.path.join(tuned_model.path, "best_model.pkl")
    joblib.dump(best_model, tuned_model_file)

    os.makedirs(tuning_metadata.path, exist_ok=True)
    tuning_file = os.path.join(tuning_metadata.path, "tuning_metadata.json")
    with open(tuning_file, 'w') as f:
        json.dump(overall_parameters, f)

    print("Tuning completed successfully.")

