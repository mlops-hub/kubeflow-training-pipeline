from kfp import dsl
from kfp.dsl import Input, Output, Dataset

@dsl.component(
    base_image="sandy345/kubeflow-employee-attrition:latest",
)
def tuning_component(
    train_data: Input[Dataset],
    test_data: Input[Dataset],
    tuning_metadata: Output[Dataset],
):
    import os
    import pandas as pd
    import json
    from src.model_pipeline._07_tuning import tuning_data

    train_path = os.path.join(train_data.path, "train.csv")
    test_path = os.path.join(test_data.path, "test.csv")

    overall_parameters = tuning_data(train_path, test_path)

    os.makedirs(tuning_metadata.path, exist_ok=True)
    tuning_file = os.path.join(tuning_metadata.path, "tuning_metadata.json")
    with open(tuning_file, 'w') as f:
        json.dump(overall_parameters, f)

    print("Tuning completed successfully.")

