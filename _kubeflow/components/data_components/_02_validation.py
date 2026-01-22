from kfp import dsl
from kfp.dsl import component, Input, Output, Dataset

BASE_IMAGE = "python:3.10-slim"

@component(
    base_image=BASE_IMAGE, 
    packages_to_install=['pandas', "git+https://github.com/mlops-hub/kubeflow-training-pipeline.git"],
)
def validation_component(
    input_data: Input[Dataset], 
    output_data: Output[Dataset]
):
    import pandas as pd
    from src.data_pipeline._02_validation import validate_data

    input_path = input_data.path + "/ingestion.csv"
    df = pd.read_csv(input_path)

    validated_df = validate_data(df)

    output_path = output_data.path + "/validated.csv"
    validated_df.to_csv(output_path, index=False)

    print(f"Validation completed. Saved to {output_path}")

