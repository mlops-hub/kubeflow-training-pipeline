from kfp import dsl
from kfp.dsl import component, Input, Output, Dataset

BASE_IMAGE = "python:3.10-slim"

@component(
    base_image=BASE_IMAGE, 
    packages_to_install=['pandas', "git+https://github.com/mlops-hub/kubeflow-training-pipeline.git"],
)
def cleaned_component(
    input_data: Input[Dataset], 
    output_data: Output[Dataset]
):
    import pandas as pd
    from src.data_pipeline._04_cleaning import clean_data

    input_path = input_data.path + "/validated.csv"
    df = pd.read_csv(input_path)

    clean_df = clean_data(df)

    output_path = output_data.path + "/cleaned_data.csv"
    clean_df.to_csv(output_path, index=False)

    print(f"Cleaning completed. Saved to {output_path}")


