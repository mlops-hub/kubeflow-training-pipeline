from kfp import dsl
from kfp.dsl import component, Input, Output, Dataset

BASE_IMAGE = "python:3.11-slim"

@component(
    base_image=BASE_IMAGE, 
    packages_to_install=['pandas', '.'],
    source="./"
)
def feature_engg_component(
    input_data: Input[Dataset], 
    output_data: Output[Dataset]
):
    import pandas as pd
    from pathlib import Path
    from src.data_pipeline._05_feature_engg import feature_data

    input_path = Path(input_data.path) + "/cleaned_data.csv"
    df = pd.read_csv(input_path)

    feature_df = feature_data(df)

    output_path = Path(output_data.path) + "/transformed_data.csv"
    feature_df.to_csv(output_path, index=False)

    print(f"Feature engg is completed. Saved to {output_path}")


