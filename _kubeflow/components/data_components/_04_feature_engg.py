from kfp import dsl
from kfp.dsl import component, Input, Output, Dataset

@component(
    base_image="sandy345/kubeflow-employee-attrition:latest", 
)
def feature_engg_component(
    input_data: Input[Dataset], 
    output_data: Output[Dataset]
):
    import os
    import pandas as pd
    from src.data_pipeline._05_feature_engg import feature_data

    input_path = os.path.join(input_data.path, "cleaned_data.csv")
    df = pd.read_csv(input_path)

    feature_df = feature_data(df)

    os.makedirs(output_data.path, exist_ok=True)
    output_path = os.path.join(output_data.path, "transformed_data.csv")
    feature_df.to_csv(output_path, index=False)

    print(f"Feature engg is completed. Saved to {output_path}")


