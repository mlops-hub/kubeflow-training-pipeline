from kfp import dsl
from kfp.dsl import component, Input, Output, Dataset

@component(
    base_image="sandy345/kubeflow-employee-attrition:latest", 
)
def validation_component(
    input_data: Input[Dataset], 
    output_data: Output[Dataset]
):
    import os
    import pandas as pd
    from src.data_pipeline._02_validation import validate_data

    input_path = os.path.join(input_data.path, "ingestion.csv")
    df = pd.read_csv(input_path)

    validated_df = validate_data(df)

    os.makedirs(output_data.path, exist_ok=True)
    output_path = os.path.join(output_data.path, "validated.csv")
    validated_df.to_csv(output_path, index=False)

    print(f"Validation completed. Saved to {output_path}")

