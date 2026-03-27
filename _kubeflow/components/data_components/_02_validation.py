
from kfp.dsl import component, Input, Output, Dataset
from _kubeflow.config import BASE_IMAGE

@component(
    base_image=BASE_IMAGE
)
def validation_component(
    input_data: Input[Dataset], 
    output_data: Output[Dataset]
):
    import os
    from src.data_preparation._02_validation import validate_data

    input_path = os.path.join(input_data.path, "ingestion.csv")

    validated_df = validate_data(input_path)

    os.makedirs(output_data.path, exist_ok=True)
    output_path = os.path.join(output_data.path, "validation.csv")
    validated_df.to_csv(output_path, index=False)

    print(f"Validation completed. Saved to {output_path}")

