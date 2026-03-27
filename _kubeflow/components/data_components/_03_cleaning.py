
from kfp.dsl import component, Input, Output, Dataset
from _kubeflow.config import BASE_IMAGE

@component(
    base_image=BASE_IMAGE
)
def cleaned_component(
    input_data: Input[Dataset], 
    output_data: Output[Dataset]
):
    import os
    from src.data_preparation._04_cleaning import clean_data

    input_path = os.path.join(input_data.path, "validation.csv")

    clean_df = clean_data(input_path)

    os.makedirs(output_data.path, exist_ok=True)
    output_path = os.path.join(output_data.path, "cleaned.csv")
    clean_df.to_csv(output_path, index=False)

    print(f"Cleaning completed. Saved to {output_path}")
