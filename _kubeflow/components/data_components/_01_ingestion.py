from kfp import dsl
from kfp.dsl import component, Output, Dataset


@component(
    base_image="sandy345/kubeflow-employee-attrition:latest"
)
def ingestion_component(
    output_data: Output[Dataset]
):
    import os
    import pandas as pd
    # from src.data_pipeline._01_ingestion import ingestion

    df = ingestion()

    # KFP v2: output_data.path is a DIRECTORY, not a file
    os.makedirs(output_data.path, exist_ok=True)
    output_path = os.path.join(output_data.path, "ingestion.csv")

    df.to_csv(output_path, index=False)

    print(f"Ingestion completed. Saved to {output_path}")

