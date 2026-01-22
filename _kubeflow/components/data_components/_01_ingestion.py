from kfp import dsl
from kfp.dsl import component, Output, Dataset

BASE_IMAGE = "python:3.10-slim"

@component(
    base_image=BASE_IMAGE, 
    packages_to_install=['pandas',  "git+https://github.com/mlops-hub/kubeflow-training-pipeline.git"],
)
def ingestion_component(
    output_data: Output[Dataset]
):
    import pandas as pd
    from src.data_pipeline._01_ingestion import ingestion

    df = ingestion()
    output_path = output_data.path + "/ingestion.csv"
    df.to_csv(output_path, index=False)

    print(f"Ingestion completed. Saved to {output_path}")

