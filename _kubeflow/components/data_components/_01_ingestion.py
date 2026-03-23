from kfp.dsl import component, Output, Dataset

@component(
    base_image="sandy345/final-kubeflow-pipeline:v1.0.0"
)
def ingestion_component(
    output_data: Output[Dataset]
):
    import os
    import boto3
    from src.data_preparation._01_ingestion import ingestion


    s3 = boto3.client(
        "s3",
        endpoint_url="http://minio-service.kubeflow:9000",
        aws_access_key_id="minio",
        aws_secret_access_key="minio123",
    )

    tmp_path = "/tmp/employee_attrition.csv"

    s3.download_file("mlpipeline", "datasets/employee_attrition.csv", tmp_path)

    df = ingestion(tmp_path)

    print('component-output: ', df.head(3))

    # KFP v2: output_data.path is a DIRECTORY, not a file
    os.makedirs(output_data.path, exist_ok=True)
    output_path = os.path.join(output_data.path, "ingestion.csv")

    df.to_csv(output_path, index=False)

    print(f"Ingestion completed. Saved to {output_path}")

