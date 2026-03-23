from kfp import dsl
from kfp.dsl import component, Input, Output, Dataset

@component(
    base_image="sandy345/final-kubeflow-pipeline:v1.0.0"
)
def feature_engg_component(
    input_data: Input[Dataset], 
    output_data: Output[Dataset]
):
    import os
    from src.data_preparation._05_feature_engg import feature_data

    input_path = os.path.join(input_data.path, "cleaned.csv")
    
    feature_df = feature_data(input_path)

    os.makedirs(output_data.path, exist_ok=True)
    output_path = os.path.join(output_data.path, "feature_engg.csv")
    feature_df.to_csv(output_path, index=False)

    print(f"Feature engg is completed. Saved to {output_path}")
