from kfp import dsl

# util
from _kubeflow.components.monitor_component.monitor import run_monitor_comp


@dsl.pipeline( 
    name="Employee Attrition Monitoring Pipeline", 
    description="Monitoring the model performance and drift using Evidently"
)
def monitor_pipeline():

    run_monitor_comp().set_caching_options(False)

# Compile pipeline 
# if __name__ == "__main__": 
#     Compiler().compile( 
#         pipeline_func=monitor_pipeline, 
#         package_path="monitor_pipeline.yaml" 
#     )
