
from kfp.dsl import component, Input, Output, Dataset

@component(
    base_image="sandy345/final-kubeflow-pipeline:v1.0.0"
)
def run_monitoring():
    from src.monitoring.evidently_monitor.data_loader import DataLoader
    from src.monitoring.evidently_monitor.monitor_core import MonitorCore
    from src.monitoring.evidently_monitor import index

    