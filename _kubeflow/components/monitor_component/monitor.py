from kfp.dsl import component, Input, Dataset, Output
from _kubeflow.config import BASE_IMAGE

@component(
    base_image=BASE_IMAGE
)
def run_monitor_comp(
    output_path: Output[Dataset]
):
    import os
    from src.monitoring.evidently_monitor.index import MonitorPipeline

    POSTGRES_URI = os.environ.get(
        "POSTGRES_URI_EXTERNAL",
        "postgresql+psycopg://feast:feast@68.183.87.245:30032/feast"
    )

    pipeline = MonitorPipeline(
        ev_token=os.environ.get("EVIDENTLY_TOKEN", "sk_prod.019d2123-689e-7697-b95b-bfd67a04d2d6.zUKgDkB4GM4XuWyzRkJmHTNLcb39GvCfUGA2-n7tVuC0BxuhIqkmFNwz7uiafTiBUwMGneosivM-iXG5bSmZZoLK_TIU45qQMk9R9cI9f2PYBhwaszePtlJTk785R3o_"),
        ev_url=os.environ.get("EVIDENTLY_URL", "https://app.evidently.cloud"),
        org_id=os.environ.get("EVIDENLTY_ORG_ID", "019d2071-06a2-79b9-b627-2c23f3cdb8c5"),
        project_id=os.environ.get("EVIDENTLY_PROJECT_ID", "019d212d-30ad-7f4b-bb92-37b7049ab60e"),
        postgres_uri=POSTGRES_URI,
        output_path=output_path.path,
    )
    pipeline.run_daily()
