from kfp.dsl import component, Input, Daatset

# component-1
@component(
    base_image="sandy345/final-kubeflow-pipeline:v1.0.0",
    packages_to_install=['sqlalchemy', 'psycopg2-binary', 'python-dotenv']
)
def get_live_data():
    from sqlalchemy import create_engine, text
    import os
    from dotenv import load_dotenv

    load_dotenv()

    POSTGRES_URI = os.environ.get(
        "POSTGRES_URI_EXTERNAL",
        "postgresql+psycopg2://feast:feast@68.183.87.245:30032/feast"
    )
    engine = create_engine(POSTGRES_URI)

    with engine.connect() as conn:
        result = conn.execute(text("SELECT * FROM live_data ORDER BY id DESC LIMIT 5"))
        print(row[3] for row in result)
        return result.fetchall()  # Return data for downstream components



@component(
    base_image="sandy345/final-kubeflow-pipeline:v1.0.0",
)
def get_reference_data(
    reference_data: Input[Dataset],
):
    import pandas as pd

    dataset_path = reference_data.path + "/train.csv"

    df = pd.read_csv(dataset_path)
    print(df.head())
    return df


# component-3
@component(
    base_image="sandy345/final-kubeflow-pipeline:v1.0.0"
)
def run_monitor_comp():
    """Run the Evidently Monitor pipeline that generates reports and alerts."""
    from src.monitoring.evidently_monitor.index import MonitorPipeline

    pipeline = MonitorPipeline()
    pipeline.run_daily()
