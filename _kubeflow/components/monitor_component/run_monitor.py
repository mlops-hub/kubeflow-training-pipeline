from kfp.dsl import component
from _kubeflow.config import BASE_IMAGE
from typing import NamedTuple

@component(
    base_image=BASE_IMAGE,
)
def run_monitoring(
    postgres_uri: str,
    evidently_token: str,
    evidently_url: str,
    evidently_org_id: str,
    evidently_project_id: str,
    retrain_threshold: float,
) -> NamedTuple("MonitoringOutput", [
    ("drift_score", float),
    ("prediction_drift_score", float),
    ("should_retrain", bool),
]):
    import pandas as pd
    from sqlalchemy import create_engine
    import json
    from collections import namedtuple

    from evidently import Report, DataDefinition, Dataset, BinaryClassification
    from evidently.presets import DataSummaryPreset, DataDriftPreset
    from evidently.ui.workspace import CloudWorkspace


    # -----------------------------
    # Setup
    # -----------------------------
    engine = create_engine(postgres_uri)

    ws = CloudWorkspace(
        token=evidently_token,
        url=evidently_url
    )

    # -----------------------------
    # Get or create project
    # -----------------------------
    if evidently_project_id:
        project_id = evidently_project_id
        print(f"Using existing project: {project_id}")
    else:
        print("Creating new Evidently project...")
        project = ws.create_project("Employee Attrition", org_id=evidently_org_id)
        project.description = "Attrition monitoring"
        project.save()
        project_id = project.id

    # -----------------------------
    # Load data
    # -----------------------------
    ref_df = pd.read_sql("SELECT * FROM reference_data", engine)

    live_df = pd.read_sql("SELECT * FROM live_data", engine)
    # add condition later:
    # SELECT * FROM live_data WHERE event_timestamp >= NOW() - INTERVAL '1 hour'

    if live_df.empty:
        print("Warning: No live data in last hour.")
        return

    # stable reference sample
    ref_sample = ref_df.sample(n=35, random_state=42)

    ref_sample["prediction"] = ref_sample["target"]

    # -----------------------------
    # Data Definition
    # -----------------------------
    data_definition = DataDefinition(
        classification=[BinaryClassification(
            target="target",
            prediction_labels="prediction",
        )],
        numerical_columns=[
            "years_at_company", "company_tenure", "tenure_gap",
            "role_stagnation_ratio", "number_of_dependents",
            "number_of_promotions", "prediction",
        ],
        categorical_columns=[
            "education_level", "job_level", "company_size",
            "performance_rating", "age_group", "annual_income",
            "overall_satisfaction", "opportunities", "company_reputation",
        ],
        id_column="employee_id",
        timestamp="event_timestamp"
    )

    ref_ds = Dataset.from_pandas(ref_sample, data_definition=data_definition)
    live_ds = Dataset.from_pandas(live_df, data_definition=data_definition)

    # -----------------------------
    # Combined Report (BEST PRACTICE)
    # -----------------------------
    report = Report([
        DataSummaryPreset(),
        DataDriftPreset(
            cat_method='chisquare', # p-value based, works with small samples
            num_method='ks', # p-value based, works with samll samples
            threshold=0.05
        ),
    ], include_tests=False)
    # p-values between 0 and 1,
    # below 0.05 indicate drift

    print("Running monitoring report...")
    eval = report.run(
        reference_data=ref_ds,
        current_data=live_ds
    )
    # print('eval: ', eval.dict())

    # -----------------------------
    # Log run (with tags)
    # -----------------------------
    ws.add_run(
        project_id,
        eval
    )

    # -----------------------------
    # Extract drift score
    # -----------------------------
    def extract_drift_score(eval):
        """
        Look for DriftedColumnsCount metric and return the share (0.0 - 1.0).
        e.g. 6 out of 16 columns drifted -> share = 0.375
        """
        try:
            for metric in eval.dict()["metrics"]:
                if metric.get("metric_name", "").startswith("DriftedColumnsCount"):
                    value = metric.get("value", {})
                    if isinstance(value, dict):
                        return float(value.get('share', 0.0))
            return 0.0
        except Exception as e:
            print(f"Error extracting drift score: {e}")
            return 0.0

    # -----------------------------
    # Extract Value drift score - Model drift
    # -----------------------------
    def extract_prediction_drift(eval):
        """
        Looks for ValueDrift(column=prediction, ...) metric and returns the drift score.
        In Evidently v2, this value is a p-value (0.0 - 1.0).
        Low p-value (< threshold, e.g. 0.05) means drift IS detected.
        e.g. value = 1.0 means NO drift (high p-value = distributions are similar)
        """
        try:
            for metric in eval.dict()["metrics"]:
                name = metric.get("metric_name", "")
                if name.startswith("ValueDrift") and "column=prediction" in name:
                    return float(metric.get("value", 0.0))
            return 0.0
        except Exception as e:
            print(f"Error extracting prediction drift: {e}")
            return 0.0


    drift_value = extract_drift_score(eval)
    prediction_drift_value = extract_prediction_drift(eval)
    is_retrain = drift_value > retrain_threshold

    print(f"Drift Score: {drift_value}")
    print(f"Prediction Drift: {prediction_drift_value}")
    print(f"Should Retrain pipeline?: {is_retrain}")
    
    MonitoringOutput = namedtuple("MoniotringOutput", [
        "drift_score", "prediction_drift_score", "should_retrain"
    ])

    print("Monitoring completed successfully.")

    return MonitoringOutput(
        drift_value,
        prediction_drift_value,
        is_retrain
    )
    
