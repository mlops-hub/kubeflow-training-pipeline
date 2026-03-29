import os
from evidently.ui.workspace.cloud import CloudWorkspace
from monitoring.data_loader import DataLoader


def setup_datasets():
    ws = CloudWorkspace(
        token=os.environ["EVIDENTLY_TOKEN"],
        url=os.environ["EVIDENTLY_URL"]
    )

    org_id = os.environ["EVIDENTLY_ORG_ID"]
    postgres_uri = os.environ["POSTGRES_URI_EXTERNAL"]

    loader = DataLoader(postgres_uri)

    # Create project
    project = ws.create_project("Employee Attrition", org_id=org_id)
    project.description = "Monitoring datasets"
    project.save()

    print(f"✅ Project created: {project.id}")

    # Load data
    ref_df = loader.load_reference_data()
    live_df = loader.load_live_data()

    ref_ds, live_ds = loader.to_datasets(ref_df, live_df)

    # Register datasets ONCE
    ws.add_dataset(
        dataset=ref_ds,
        name="reference_dataset",
        project_id=project.id
    )

    ws.add_dataset(
        dataset=live_ds,
        name="live_dataset",
        project_id=project.id
    )

    print("✅ Datasets registered successfully")


if __name__ == "__main__":
    setup_datasets()