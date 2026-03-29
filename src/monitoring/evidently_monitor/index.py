from datetime import datetime
# ensure dotenv is available when called below
from dotenv import load_dotenv
from .data_loader import DataLoader
from .monitor_core import MonitorCore
# from .alerting import AlertEngine
# from .prometheus_metric import start_prometheus_server
from evidently.ui.workspace import CloudWorkspace


load_dotenv()

class MonitorPipeline:
    def __init__(self, ev_token, ev_url, output_path, postgres_uri, org_id, project_id=None):
        self.ws = CloudWorkspace(
            token=ev_token,
            url=ev_url
        )
        self.org_id = org_id
        self.project_id = self._get_project_id(project_id)
        self.loader = DataLoader(postgres_uri)
        # self.alert_engine = AlertEngine(output_path)
        # os.makedirs(output_path, exist_ok=True)


    def _get_project_id(self, project_id):
        if project_id:
            print(f"Using existing project: {project_id}")
            return project_id
        print("Creating new Evidently AI project.... ")
        project = self.ws.create_project("Employee Attrition", org_id=self.org_id)
        project.description = "Predict whether employee stay or leave"
        project.save()
        return project.id
    

    def run_daily(self):
        # start_prometheus_server(PROMETHEUS_PORT)
        # date_str = datetime.now().strftime("%Y%m%d")

        print("🚀 Starting Employee Attrition Monitoring...")
        monitor = MonitorCore(self.ws, self.project_id, self.loader)
        reports = monitor.generate_reports()
        if not reports:
            print("No reports generated.")
            return
        # else:
            # self.alert_engine.process_reports(reports, date_str)
        print("✅ Monitoring completed successfully!")
        return


if __name__ == '__main__':
    # initialize constants 
    import os
    from pathlib import Path

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    print(PROJECT_ROOT)
    LIVE_DB_PATH = PROJECT_ROOT / "db" / "live_data.db"
    REFERENCE_DB_PATH = PROJECT_ROOT / "db" / "reference_data.db"
    OUTPUT_DB_PATH =  PROJECT_ROOT / "reports"

    ORG_ID = os.environ.get("EVIDENLTY_ORG_ID", "")
    PROJECT_ID = os.environ.get("EVIDENTLY_PROJECT_ID", "")
    EVIDENTLY_TOKEN = os.environ.get("EVIDENTLY_TOKEN", "")
    EVIDENTLY_URL = os.environ.get("EVIDENTLY_URL", "https://app.evidently.cloud")

    PROMETHEUS_PORT = int(os.environ.get("MONITOR_PROMETHEUS_PORT", 9090))
    POSTGRES_URI = os.environ.get(
        "POSTGRES_URI_EXTERNAL",
        "postgresql+psycopg2://feast:feast@68.183.87.245:30032/feast"
    )

    pipeline = MonitorPipeline(
        ev_token=EVIDENTLY_TOKEN,
        ev_url=EVIDENTLY_URL,
        output_path=str(OUTPUT_DB_PATH),
        postgres_uri=str(POSTGRES_URI),
        org_id=ORG_ID,
        project_id=PROJECT_ID
    )
    pipeline.run_daily()
