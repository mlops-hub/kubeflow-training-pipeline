from datetime import datetime
import os
import time
from dotenv import load_dotenv
from .config import *
from .data_loader import DataLoader
from .monitor_core import MonitorCore
from .alerting import AlertEngine
# from .prometheus_metric import start_prometheus_server
from evidently.ui.workspace import CloudWorkspace


load_dotenv()

class MonitorPipeline:
    def __init__(self):
        self.ws = CloudWorkspace(
            token=EVIDENTLY_TOKEN,
            url=EVIDENTLY_URL
        )
        self.project_id = self._get_project_id()
        self.loader = DataLoader(REFERENCE_DB_PATH, LIVE_DB_PATH)
        self.alert_engine = AlertEngine(OUTPUT_DB_PATH)
        os.makedirs(OUTPUT_DB_PATH, exist_ok=True)


    def _get_project_id(self):
        if PROJECT_ID:
            print(f"Using existing project: {PROJECT_ID}")
            return PROJECT_ID
        print("Creating new Evidently Cloud project.... ")
        project = self.ws.create_project("Employee Attrition", org_id=ORG_ID)
        project.description = "Predict whether employee saty or leave"
        project.save()
        return project.id
    

    def run_daily(self):
        # start_prometheus_server(PROMETHEUS_PORT)
        date_str = datetime.now().strftime("%Y%m%d")

        print("🚀 Starting Animal Classification Monitoring...")
        monitor = MonitorCore(self.ws, self.project_id, self.loader)
        reports = monitor.generate_reports()
        if not reports:
            print("No reports generated.")
            return
        self.alert_engine.process_reports(reports, date_str)
        print("✅ Monitoring completed successfully!")
        return


if __name__ == '__main__':
    pipeline = MonitorPipeline()
    pipeline.run_daily()
    while True:
        time.sleep(60)

