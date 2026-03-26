from evidently import Report
from evidently.presets import DataSummaryPreset, DataDriftPreset
from evidently.metrics import ValueDrift

class MonitorCore:
    def __init__(self, ws, project_id, loader):
        self.ws = ws
        self.project_id = project_id
        self.loader = loader

    def generate_reports(self):
        ref_data = self.loader.load_reference_data()
        ref_sample_data = self.loader.load_sample_reference_data()
        live_data = self.loader.load_live_data()

        if live_data.empty:
            print("No live data available for monitoring.")
            return
        
        ref_ds, ref_sample_ds, live_ds = self.loader.to_datasets(ref_data, ref_sample_data, live_data)

        # save dataset in evidently
        self.ws.add_dataset(
            dataset=ref_ds,
            name="reference_dataset",
            description="Reference dataset during training",
            project_id=self.project_id,
        )
        self.ws.add_dataset(
            dataset=ref_sample_ds,
            name="sample_reference_dataset",
            description="Sample of reference dataset to test with small live datasets",
            project_id=self.project_id,
        )
        self.ws.add_dataset(
            dataset=live_ds,
            name="live_dataset",
            description="Live dataset during infernece",
            project_id=self.project_id,
        )

        # generate reports
        data_quality = Report([DataSummaryPreset()], include_tests=True)
        data_drift = Report([DataDriftPreset()], include_tests=True)
        data_drift = Report([DataDriftPreset(
            cat_method='psi',
            num_method='wasserstein',
            threshold=0.1
        )], include_tests=True)

        reports = {
            "data_quality": data_quality,
            "data_drift": data_drift,
        }

        results = {}

        for name, report in reports.items():
            print(f"Generating {name} report...")
            eval = report.run(
                reference_data=ref_sample_ds,
                current_data=live_ds
            )
            self.ws.add_run(self.project_id, eval)
            results[name] = eval

        return results
            

