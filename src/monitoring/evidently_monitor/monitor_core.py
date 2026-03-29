from evidently import Report
from evidently.presets import DataSummaryPreset, DataDriftPreset
from evidently.metrics import ValueDrift

class MonitorCore:
    def __init__(self, ws, project_id, loader):
        self.ws = ws
        self.project_id = project_id
        self.loader = loader
        # thresholds
        self.drift_threshold = 0.1
        self.retrain_threshold = 0.65
        self.prediction_drift_threshold = 0.2
        
        
    def generate_reports(self):
        ref_data = self.loader.load_reference_data()
        ref_sample_data = self.loader.load_sample_reference_data()
        live_data = self.loader.load_live_data()
        # live_data = self.loader.load_recent_live_data(hours=1)

        if live_data.empty:
            print("No live data available for monitoring.")
            return
        
        ref_ds, ref_sample_ds, live_ds = self.loader.to_datasets(
            ref_data, 
            ref_sample_data, 
            live_data
        )

        # generate reports
        report = Report([
            DataSummaryPreset(),
            DataDriftPreset(
                cat_method='psi',
                num_method='wasserstein',
                threshold=self.drift_threshold
            ),
            ValueDrift(column_name="prediction")
        ], include_tests=True)

        print("📊 Generating combined monitoring report...")

        eval = report.run(
            reference_data=ref_sample_ds,
            current_data=live_ds
        )

        # ✅ log ONE run instead of many
        self.ws.add_run(
            self.project_id,
            eval,
            tags={"window": "1h"}
        )
        # data_quality = Report([DataSummaryPreset()], include_tests=True)
        # data_drift = Report([DataDriftPreset(
        #     cat_method='psi',
        #     num_method='wasserstein',
        #     threshold=0.1
        # )], include_tests=True)
        # value_drift = Report([ValueDrift(column_name='prediction',)], include_tests=True)

        # reports = {
        #     "data_quality": data_quality,
        #     "data_drift": data_drift,
        #     "value_drift": value_drift
        # }

        # results = {}
        # from datetime import datetime
        # timestamp = datetime.utcnow().isoformat()

        # for name, report in reports.items():
        #     print(f"Generating {name} report...")
        #     eval = report.run(
        #         reference_data=ref_sample_ds,
        #         current_data=live_ds
        #     )
        #     self.ws.add_run(
        #         self.project_id, 
        #         eval,
        #         tags={
        #             "window": "last_1_hour",
        #             "run_time": timestamp
        #         }
        #     )
        #     results[name] = eval

        # check drift thresholds
        drift_score = self._extract_drift_score(eval)
        pred_drift = self._extract_prediction_drift(eval)
        print(f"📊 Dataset Drift: {drift_score}")
        print(f"📉 Prediction Drift: {pred_drift}")

        self._decision_logic(drift_score, pred_drift)

        return eval


    def _extract_dataset_drift(self, eval):
        try:
            for metric in eval.dict()["metrics"]:
                result = metric.get("result", {})

                if "share_of_drifted_columns" in result:
                    return result["share_of_drifted_columns"]

            return 0.0
        except Exception as e:
            print(f"Error extracting dataset drift: {e}")
            return 0.0


    def _extract_prediction_drift(self, eval):
        try:
            for metric in eval.dict()["metrics"]:
                if metric.get("metric") == "ValueDrift":
                    return metric["result"]["drift_score"]

            return 0.0
        except Exception as e:
            print(f"Error extracting prediction drift: {e}")
            return 0.0
        

    def _decision_logic(self, drift_score, pred_drift):
        # 🚨 Severe case → retrain
        if drift_score > self.retrain_threshold or pred_drift > 0.4:
            print("🚨 Severe drift detected → Retraining")
            self._send_alert(
                f"🚨 Severe drift: dataset={drift_score}, prediction={pred_drift}"
            )
            self._trigger_retraining()

        # ⚠️ Moderate case → alert only
        elif drift_score > self.drift_threshold or pred_drift > self.prediction_drift_threshold:
            print("⚠️ Moderate drift detected")
            self._send_alert(
                f"⚠️ Drift detected: dataset={drift_score}, prediction={pred_drift}"
            )

        else:
            print("✅ System stable")