import re
from datetime import datetime
from .prometheus_metric import (
    DATA_DRIFT_DETECTED, DRIFTED_COLUMNS, PREDICTION_DRIFT_DETECTED,
    MODEL_ACCURACY_CURRENT, MODEL_ACCURACY_REFERENCE, ALERT_COUNT
)

class AlertEngine:
    def __init__(self, output_path):
        self.output_path = output_path

    def process_reports(self, reports, date_str):
        alerts = []

        # ---- Data drift alerts ----
        dd = reports.get('data_drift')
        drift_detected = False
        drifted_columns_count = 0

        if dd:
            drift_result = dd.dict()
            tests = drift_result.get('tests', [])
            for test in tests:
                metric_config = test.get('metric_config', {}).get('params', {})
                name = metric_config.get("column", "Unnamed Test")
                desc = test.get("description", "") 
                status = str(test.get("status", ""))
                if status == "TestStatus.FAIL":
                    drift_detected = True
                    alerts.append(f"{name} — {desc}")
                
            metrics = drift_result.get('metrics', [])
            for m in metrics:
                metric_id = m.get('metric_id', '')
                if "DriftedColumnsCount" in metric_id:
                    value = m.get('value', {})
                    if isinstance(value, dict):
                        drifted_columns_count = value.get("count", 0)
                    break
            DATA_DRIFT_DETECTED.set(1 if drift_detected else 0)
            DRIFTED_COLUMNS.set(drifted_columns_count)

        # --- Value Drift (Prediction drift) ---
        vd = reports.get('prediction_drift')
        prediction_drift_detected = False

        if vd:
            vd_result = vd.dict()
            tests = vd_result.get('tests', [])
            for test in tests:
                status = str(test.get("status", "")).upper()
                if status == "FAIL":
                    prediction_drift_detected = True
                    desc = test.get("description", "")
                    alerts.append(f"Prediction drift detected: {desc}")
            PREDICTION_DRIFT_DETECTED.set(1 if prediction_drift_detected else 0)

        # --- classification performance alerts ---
        classy_report = reports.get('classification')
        acc_curr = acc_ref = None

        if classy_report:
            classy_dict = classy_report.dict()
            tests = classy_dict.get("tests", [])[0]
            description = tests.get('description', "")
            match = re.search(r"Actual value ([0-9.]+).*expected ([0-9.]+)", description)
            if match:
                acc_curr = float(match.group(1))  # 0.938
                acc_ref = float(match.group(2))  
            if acc_curr and acc_ref and acc_curr < acc_ref * 0.9:
                alerts.append(f"Model accuracy dropped significantly ({acc_ref:.3f} → {acc_curr:.3f})")
            MODEL_ACCURACY_CURRENT.set(acc_curr)
            MODEL_ACCURACY_REFERENCE.set(acc_ref)

        # Save alerts to a file
        if alerts:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            ALERT_COUNT.inc(len(alerts))
            alerts_path = f"{self.output_path}/alerts_{date_str}.txt"
            
            with open(alerts_path, "a") as f:
                for alert in alerts:
                    f.write(f"{timestamp} - {alert}\n")
            print(f"⚠️ Alerts generated and saved to alerts_{timestamp}.txt")
        else:
            print("✅ No alerts generated.")
