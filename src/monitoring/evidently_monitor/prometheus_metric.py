from prometheus_client import start_http_server, Gauge, Counter
import threading

# Define metrics
DATA_DRIFT_DETECTED = Gauge(
    'employee_attrition_data_drift_detected', 
    '1 if data drift is detected, 0 otherwise'
)
DRIFTED_COLUMNS = Gauge(
    'employee_attrition_drifted_columns_count', 
    'Number of drifted columns'
)
PREDICTION_DRIFT_DETECTED = Gauge(
    'employee_attrition_prediction_drift_detected', 
    '1 if prediction drift is detected, 0 otherwise'
)
MODEL_ACCURACY_LIVE = Gauge(
    'employee_attrition_model_accuracy_current', 
    'Current dataset model accuracy'
)
MODEL_ACCURACY_REFERENCE = Gauge(
    'employee_attrition_model_accuracy_reference', 
    'Reference dataset model accuracy')
ALERT_COUNT = Counter(
    'employee_attrition_monitor_alerts_total', 
    'Number of generated alerts'
)

_server_started = False

def start_prometheus_server(port: int = 9090):
    global _server_started
    if _server_started:
        return
    def _start():
        start_http_server(port)
    t = threading.Thread(target=_start, daemon=True)
    t.start()
    print(f"🚀 Prometheus metrics available at http://localhost:{port}/metrics")
    _server_started = True
