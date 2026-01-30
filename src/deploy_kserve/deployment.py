import pandas as pd
from kserve import Model, ModelServer
from _mlflow.registry import MLflowRegistry
import os


class EmployeeAttiritionPrediction(Model):
    def __init__(self, name, model_uri):
        super().__init__(name)
        self.model_uri = model_uri
        self.ready = False

    def load(self, tracking_uri, experiment_name):
        try:
            registry = MLflowRegistry(
                tracking_uri=tracking_uri,
                experiment_name=experiment_name
            )

            self.model = registry.load_model(model_uri=model_uri)
            self.ready = True
        except Exception as e:
            print(f"Error during load: {e}")
            self.ready = False

    def predict(self, payload, headers=None):
        print(f"Recieved payload: {payload}")

        instances = payload.get("instances", [])

        if not instances:
            return {"error": "No instances provided."}
        
        df = pd.DataFrame(instances)
        try:
            prediction = self.model.predict(df)
            probs = self.model.predict_proba(df)
            probability = probs.max(axis=1).tolist()
            print('prediction: ', prediction)
            print('probs: ', probs)
            print("probability: ", probability)

            return {
                "prediction": prediction,
                "probs": probability,
            }
        
        except Exception as e:
            print(f"Error during prediciton: {e}")
            return {"error": str(e)}
        

if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    MODEL_VERSION = os.environ.get("MODEL_VERSION", "3")
    MODEL_NAME = os.environ.get("MODEL_NAME", "register-employee-attrition-model")
    KSERVE_PORT = int(os.environ.get("KSERVE_PORT", 7070))


    model_uri = f"models:/{MODEL_NAME}/{MODEL_VERSION}"
    print(f"Using MLflow tracking URI: {MLFLOW_TRACKING_URI}")
    print(f"Using model registry URI: {model_uri}")

    server = ModelServer(http_port=KSERVE_PORT)

    model = EmployeeAttiritionPrediction(
        name="employee_attrition_prediction",
        model_uri=model_uri
    )
    
    model.load(
        tracking_uri="http://localhost:5000",
        experiment_name="employee-attrition-v1"
    )
    server.start(models=[model])

