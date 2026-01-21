import joblib
import pandas as pd
from pathlib import Path
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS


BASE_DIR = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = BASE_DIR / "artifacts" / "model_v1"

MODEL_ARTIFACT = ARTIFACT_PATH / "best_model.pkl"
FEATURE_STORE = ARTIFACT_PATH / "features.pkl"
SCALER_PATH = ARTIFACT_PATH / "preprocessor.pkl"

THRESHOLD = 0.35 # tuned for recall``

app = Flask(__name__)
CORS(app)

model = joblib.load(MODEL_ARTIFACT)
features = joblib.load(FEATURE_STORE)
prerpocessor = joblib.load(SCALER_PATH)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    print('incoming-data: ', data)

    try:
        df_input = pd.DataFrame([data], columns=features)

        numeric_cols = ['Years at Company', 'Company Tenure', 'RoleStagnationRatio', 'TenureGap']
        df_input[numeric_cols] = prerpocessor.transform(df_input[numeric_cols])

        probability = model.predict_proba(df_input)[0]
        p_stay = float(probability[0])
        p_leave = probability[1]
        print('proba: ', probability)

        prediction = int(p_leave >= THRESHOLD)

        if p_leave < 0.30: 
            risk = "Low"
        elif p_leave < 0.60:
            risk = "Medium"
        else:
            risk = "High"

        return jsonify({
            "prediction": prediction,
            "p_leave": round(p_leave, 4),
            "p_stay": round(p_stay, 4),
            "risk": risk,
            "threshold": THRESHOLD
        })
    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 400


if __name__ == "__main__":
    app.run(debug=True)