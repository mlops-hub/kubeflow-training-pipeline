import os
import pandas as pd
from pathlib import Path
import joblib
from sklearn.linear_model import LogisticRegression
from dotenv import load_dotenv
import argparse

load_dotenv()

# BASE_DIR = Path(__file__).resolve().parents[2]
# PREPROCESSED_TRAIN_PATH = BASE_DIR / "datasets" / "data-engg" / "06_preprocess_train_df.csv"

# ARTIFACT_PATH = BASE_DIR / "artifacts" / "model_v1"
# os.makedirs(ARTIFACT_PATH, exist_ok=True)

# MODEL_ARTIFACT = ARTIFACT_PATH / "model.pkl"
# FEATURE_STORE = ARTIFACT_PATH / "features.pkl"


def training_data(train_path: str, model_path: str, feature_store_path: str):
    df = pd.read_csv(train_path)

    X_train = df.drop(columns=['Attrition'])
    y_train = df['Attrition']

    # save features before training
    feature_columns = X_train.columns.to_list()
    joblib.dump(feature_columns, feature_store_path)

    print("Training the model....")
    model = LogisticRegression(max_iter=1000, class_weight='balanced')
    model.fit(X_train, y_train)

    print("training completed...")
    joblib.dump(model, model_path)

    return model



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--feature_store_path", required=True)
    args = parser.parse_args()

    training_data(
        train_path=args.train_path,
        model_path=args.model_path, 
        feature_store_path=args.feature_store_path,
    )


# run command
# python src/training.py \
#     --train_path datasets/data-engg/06_preprocess_train_df.csv \
#     --model_path artifacts/model_v1/model.pkl \
#     --feature_store_path artifacts/model_v1/features.pkl
