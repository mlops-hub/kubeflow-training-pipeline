import argparse
import json
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression


def training_data(train_path: str, best_params_path: str, model_path: str, feature_store_path: str):
    df = pd.read_csv(train_path)
    X_train = df.drop(columns=['Attrition'])
    y_train = df['Attrition']

    # load parameters
    with open(best_params_path, 'r') as f:
        overall_params = json.load(f)

    best_params = {
        key: overall_params[key]
        for key in overall_params
        if key not in ["accuracy", "recall", "train_score", "test_score", "base_cv_mean", "base_cv_std", "tuned_cv_mean", "tuned_cv_std"]
    }

    # save features before training
    feature_columns = X_train.columns.to_list()
    joblib.dump(feature_columns, feature_store_path)

    print("Training the model....")
    model = LogisticRegression(**best_params)
    model.fit(X_train, y_train)

    print("training completed...")
    joblib.dump(model, model_path)

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", required=True)
    parser.add_argument("--best_params_path", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--feature_store_path", required=True)
    args = parser.parse_args()

    training_data(
        train_path=args.train_path,
        best_params_path=args.best_params_path,
        model_path=args.model_path, 
        feature_store_path=args.feature_store_path,
    )


# run command
# python src/model_pipeline/_08_training.py \
#     --train_path datasets/data-engg/06_preprocess_train_df.csv \
#     --best_params_path artifacts/model_v1/tuning_metadata.json \
#     --model_path artifacts/model_v1/best_model.pkl \
#     --feature_store_path artifacts/model_v1/features.pkl
