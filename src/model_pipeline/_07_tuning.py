import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, recall_score

def tuning_data(df_train, df_test, model):
    X_train = df_train.drop(columns=['Attrition'])
    y_train = df_train['Attrition']

    X_test = df_test.drop(columns=['Attrition'])
    y_test = df_test['Attrition']

    # set parameters
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'solver': ['liblinear', 'saga'],
        'l1_ratio': [0],     # equivalent to L2
        'max_iter': [1000]
    }

    # set cv
    strat_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    # use gridserachcv for tuning model
    grid = GridSearchCV(
        estimator=model, 
        param_grid=param_grid, 
        cv=strat_cv, 
        scoring='recall',
        n_jobs=-1
    )
    grid.fit(X_train, y_train)

    # get best model and save in models/
    tuned_model = grid.best_estimator_

    best_parameters = grid.best_params_
    print(f'best params: {best_parameters}')
        
    # predict the output with tuned_model
    y_pred = tuned_model.predict(X_test)

    # tuned model evaluation
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "train_score": tuned_model(X_train, X_test),
        "test_score": tuned_model(X_test, y_test)
    }        
    print('accuracy: ', metrics['accuracy_ht'])
    print('recall: ', metrics['recall_ht'])

    # Compare CV scores of base model and tuned model
    base_cv_scores = cross_val_score(model, X_train, y_train, cv=strat_cv)
    tuned_cv_scores = cross_val_score(tuned_model, X_train, y_train, cv=strat_cv)
    print("=== MODEL COMPARISON ===")
    print(f"Base Model CV: {base_cv_scores.mean():.4f} (+/- {base_cv_scores.std():.4f})")
    print(f"Tuned Model CV: {tuned_cv_scores.mean():.4f} (+/- {tuned_cv_scores.std():.4f})")

    cv_summary = {
        "base_cv_mean": base_cv_scores.mean(),
        "tuned_cv_mean": tuned_cv_scores.mean()
    }

    overall_parameters = {
        **best_parameters,
        **metrics,
        **cv_summary
    }

    return tuned_model, overall_parameters



if __name__ == "__main__":
    from sklearn.linear_model import LogisticRegression
    from dotenv import load_dotenv
    from pathlib import Path
    import os
    import json
    import joblib

    load_dotenv()

    BASE_DIR = Path(__file__).resolve().parents[2]
    PREPROCESSED_TRAIN_PATH = BASE_DIR / "datasets" / "data-engg" / "06_preprocess_train_df.csv"
    PREPROCESSED_TEST_PATH = BASE_DIR / "datasets" / "data-engg" / "06_preprocess_test_df.csv"

    MODEL_ARTIFACT = BASE_DIR / "artifacts" / "model_v1" / "model.pkl"
    TUNED_MODEL_ARTIFACT = BASE_DIR / "artifacts" / "model_v1" / "tuned_model.pkl"
    TUNING_METADATA = BASE_DIR / "artifacts" / "model_v1" / "tuning_metadata.json"
    os.makedirs(TUNED_MODEL_ARTIFACT.parent, exist_ok=True)

    
    df_train = pd.read_csv(PREPROCESSED_TRAIN_PATH)
    df_test = pd.read_csv(PREPROCESSED_TEST_PATH)

    model = LogisticRegression(max_iter=1000, class_weight='balanced')

    tuned_model, overall_parameters = tuning_data(df_train, df_test, model)
    joblib.dump(tuned_model, TUNED_MODEL_ARTIFACT)

    with open(TUNING_METADATA, 'w') as f:
        json.dump(overall_parameters, f)
