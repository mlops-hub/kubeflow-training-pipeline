# | Column               | Meaning                                | Evidence                       |
# | -------------------- | -------------------------------------- | ------------------------------ |
# | **years_at_company** | Years in the *current role / position* | Usually smaller, role-specific |
# | **company_tenure**   | Total years employed at the company    | Usually ≥ years_at_company     |

import pandas as pd


def feature_data(df_path: str) -> pd.DataFrame:

    df_fe = pd.read_csv(df_path)

    # Encoding
    # ------------------------------------
    # 1. Target column
    df_fe["attrition"] = df_fe["attrition"].map({"Stayed": 0, "Left": 1}).astype("int")

    # 2. Binary Encoding
    binary_cols = ["remote_work", "leadership_opportunities", "innovation_opportunities", "overtime"]
    for col in binary_cols:
        df_fe[col] = df_fe[col].map({"Yes": 1, "No": 0}).astype("Int64")


    # ordinal encoding for features
    ordinal_maps = {
        "work_life_balance": { "Poor": 1, "Fair": 2, "Good": 3, "Excellent": 4 },
        "job_satisfaction": { "Low": 1, "Medium": 2, "High": 3, "Very High": 4 },
        "performance_rating": { "Low": 1, "Below Average": 2, "Average": 3, "High": 4 },
        "employee_recognition": { "Low": 1, "Medium": 2, "High": 3, "Very High": 4 },
        "company_reputation": { "Poor": 1, "Fair": 2, "Good": 3, "Excellent": 4 },
        "job_level": { "Entry": 1, "Mid": 2, "Senior": 3},
        "company_size": { "Small": 1, "Medium": 2, "Large": 3 },
        "education_level": {"High School": 1, "Bachelor’s Degree": 2, "Master’s Degree": 3, "Associate Degree": 4, "PhD": 5},
    }

    for col, mapping in ordinal_maps.items():
        df_fe[col] = df_fe[col].map(mapping)


    # Aggregate satisfaction
    # --------------------------------------------------
    df_fe["overall_satisfaction"] = (
        (df_fe["work_life_balance"] + df_fe["job_satisfaction"] + df_fe["employee_recognition"]) / 3
    ).round().astype("Int64")
    df_fe = df_fe.drop(columns=["work_life_balance", "job_satisfaction", "employee_recognition"])

    df_fe["opportunities"] = df_fe["leadership_opportunities"] + df_fe["innovation_opportunities"]
    df_fe = df_fe.drop(columns=["leadership_opportunities", "innovation_opportunities"])

    # annual income binning
    # -----------------------------------------------
    df_fe["annual_income"] = df_fe["monthly_income"] * 12

    print(df_fe["annual_income"].min(), "-", df_fe["annual_income"].max())
    # 14712 - 193788

    df_fe["annual_income"] = pd.cut(
        df_fe["annual_income"],
        bins=[0, 30000, 60000, 90000, 120000, float("inf")],
        labels=[1, 2, 3, 4, 5],
        include_lowest=True,
    ).astype("Int64")
    df_fe = df_fe.drop(columns=["monthly_income"])

    # age binning
    print(df_fe['age'].min(), "-", df_fe['age'].max())
    df_fe["age_group"] = pd.cut(
        df_fe["age"],
        bins=[17, 25, 35, 45, 60, 65],
        labels=[1, 2, 3, 4, 5],
    ).astype("Int64")
    df_fe = df_fe.drop(columns=["age"])

    # Years at company and company tenure
    # -----------------------------------------------------
    # 1. Convert to proper years as dataset is in months.
    df_fe["years_at_company"] = (df_fe["years_at_company"] / 12).round(2)
    df_fe["company_tenure"] = (df_fe["company_tenure"] / 12).round(2)

    # 2. Role Stagnation: 1 -> same role entire tenure (possible stagnation); low value -> role mobility
    df_fe["role_stagnation_ratio"] = (df_fe["years_at_company"] / (df_fe["company_tenure"] + 1)).round(3)

    # 3. Tenure Gap: high gap -> role changes/promotion; low gap -> same role for long time
    df_fe["tenure_gap"] = df_fe["company_tenure"] - df_fe["years_at_company"]

    # 4. Early company risk: most attrition happens in first 2 years
    df_fe["early_company_tenure_risk"] = (df_fe["years_at_company"] <= 2).astype("Int64")

    # 5. Long Term Stagnation
    df_fe["long_tenure_low_role_risk"] = ((df_fe["company_tenure"] > 5) & (df_fe["job_level"] <= 2)).astype("Int64")

    # drop unnecessary columns
    df_fe = df_fe.drop(columns=["job_role", "distance_from_home", "marital_status", "gender", "dataset_type"])

    print(df_fe.tail(10))

    return df_fe


if __name__ == "__main__":
    from pathlib import Path

    BASE_DIR = Path(__file__).resolve().parents[2]
    DATASET_PATH = BASE_DIR / "datasets" / "data-pipeline"
    CLEANED_PATH = DATASET_PATH / "04_cleaned_df.csv"
    FEATURED_PATH = DATASET_PATH / "05_feature_engg_df.csv"

    feature_df = feature_data(CLEANED_PATH)

    feature_df.to_csv(FEATURED_PATH, index=False)
