def enforce_schema(df):
    # df["employee_id"] = df["employee_id"].astype("int64")
    df["employee_id"] = df["employee_id"].fillna(-1).astype("int64")

    # integers
    int_cols = [
        "age_group", "years_at_company", "annual_income",
        "overall_satisfaction", "performance_rating",
        "number_of_promotions", "overtime", "education_level",
        "number_of_dependents", "job_level", "company_size",
        "remote_work", "opportunities", "company_reputation",
        "prediction", "target"
    ]

    for col in int_cols:
        if col in df:
            df[col] = df[col].astype("int64")

    # floats
    float_cols = [
        "company_tenure",
        "role_stagnation_ratio",
        "tenure_gap",
        "early_company_tenure_risk",
        "long_tenure_low_role_risk"
    ]

    for col in float_cols:
        if col in df:
            df[col] = df[col].astype("float64")

    # timestamp (CRITICAL)
    df["event_timestamp"] = pd.to_datetime(df["event_timestamp"])

    return df