from feast import Entity, FeatureView, Field, FileSource, ValueType, FeatureService
from feast.types import Int64, Float32
from datetime import timedelta
from dotenv import load_dotenv
import os

load_dotenv()

# 1. Define the entity (the primary key)
employee = Entity(
    name="employee_id",
    value_type=ValueType.INT64,
    description="Employee ID as primary key",
)

# 2. Define the Data Source
# This points to where your preprocessed data is stored (Parquet is recommended for Feast)
DATA_PATH = os.getenv("FEATURE_DATA_PATH", "data/preprocessed_data.parquet")
print(DATA_PATH)

employee_preprocessed_source = FileSource(
    # path="data/preprocessed_data.parquet", # Path to offline store
    path="data/preprocessed_data.parquet",
    event_timestamp_column="event_timestamp",   # Feast requires a timestamp column
)

# 3. Define the Feature View
employee_features_fv = FeatureView(
    name="employee_features",
    entities=[employee],
    ttl=timedelta(days=365), # A long TTL as these are mostly static features
    schema=[
        Field(name="years_at_company", dtype=Float32),
        Field(name="performance_rating", dtype=Int64),
        Field(name="number_of_promotions", dtype=Int64),
        Field(name="overtime", dtype=Int64),
        Field(name="education_level", dtype=Int64),
        Field(name="number_of_dependents", dtype=Int64),
        Field(name="job_level", dtype=Int64),
        Field(name="company_size", dtype=Int64),
        Field(name="company_tenure", dtype=Float32),
        Field(name="remote_work", dtype=Int64),
        Field(name="company_reputation", dtype=Int64),
        Field(name="attrition", dtype=Int64),  # The target label for training
        Field(name="overall_satisfaction", dtype=Int64),
        Field(name="opportunities", dtype=Int64),
        Field(name="annual_income", dtype=Int64),
        Field(name="age_group", dtype=Int64),
        Field(name="role_stagnation_ratio", dtype=Float32),
        Field(name="tenure_gap", dtype=Float32),
        Field(name="early_company_tenure_risk", dtype=Float32),
        Field(name="long_tenure_low_role_risk", dtype=Float32),
    ],
    online=True,
    source=employee_preprocessed_source,
    tags={"team": "hr_analytics"}
)


# Define a FeatureService for your employee attrition model
employee_features_fs = FeatureService(
    name="employee_attrition_features",
    features=[
        employee_features_fv[[
            "age_group",
            "annual_income",
            "size",
            "comcompany_reputation",
            "company_pany_tenure",
            "education_level",
            "early_company_tenure_risk",
            "job_level",
            "long_tenure_low_role_risk",
            "number_of_dependents",
            "number_of_promotions",
            "opportunities",
            "overtime",
            "overall_satisfaction",
            "performance_rating",
            "remote_work",
            "role_stagnation_ratio",
            "tenure_gap",
            "years_at_company",
            "attrition",
        ]]
    ]
    # Note: 'attrition_label' is not included here as it's typically the target, not a feature for inference
)

#  testing or logging the feature values
feature_count = len(employee_features_fv.schema)
print("total features in feast:", feature_count)