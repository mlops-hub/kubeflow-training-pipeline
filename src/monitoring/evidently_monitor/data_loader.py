import pandas as pd
import sqlite3
from evidently import DataDefinition, Dataset, BinaryClassification
from scripts.schema import enforce_schema

class DataLoader:
    def __init__(self, reference_db_path, live_db_path):
        self.reference_db_path = reference_db_path
        self.live_db_path = live_db_path
        self.data_definition = self._create_column_mapping()

    def _create_column_mapping(self):
        dd = DataDefinition(
            classification=[BinaryClassification(
                target="target", # the common use in both datasets(ref, live)
                prediction_labels="prediction",
            )],
            numerical_columns=[
                "years_at_company", "company_tenure", "annual_income", 
                "role_stagnation_ratio", "tenure_gap", "number_of_promotions", "number_of_dependents",
            ],
            categorical_columns=[
                "education_level", "job_level", "company_size", "performance_rating", 
                "age_group", "overall_satisfaction", "opportunities", "company_reputation",
            ],
            id_column="employee_id",
            timestamp="event_timestamp"
        )
        return dd
    
    def load_reference_data(self):
        conn = sqlite3.connect(self.reference_db_path)
        query = "SELECT * FROM reference_data"
        df = pd.read_sql(query, conn)
        conn.close()
        df['event_timestamp'] = pd.to_datetime(df['event_timestamp'])
        # rename target column
        df = df.rename(columns={'attrition': 'target'})
        df['prediciton'] = df['target'].copy()
        return df
    

    def load_sample_reference_data(self):
        conn = sqlite3.connect(self.reference_db_path)
        query = "SELECT * FROM reference_data"
        df = pd.read_sql(query, conn)
        conn.close()
        df['event_timestamp'] = pd.to_datetime(df['event_timestamp'])
        # rename target column
        df = df.rename(columns={'attrition': 'target'})
        df['prediciton'] = df['target'].copy()
        df_sampled = df.sample(n=500, random_state=42)
        return df_sampled
    

    def load_live_data(self):
        conn = sqlite3.connect(self.live_db_path)
        query = "SELECT * FROM live_data"
        df = pd.read_sql(query, conn)
        conn.close()
        df['event_timestamp'] = pd.to_datetime(df['event_timestamp'])
        return df
    
    
    def to_datasets(self, ref_df, ref_sample_df, live_df):
        return (
            Dataset.from_pandas(ref_df, data_definition=self.data_definition),
            Dataset.from_pandas(ref_sample_df, data_definition=self.data_definition),
            Dataset.from_pandas(live_df, data_definition=self.data_definition),
        )
    