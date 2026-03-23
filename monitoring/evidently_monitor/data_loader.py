import pandas as pd
import sqlite3
from evidently import DataDefinition, Dataset, MulticlassClassification

class DataLoader:
    def __init__(self, reference_db_path, live_db_path):
        self.reference_db_path = reference_db_path
        self.live_db_path = live_db_path
        self.data_definition = self._create_column_mapping()

    def _create_column_mapping(self):
        dd = DataDefinition(
            classification=[MulticlassClassification(
                target='class_name',
                prediction_labels='prediction',
            )],
            numerical_columns=["legs"],
            categorical_columns=[
                'hair', 'eggs', 'milk', 'predator', 'toothed',
                'backbone', 'breathes', 'venomous', 'tail',
                'can_fly', 'can_swim', 'is_domestic_pet', 'prediction'
            ],
            datetime_columns=["event_timestamp"]
        )
        return dd
    
    def load_reference_data(self):
        conn = sqlite3.connect(self.reference_db_path)
        query = "SELECT * FROM reference_data"
        df = pd.read_sql(query, conn)
        conn.close()

        df['event_timestamp'] = pd.to_datetime(df['event_timestamp'])
        
        if "prediction" not in df.columns:
            df["prediction"] = df["class_name"]

        return df
    

    def load_live_data(self):
        conn = sqlite3.connect(self.live_db_path)
        query = "SELECT * FROM live_data"
        df = pd.read_sql(query, conn)
        conn.close()

        df['event_timestamp'] = pd.to_datetime(df['event_timestamp'])
        
        if "class_name" not in df.columns and "true_label" in df.columns:
            df.rename(columns={'true_label': 'class_name'}, inplace=True)

        return df
    
    
    def to_datasets(self, ref_df, live_df):
        return (
            Dataset.from_pandas(ref_df, data_definition=self.data_definition),
            Dataset.from_pandas(live_df, data_definition=self.data_definition),
        )
    