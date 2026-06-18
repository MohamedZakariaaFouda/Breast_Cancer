import os
import sys
import pandas as pd

from dataclasses import dataclass, field
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging


# ================================
# CONFIG
# ================================
@dataclass
class DataIngestionConfig:
    artifact_dir: str = "artifacts"

    train_data_path: str = field(init=False)
    test_data_path: str = field(init=False)
    raw_data_path: str = field(init=False)

    def __post_init__(self):
        self.train_data_path = os.path.join(self.artifact_dir, "train.csv")
        self.test_data_path = os.path.join(self.artifact_dir, "test.csv")
        self.raw_data_path = os.path.join(self.artifact_dir, "data.csv")


# ================================
# DATA INGESTION
# ================================
class DataIngestion:
    def __init__(self):
        self.config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Starting data ingestion process")

        try:
            # Read data
            df = pd.read_csv("data/raw/data.csv")
            logging.info("Dataset loaded successfully")

            # Create artifacts folder
            os.makedirs(self.config.artifact_dir, exist_ok=True)

            # Save raw data
            df.to_csv(self.config.raw_data_path, index=False)

            # Train-test split
            train_set, test_set = train_test_split(
                df,
                test_size=0.2,
                random_state=42
            )

            # Save train data
            train_set.to_csv(self.config.train_data_path, index=False)

            # Save test data
            test_set.to_csv(self.config.test_data_path, index=False)

            logging.info("Data ingestion completed successfully")

            return self.config.train_data_path, self.config.test_data_path

        except Exception as e:
            raise CustomException(e, sys)


# ================================
# RUN PIPELINE
# ================================
if __name__ == "__main__":
    data_ingestion = DataIngestion()

    train_path, test_path = data_ingestion.initiate_data_ingestion()

    print(f"Train data saved to: {train_path}")
    print(f"Test data saved to: {test_path}")