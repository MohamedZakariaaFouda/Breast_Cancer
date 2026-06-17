import os
import sys
import pandas as pd

from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging


@dataclass
class DataIngestionConfig:
    artifact_dir: str = "artifacts"

    train_data_path: str = os.path.join(artifact_dir, "train.csv")
    test_data_path: str = os.path.join(artifact_dir, "test.csv")
    raw_data_path: str = os.path.join(artifact_dir, "data.csv")


class DataIngestion:
    def __init__(self):
        self.config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Starting data ingestion process")

        try:
            # Read dataset
            df = pd.read_csv("data/raw/data.csv")
            logging.info("Dataset loaded successfully")

            # Create artifacts directory
            os.makedirs(self.config.artifact_dir, exist_ok=True)

            # Save raw data
            df.to_csv(self.config.raw_data_path,
                      index=False,
                      header=True)

            # Split data
            train_set, test_set = train_test_split(
                df,
                test_size=0.2,
                random_state=42
            )

            # Save train and test data
            train_set.to_csv(
                self.config.train_data_path,
                index=False,
                header=True
            )

            test_set.to_csv(
                self.config.test_data_path,
                index=False,
                header=True
            )

            logging.info("Data ingestion completed successfully")

            return (
                self.config.train_data_path,
                self.config.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    data_ingestion = DataIngestion()

    train_path, test_path = (
        data_ingestion.initiate_data_ingestion()
    )

    print(f"Train data saved to: {train_path}")
    print(f"Test data saved to: {test_path}")