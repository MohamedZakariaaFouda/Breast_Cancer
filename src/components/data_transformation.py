import os
import sys
import numpy as np
import pandas as pd

from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


# =========================
# CONFIG
# =========================
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join(
        "artifacts", "preprocessor.pkl"
    )


# =========================
# DATA TRANSFORMATION
# =========================
class DataTransformation:

    def __init__(self):
        self.config = DataTransformationConfig()

    # -------------------------------------------------
    # Build Preprocessing Pipeline
    # -------------------------------------------------
    def get_data_transformer_object(self, X: pd.DataFrame):
        try:
            logging.info("Creating preprocessing pipeline")

            # All features are numeric after removing target
            numerical_columns = X.columns.tolist()

            num_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", num_pipeline, numerical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    # -------------------------------------------------
    # Main Transformation Pipeline
    # -------------------------------------------------
    def initiate_data_transformation(self, train_path, test_path):
        try:
            logging.info("Reading data")

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            # -------------------------
            # Drop useless columns
            # -------------------------
            drop_cols = ["id", "Unnamed: 32"]

            train_df.drop(columns=drop_cols, inplace=True, errors="ignore")
            test_df.drop(columns=drop_cols, inplace=True, errors="ignore")

            target_column = "diagnosis"

            # -------------------------
            # Encode target (BEST PRACTICE)
            # -------------------------
            train_df[target_column] = train_df[target_column].map({"M": 1, "B": 0})
            test_df[target_column] = test_df[target_column].map({"M": 1, "B": 0})

            # -------------------------
            # Split X / y
            # -------------------------
            X_train = train_df.drop(columns=[target_column])
            y_train = train_df[target_column]

            X_test = test_df.drop(columns=[target_column])
            y_test = test_df[target_column]

            logging.info("Building preprocessor")

            preprocessor = self.get_data_transformer_object(X_train)

            # -------------------------
            # Fit & transform
            # -------------------------
            X_train_scaled = preprocessor.fit_transform(X_train)
            X_test_scaled = preprocessor.transform(X_test)

            # Keep DataFrame (NOT only numpy)
            feature_names = X_train.columns.tolist()

            X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_names)
            X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_names)

            logging.info("Saving preprocessor")

            save_object(
                file_path=self.config.preprocessor_obj_file_path,
                obj=preprocessor
            )

            logging.info("Data transformation completed successfully")

            # -------------------------
            # RETURN CLEAN STRUCTURE
            # -------------------------
            return (
                X_train_scaled,
                X_test_scaled,
                y_train,
                y_test,
                self.config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)