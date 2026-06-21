print("MAIN STARTED")

import src
import numpy as np
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

print("IMPORTS DONE")

dt = DataTransformation()

print("TRANSFORMATION OBJECT CREATED")

X_train_scaled, X_test_scaled, y_train, y_test, _ = dt.initiate_data_transformation(
    "artifacts/train.csv",
    "artifacts/test.csv"
)

train_arr = np.c_[X_train_scaled.values, y_train.values]
test_arr = np.c_[X_test_scaled.values, y_test.values]

print("DATA TRANSFORMATION DONE")

mt = ModelTrainer()

print("MODEL TRAINER CREATED")

result = mt.initiate_model_training(train_arr, test_arr)

print("MODEL TRAINING DONE")

print(result)