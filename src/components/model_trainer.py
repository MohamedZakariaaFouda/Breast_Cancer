import os
import sys
import pandas as pd
import numpy as np

from dataclasses import dataclass

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier
)
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

from src.components.feature_selection import get_feature_selectors


# ==========================================================
# CONFIG
# ==========================================================
@dataclass
class ModelTrainerConfig:
    model_file_path: str = os.path.join("artifacts", "best_model.pkl")
    selector_file_path: str = os.path.join("artifacts", "best_selector.pkl")
    results_file_path: str = os.path.join("artifacts", "model_results.csv")


# ==========================================================
# MODEL TRAINER
# ==========================================================
class ModelTrainer:

    def __init__(self):
        self.config = ModelTrainerConfig()

    # ------------------------------------------------------
    # Models
    # ------------------------------------------------------
    def get_models(self):
        return {
            "Naive Bayes": GaussianNB(),

            "Logistic Regression": LogisticRegression(
                random_state=42,
                max_iter=1000
            ),

            "KNN": KNeighborsClassifier(n_neighbors=5),

            "SVC": SVC(random_state=42),

            "Random Forest": RandomForestClassifier(
                random_state=42,
                n_estimators=100
            ),

            "XGBoost": XGBClassifier(
                random_state=42,
                eval_metric="logloss"
            ),

            "AdaBoost": AdaBoostClassifier(
                random_state=42,
                n_estimators=100
            ),

            "Neural Network": MLPClassifier(
                random_state=42,
                max_iter=1000,
                hidden_layer_sizes=(100,)
            ),

            "Gradient Boosting": GradientBoostingClassifier(
                random_state=42,
                n_estimators=100
            )
        }

    # ------------------------------------------------------
    # TRAINING PIPELINE
    # ------------------------------------------------------
    def initiate_model_training(self, train_arr, test_arr):

        try:
            logging.info("Starting training pipeline")

            # --------------------------
            # Split data
            # --------------------------
            X_train = train_arr[:, :-1]
            y_train = train_arr[:, -1]

            X_test = test_arr[:, :-1]
            y_test = test_arr[:, -1]

            # convert to DataFrame
            X_train = pd.DataFrame(X_train)
            X_test = pd.DataFrame(X_test)

            selectors = get_feature_selectors()
            models = self.get_models()

            results = []

            best_score = -1
            best_model = None
            best_selector = None
            best_model_name = None
            best_selector_name = None

            # ==================================================
            # FEATURE SELECTION LOOP
            # ==================================================
            for selector_name, selector in selectors.items():

                logging.info(f"Using selector: {selector_name}")

                # fit selector
                selector.fit(X_train, y_train)

                X_train_sel = selector.transform(X_train)
                X_test_sel = selector.transform(X_test)

                # convert if numpy
                if isinstance(X_train_sel, np.ndarray):
                    X_train_sel = pd.DataFrame(X_train_sel)

                if isinstance(X_test_sel, np.ndarray):
                    X_test_sel = pd.DataFrame(X_test_sel)

                # ==================================================
                # MODEL LOOP
                # ==================================================
                for model_name, model in models.items():

                    logging.info(f"Training model: {model_name}")

                    model.fit(X_train_sel, y_train)
                    y_pred = model.predict(X_test_sel)

                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred, zero_division=0)
                    recall = recall_score(y_test, y_pred, zero_division=0)
                    f1 = f1_score(y_test, y_pred, zero_division=0)

                    results.append({
                        "Feature_Selection": selector_name,
                        "Model": model_name,
                        "Accuracy": accuracy,
                        "Precision": precision,
                        "Recall": recall,
                        "F1": f1
                    })

                    # ==================================================
                    # BEST MODEL (IMPORTANT FIX → F1 instead of Accuracy)
                    # ==================================================
                    if f1 > best_score:
                        best_score = f1
                        best_model = model
                        best_selector = selector
                        best_model_name = model_name
                        best_selector_name = selector_name

            # --------------------------
            # SAVE RESULTS
            # --------------------------
            results_df = pd.DataFrame(results)
            results_df.sort_values(by="F1", ascending=False, inplace=True)

            os.makedirs("artifacts", exist_ok=True)
            results_df.to_csv(self.config.results_file_path, index=False)

            # --------------------------
            # SAVE BEST MODEL + SELECTOR
            # --------------------------
            save_object(self.config.model_file_path, best_model)
            save_object(self.config.selector_file_path, best_selector)

            logging.info(f"Best Model: {best_model_name}")
            logging.info(f"Best Selector: {best_selector_name}")
            logging.info(f"Best F1 Score: {best_score:.4f}")

            return {
                "best_model": best_model_name,
                "best_selector": best_selector_name,
                "best_f1": best_score,
                "results": results_df
            }

        except Exception as e:
            raise CustomException(e, sys)