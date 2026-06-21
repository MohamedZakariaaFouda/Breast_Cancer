import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler


# ==========================================================
# Filter Method
# ==========================================================
class FilterSelector(BaseEstimator, TransformerMixin):

    def __init__(self, columns_to_drop=None):
        self.columns_to_drop = columns_to_drop or []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(columns=self.columns_to_drop, errors="ignore")


# ==========================================================
# SelectKBest (Chi-Square FIXED)
# ==========================================================
class SelectKBestSelector(BaseEstimator, TransformerMixin):

    def __init__(self, k=5):
        self.k = k
        self.scaler = MinMaxScaler()
        self.selector = SelectKBest(score_func=chi2, k=self.k)

    def fit(self, X, y):

        # 🔥 FIX: chi2 requires non-negative values
        X_scaled = self.scaler.fit_transform(X)

        self.selector.fit(X_scaled, y)
        return self

    def transform(self, X):

        X_scaled = self.scaler.transform(X)
        return self.selector.transform(X_scaled)

    def get_selected_features(self, X):

        mask = self.selector.get_support()
        return X.columns[mask].tolist()


# ==========================================================
# Random Forest Importance
# ==========================================================
class RandomForestSelector(BaseEstimator, TransformerMixin):

    def __init__(self, n_features=10, n_estimators=100, random_state=42):
        self.n_features = n_features

        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state
        )

        self.selected_features_ = None

    def fit(self, X, y):

        self.model.fit(X, y)

        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]

        self.selected_features_ = X.columns[indices[:self.n_features]].tolist()

        return self

    def transform(self, X):

        return X[self.selected_features_]

    def get_selected_features(self):
        return self.selected_features_


# ==========================================================
# PCA Selector
# ==========================================================
class PCASelector(BaseEstimator, TransformerMixin):

    def __init__(self, n_components=5):
        self.n_components = n_components
        self.pca = PCA(n_components=self.n_components)

    def fit(self, X, y=None):

        self.pca.fit(X)
        return self

    def transform(self, X):

        return self.pca.transform(X)

    def explained_variance(self):

        return self.pca.explained_variance_ratio_


# ==========================================================
# Factory
# ==========================================================
def get_feature_selectors():

    return {

        "Filter": FilterSelector(
            columns_to_drop=[
                "radius_mean",
                "area_mean",
                "texture_mean",
                "smoothness_mean",
                "symmetry_mean",
                "fractal_dimension_mean",

                "radius_se",
                "perimeter_se",
                "concave points_se",
                "smoothness_se",
                "concavity_se",
                "symmetry_se",
                "texture_se",

                "perimeter_worst",
                "radius_worst",
                "compactness_worst",
                "texture_worst",
                "smoothness_worst",
                "symmetry_worst",
                "fractal_dimension_worst",
            ]
        ),

        "SelectKBest": SelectKBestSelector(k=5),

        "RandomForest": RandomForestSelector(n_features=10),

        "PCA": PCASelector(n_components=5),
    }
