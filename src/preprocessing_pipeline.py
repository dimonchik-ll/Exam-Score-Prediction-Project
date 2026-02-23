import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from data import X_train

cat_features_for_onehot = ["gender", "course", "study_method", "internet_access"]
cat_features_for_ordinal = ["sleep_quality", "facility_rating", "exam_difficulty"]
num_fetures = ["age", "study_hours", "class_attendance", "sleep_hours", "study_sleep_ratio"]


class AddStudySleepRatio(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        X["study_sleep_ratio"] = X["study_hours"] / X["sleep_hours"]
        return X


def build_preprocessing_pipeline():
    scaler = StandardScaler()

    onehot_encoder = OneHotEncoder()

    categories = [
        ["poor", "average", "good"],
        ["low", "medium", "high"],
        ["easy", "moderate", "hard"]
        ]
    
    ordinal_encoder = OrdinalEncoder(categories=categories)

    imputer_for_num = SimpleImputer(strategy="mean")
    imputer_for_cat = SimpleImputer(strategy="most_frequent")

    ratio_adder = AddStudySleepRatio()

    numeric_pipeline = Pipeline([
        ('imputer', imputer_for_num),
        ('scaler', scaler)
    ])

    ordinal_categorical_pipeline = Pipeline([
        ('imputer', imputer_for_cat),
        ('encoder', ordinal_encoder)
    ])

    onehot_categorical_pipeline = Pipeline([
        ('imputer', imputer_for_cat),
        ('encoder', onehot_encoder)
    ])

    transformer = ColumnTransformer([
        ('cat_one_hot', onehot_categorical_pipeline, cat_features_for_onehot),
        ('cat_ordinal', ordinal_categorical_pipeline, cat_features_for_ordinal),
        ('num', numeric_pipeline, num_fetures)
    ])

    preprocessing_pipeline = Pipeline([
        ('add_ratio', ratio_adder),
        ('transform', transformer)
    ])

    return preprocessing_pipeline

def fit_preprocessor(preprocessor):
    preprocessor.fit(X_train)

    return preprocessor