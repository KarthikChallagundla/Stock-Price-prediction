"""
Train Linear Regression model for stock return prediction.
"""

import joblib
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import (
    PROCESSED_DATA_DIR,
    MODEL_DIR,
    TEST_SIZE
)

from src.utils import setup_logger

logger = setup_logger()


class ModelTrainer:

    def __init__(self):

        self.input_file = PROCESSED_DATA_DIR / "featured_stock.csv"

    def load(self):

        return pd.read_csv(self.input_file)

    def prepare_data(self, df):

        # Remove Date column
        X = df.drop(columns=["Date", "Target"])

        y = df["Target"]

        return X, y

    def split_data(self, X, y):

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            shuffle=False
        )

        return X_train, X_test, y_train, y_test

    def scale_data(self, X_train, X_test):

        scaler = StandardScaler()

        X_train_scaled = scaler.fit_transform(X_train)

        X_test_scaled = scaler.transform(X_test)

        return scaler, X_train_scaled, X_test_scaled

    def train(self, X_train, y_train):

        model = LinearRegression()

        model.fit(X_train, y_train)

        return model

    def save(self, model, scaler, feature_names):

        MODEL_DIR.mkdir(parents=True, exist_ok=True)

        joblib.dump(
            model,
            MODEL_DIR / "linear_regression.pkl"
        )

        joblib.dump(
            scaler,
            MODEL_DIR / "scaler.pkl"
        )

        joblib.dump(
            feature_names,
            MODEL_DIR / "feature_names.pkl"
        )

        logger.info("Model saved successfully.")

    def run(self):

        df = self.load()

        X, y = self.prepare_data(df)

        X_train, X_test, y_train, y_test = self.split_data(X, y)

        scaler, X_train, X_test = self.scale_data(
            X_train,
            X_test
        )

        model = self.train(
            X_train,
            y_train
        )

        self.save(
            model,
            scaler,
            list(X.columns)
        )

        return (
            model,
            X_test,
            y_test,
            X.columns
        )