"""
Model evaluation module.
"""

import json
import joblib
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from config import (
    PROCESSED_DATA_DIR,
    MODEL_DIR,
    METRIC_DIR
)

from src.utils import setup_logger

logger = setup_logger()


class ModelEvaluator:

    def __init__(self):

        self.data_file = PROCESSED_DATA_DIR / "featured_stock.csv"

        self.model = joblib.load(
            MODEL_DIR / "linear_regression.pkl"
        )

        self.scaler = joblib.load(
            MODEL_DIR / "scaler.pkl"
        )

        self.feature_names = joblib.load(
            MODEL_DIR / "feature_names.pkl"
        )

    def load_test_data(self):

        df = pd.read_csv(self.data_file)

        X = df.drop(
            columns=["Date", "Target"]
        )

        y = df["Target"]

        split = int(len(df) * 0.8)

        X_test = X.iloc[split:]

        y_test = y.iloc[split:]

        return X_test, y_test

    def predict(self, X_test):

        X_scaled = self.scaler.transform(X_test)

        predictions = self.model.predict(X_scaled)

        return predictions

    def evaluate(self, y_test, predictions):

        metrics = {

            "MAE": float(
                mean_absolute_error(
                    y_test,
                    predictions
                )
            ),

            "RMSE": float(
                mean_squared_error(
                    y_test,
                    predictions
                ) ** 0.5
            ),

            "R2": float(
                r2_score(
                    y_test,
                    predictions
                )
            )

        }

        return metrics

    def save_metrics(self, metrics):

        METRIC_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        output_file = METRIC_DIR / "metrics.json"

        with open(output_file, "w") as f:

            json.dump(
                metrics,
                f,
                indent=4
            )

        logger.info(
            f"Metrics saved to {output_file}"
        )

    def run(self):

        X_test, y_test = self.load_test_data()

        predictions = self.predict(X_test)

        metrics = self.evaluate(
            y_test,
            predictions
        )

        self.save_metrics(metrics)

        print("\nModel Performance")

        for k, v in metrics.items():

            print(f"{k}: {v:.6f}")

        return (
            y_test,
            predictions
        )