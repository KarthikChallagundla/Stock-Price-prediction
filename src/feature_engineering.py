"""
Feature Engineering Module

Creates statistical features for stock return prediction.
"""

import pandas as pd

from config import (
    PROCESSED_DATA_DIR,
    SHORT_WINDOW,
    MEDIUM_WINDOW,
    LONG_WINDOW
)

from src.utils import setup_logger

logger = setup_logger()


class FeatureEngineer:

    def __init__(self):

        self.input_file = PROCESSED_DATA_DIR / "clean_stock.csv"
        self.output_file = PROCESSED_DATA_DIR / "featured_stock.csv"

    def load(self):

        df = pd.read_csv(
            self.input_file,
            parse_dates=["Date"]
        )

        return df

    def create_features(self, df):

        df = df.copy()

        # ------------------------
        # Daily Return
        # ------------------------
        df["Daily_Return"] = df["Close"].pct_change()

        # ------------------------
        # Moving Averages
        # ------------------------
        df["MA_5"] = df["Close"].rolling(SHORT_WINDOW).mean()

        df["MA_10"] = df["Close"].rolling(MEDIUM_WINDOW).mean()

        df["MA_20"] = df["Close"].rolling(LONG_WINDOW).mean()

        # ------------------------
        # Rolling Volatility
        # ------------------------
        df["Volatility_5"] = (
            df["Daily_Return"]
            .rolling(SHORT_WINDOW)
            .std()
        )

        df["Volatility_10"] = (
            df["Daily_Return"]
            .rolling(MEDIUM_WINDOW)
            .std()
        )

        # ------------------------
        # Momentum
        # ------------------------
        df["Momentum"] = (
            df["Close"] -
            df["Close"].shift(1)
        )

        # ------------------------
        # Lagged Returns
        # ------------------------
        df["Lag_Return_1"] = df["Daily_Return"].shift(1)

        df["Lag_Return_2"] = df["Daily_Return"].shift(2)

        df["Lag_Return_3"] = df["Daily_Return"].shift(3)

        # ------------------------
        # Rolling Mean Return
        # ------------------------
        df["Rolling_Return_Mean"] = (
            df["Daily_Return"]
            .rolling(5)
            .mean()
        )

        # ------------------------
        # Rolling Std Return
        # ------------------------
        df["Rolling_Return_STD"] = (
            df["Daily_Return"]
            .rolling(5)
            .std()
        )

        # ------------------------
        # Volume Change
        # ------------------------
        df["Volume_Change"] = (
            df["Volume"]
            .pct_change()
        )

        # ------------------------
        # High Low Spread
        # ------------------------
        df["High_Low_Spread"] = (
            df["High"] -
            df["Low"]
        )

        # ------------------------
        # Open Close Difference
        # ------------------------
        df["Open_Close_Diff"] = (
            df["Open"] -
            df["Close"]
        )

        # ------------------------
        # Target
        # ------------------------
        df["Target"] = (
            df["Daily_Return"]
            .shift(-1)
        )

        # Remove rows with NaN values
        df.dropna(inplace=True)

        df.reset_index(
            drop=True,
            inplace=True
        )

        return df

    def save(self, df):

        df.to_csv(
            self.output_file,
            index=False
        )

        logger.info(
            f"Feature dataset saved to {self.output_file}"
        )

        return self.output_file


if __name__ == "__main__":

    engineer = FeatureEngineer()

    data = engineer.load()

    featured = engineer.create_features(data)

    engineer.save(featured)

    print(featured.head())