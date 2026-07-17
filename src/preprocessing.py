"""
Preprocessing stock data.
"""

import pandas as pd

from config import PROCESSED_DATA_DIR
from src.utils import setup_logger

logger = setup_logger()


class Preprocessor:

    def __init__(self, filepath):

        self.filepath = filepath

    def load(self):

        df = pd.read_csv(
            self.filepath,
            parse_dates=["Date"]
        )

        return df

    def clean(self, df):

        df = df.copy()

        df.drop_duplicates(inplace=True)

        df.sort_values(
            "Date",
            inplace=True
        )

        df.ffill(inplace=True)

        df.bfill(inplace=True)

        df.reset_index(
            drop=True,
            inplace=True
        )

        return df

    def save(self, df):

        path = PROCESSED_DATA_DIR / "clean_stock.csv"

        df.to_csv(path, index=False)

        logger.info(f"Saved clean data to {path}")

        return path