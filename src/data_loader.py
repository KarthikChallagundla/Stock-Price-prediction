"""
Download stock data from Yahoo Finance.
"""

import yfinance as yf

from config import (
    TICKER,
    START_DATE,
    END_DATE,
    RAW_DATA_DIR
)

from src.utils import setup_logger

logger = setup_logger()


class StockDownloader:

    def __init__(self):

        self.ticker = TICKER

    def download(self):

        logger.info(f"Downloading {self.ticker}")

        df = yf.download(
            self.ticker,
            start=START_DATE,
            end=END_DATE,
            progress=False,
            auto_adjust=True,
            multi_level_index=False   # <-- Add this
        )

        df.index.name = "Date"

        return df

    def save(self, df):

        file_path = RAW_DATA_DIR / f"{self.ticker}.csv"

        df.to_csv(file_path)

        logger.info(f"Saved data to {file_path}")

        return file_path