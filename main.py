"""
Main execution script.
"""

from config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    MODEL_DIR,
    FIGURE_DIR,
    METRIC_DIR
)

from src.utils import create_directories

from src.data_loader import StockDownloader

from src.preprocessing import Preprocessor


def main():

    create_directories([
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        MODEL_DIR,
        FIGURE_DIR,
        METRIC_DIR
    ])

    downloader = StockDownloader()

    df = downloader.download()

    raw_file = downloader.save(df)

    processor = Preprocessor(raw_file)

    data = processor.load()

    clean = processor.clean(data)

    processor.save(clean)

    print("\nProject initialized successfully.")


if __name__ == "__main__":

    main()