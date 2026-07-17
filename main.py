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

from src.feature_engineering import FeatureEngineer

from src.train import ModelTrainer

from src.evaluate import ModelEvaluator
from src.visualization import Visualizer
import joblib
from config import MODEL_DIR


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

    # Feature Engineering

    engineer = FeatureEngineer()

    clean_df = engineer.load()

    featured_df = engineer.create_features(clean_df)

    engineer.save(featured_df)

    trainer = ModelTrainer()

    model, X_test, y_test, feature_names = trainer.run()

    evaluator = ModelEvaluator()

    y_test, predictions = evaluator.run()

    visualizer = Visualizer()

    visualizer.actual_vs_predicted(y_test, predictions)
    visualizer.residual_plot(y_test, predictions)
    visualizer.residual_histogram(y_test, predictions)

    model = joblib.load(MODEL_DIR/"linear_regression.pkl")
    feature_names = joblib.load(MODEL_DIR/"feature_names.pkl")

    visualizer.feature_coefficients(model, feature_names)


if __name__ == "__main__":

    main()