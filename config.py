"""
Configuration file for Stock Return Prediction Project
"""

from pathlib import Path

# =====================================
# Project Directories
# =====================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODEL_DIR = BASE_DIR / "models"

OUTPUT_DIR = BASE_DIR / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
METRIC_DIR = OUTPUT_DIR / "metrics"

# =====================================
# Stock Configuration
# =====================================

TICKER = "AAPL"

START_DATE = "2015-01-01"
END_DATE = "2025-01-01"

# =====================================
# Feature Engineering
# =====================================

SHORT_WINDOW = 5
MEDIUM_WINDOW = 10
LONG_WINDOW = 20

# =====================================
# Train Test Split
# =====================================

TEST_SIZE = 0.2

# =====================================
# Random Seed
# =====================================

RANDOM_STATE = 42