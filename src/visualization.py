"""
Visualization module.
"""

import matplotlib.pyplot as plt

from config import FIGURE_DIR


class Visualizer:

    def __init__(self):

        FIGURE_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

    def actual_vs_predicted(
            self,
            actual,
            predicted
    ):

        plt.figure(figsize=(12,6))

        plt.plot(
            actual.values,
            label="Actual"
        )

        plt.plot(
            predicted,
            label="Predicted"
        )

        plt.legend()

        plt.title(
            "Actual vs Predicted Returns"
        )

        plt.tight_layout()

        plt.savefig(
            FIGURE_DIR /
            "actual_vs_predicted.png"
        )

        plt.close()

    def residual_plot(
            self,
            actual,
            predicted
    ):

        residuals = actual.values - predicted

        plt.figure(figsize=(8,6))

        plt.scatter(
            predicted,
            residuals,
            alpha=0.6
        )

        plt.axhline(
            0,
            linestyle="--"
        )

        plt.xlabel("Predicted")

        plt.ylabel("Residual")

        plt.title(
            "Residual Plot"
        )

        plt.tight_layout()

        plt.savefig(
            FIGURE_DIR /
            "residual_plot.png"
        )

        plt.close()

    def residual_histogram(
            self,
            actual,
            predicted
    ):

        residuals = actual.values - predicted

        plt.figure(figsize=(8,6))

        plt.hist(
            residuals,
            bins=30
        )

        plt.title(
            "Residual Distribution"
        )

        plt.tight_layout()

        plt.savefig(
            FIGURE_DIR /
            "residual_histogram.png"
        )

        plt.close()

    def feature_coefficients(
            self,
            model,
            feature_names
    ):

        plt.figure(figsize=(10,7))

        plt.barh(
            feature_names,
            model.coef_
        )

        plt.title(
            "Linear Regression Coefficients"
        )

        plt.tight_layout()

        plt.savefig(
            FIGURE_DIR /
            "feature_importance.png"
        )

        plt.close()