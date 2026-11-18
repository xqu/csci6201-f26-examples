"""Week 13: exploratory data analysis for the scikit-learn Diabetes dataset.

Run from the repository root:
    python diabetes_regression/11_data_analysis_diabetes.py

Creates one CSV and four PNG figures using pandas and Matplotlib.
Educational dataset exploration only: NOT a clinical prediction tool.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Save plots without opening windows.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes


HERE = Path(__file__).resolve().parent
FIGURES = HERE / "figures"
RESULTS = HERE / "results"


def load_dataframe():
    """Load ten preprocessed features and the continuous progression target."""
    diabetes = load_diabetes()
    table = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    table["target"] = diabetes.target
    return table, list(diabetes.feature_names)


def make_summary(table):
    """Describe the shape and spread of each numeric column."""
    summary = table.agg(["count", "mean", "median", "std", "min", "max"]).T
    summary.index.name = "variable"
    return summary.reset_index()


def save_target_distribution(table):
    """Question 1: How much do the outcome values vary?"""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(table["target"], bins=18, color="steelblue", edgecolor="white")
    ax.axvline(table["target"].median(), color="darkred", linestyle="--",
               label=f"Median = {table['target'].median():.1f}")
    ax.set(title="Diabetes progression target: distribution and median",
           xlabel="One-year progression target (dataset units)",
           ylabel="Number of observations")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "11_diabetes_target_histogram.png", dpi=180)
    plt.close(fig)


def save_feature_distributions(table):
    """Question 2: How do selected standardized features vary?"""
    names = ["bmi", "bp", "s5", "age"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, name in zip(axes.flat, names):
        ax.hist(table[name], bins=18, edgecolor="white", color="teal")
        ax.set(title=name.upper(), xlabel="Preprocessed feature value",
               ylabel="Observations")
    fig.suptitle("Selected Diabetes input-feature distributions")
    fig.tight_layout()
    fig.savefig(FIGURES / "11_diabetes_feature_histograms.png", dpi=180)
    plt.close(fig)


def save_target_correlations(table, features):
    """Question 3: Which features are most associated with the target?"""
    # These are descriptive associations, NOT proof of causation.
    correlations = table[features].corrwith(table["target"]).sort_values()
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["tomato" if value < 0 else "steelblue"
              for value in correlations]
    ax.barh(correlations.index, correlations.values, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set(xlim=(-1, 1), xlabel="Pearson correlation with target",
           title="Diabetes features: association with progression target")
    fig.tight_layout()
    fig.savefig(FIGURES / "11_diabetes_target_correlations.png", dpi=180)
    plt.close(fig)
    print("\nFeature correlations with target (strongest absolute first):")
    print(correlations.reindex(correlations.abs().sort_values(ascending=False).index)
          .round(3).to_string())


def save_correlation_heatmap(table):
    """Question 4: Do some input features also correlate with each other?"""
    correlation = table.corr(numeric_only=True)
    labels = list(correlation.columns)
    fig, ax = plt.subplots(figsize=(9, 8))
    image = ax.imshow(correlation, vmin=-1, vmax=1, cmap="coolwarm")
    fig.colorbar(image, ax=ax, label="Pearson correlation")
    ax.set_xticks(range(len(labels)), labels=labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels=labels)
    # Eleven variables are readable with two-decimal annotations.
    for row in range(len(labels)):
        for col in range(len(labels)):
            value = correlation.iloc[row, col]
            ax.text(col, row, f"{value:.2f}", ha="center", va="center",
                    fontsize=7, color="white" if abs(value) > 0.65 else "black")
    ax.set_title("Feature and target correlation matrix")
    fig.tight_layout()
    fig.savefig(FIGURES / "11_diabetes_correlation_heatmap.png", dpi=200)
    plt.close(fig)


def main():
    FIGURES.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    table, features = load_dataframe()
    print("Samples and columns (including target):", table.shape)
    print("Missing entries by column:\n", table.isna().sum().to_string())
    print("Note: the 10 input features are preprocessed/scaled.")
    print("Their numerical values are NOT raw clinical measurements.")

    summary = make_summary(table)
    csv_path = RESULTS / "11_diabetes_descriptive_statistics.csv"
    summary.to_csv(csv_path, index=False, float_format="%.4f")
    print("\nSelected descriptive statistics:")
    print(summary.loc[summary["variable"].isin(["bmi", "bp", "target"])]
          .round(3).to_string(index=False))

    save_target_distribution(table)
    save_feature_distributions(table)
    save_target_correlations(table, features)
    save_correlation_heatmap(table)
    print(f"\nSaved summary: {csv_path}")
    print(f"Saved four Week 13 figures in: {FIGURES}")
    print("Interpretation: Correlation does not imply causation, and this")
    print("educational dataset does not validate real-world clinical decisions.")


if __name__ == "__main__":
    main()


# STUDENT EXERCISES (adapt the ideas to your own project):
# 1. Change the four selected feature histograms. Which distributions differ?
#    Explain why a scaled feature value should not be called raw BMI or BP.
# 2. Identify the most positively and negatively correlated features with
#    the target. Are their correlations strong enough to establish causation?
# 3. Add a scatter plot: x=bmi, y=target, with clear axis labels and a
#    two-sentence caption. Explain what you can and cannot conclude.
