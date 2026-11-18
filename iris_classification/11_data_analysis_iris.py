"""Week 13: explore the Iris dataset before interpreting ML results.

Run from the repository root:
    python iris_classification/11_data_analysis_iris.py

Requires pandas, numpy, matplotlib, and scikit-learn.
Outputs a CSV summary and four PNG figures. No model is trained here.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Save PNGs on computers without a display.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris


HERE = Path(__file__).resolve().parent
FIGURES = HERE / "figures"
RESULTS = HERE / "results"


def load_dataframe():
    """Give the four measurements readable column names and species labels."""
    iris = load_iris()
    columns = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    table = pd.DataFrame(iris.data, columns=columns)
    table["species"] = [iris.target_names[label] for label in iris.target]
    return table, columns


def make_summary(table, features):
    """Summarize all flowers and each species; keep units in centimeters."""
    records = []
    for group, subset in [("all", table), *list(table.groupby("species", sort=True))]:
        for feature in features:
            values = subset[feature]
            records.append({
                "group": group,
                "feature": feature,
                "count": len(values),
                "mean": values.mean(),
                "median": values.median(),
                "std": values.std(),  # Sample standard deviation (ddof=1).
                "min": values.min(),
                "max": values.max(),
            })
    return pd.DataFrame(records)


def save_distribution(table):
    """Question 1: Do different species have different petal lengths?"""
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, group in table.groupby("species", sort=True):
        ax.hist(group["petal_length"], bins=np.arange(1, 7.6, 0.4),
                alpha=0.55, label=name, edgecolor="white")
    ax.set(xlabel="Petal length (cm)", ylabel="Number of flowers",
           title="Iris petal-length distributions by species")
    ax.legend(title="Species")
    fig.tight_layout()
    fig.savefig(FIGURES / "11_iris_petal_histograms.png", dpi=180)
    plt.close(fig)


def save_feature_boxplots(table, features):
    """Question 2: Which measurements vary most within each species?"""
    names = sorted(table["species"].unique())
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, feature in zip(axes.flat, features):
        groups = [table.loc[table["species"] == name, feature] for name in names]
        ax.boxplot(groups, tick_labels=names)
        ax.set(title=feature.replace("_", " ").title(), ylabel="Centimeters")
    fig.suptitle("Iris features: spread and overlap across species")
    fig.tight_layout()
    fig.savefig(FIGURES / "11_iris_feature_boxplots.png", dpi=180)
    plt.close(fig)


def save_correlation_heatmap(table, features):
    """Question 3: Which numeric measurements change together?"""
    # Correlation is not causation; species labels are NOT numeric features.
    correlations = table[features].corr()
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(correlations, cmap="coolwarm", vmin=-1, vmax=1)
    fig.colorbar(im, ax=ax, label="Pearson correlation")
    short_names = [name.replace("_", "\n") for name in features]
    ax.set_xticks(range(len(features)), labels=short_names)
    ax.set_yticks(range(len(features)), labels=short_names)
    for row in range(len(features)):
        for col in range(len(features)):
            ax.text(col, row, f"{correlations.iloc[row, col]:.2f}",
                    ha="center", va="center", fontsize=10)
    ax.set_title("Iris feature correlations (all species combined)")
    fig.tight_layout()
    fig.savefig(FIGURES / "11_iris_correlation_heatmap.png", dpi=180)
    plt.close(fig)


def save_group_means(table, features):
    """Question 4: How do average feature values compare across species?"""
    means = table.groupby("species", sort=True)[features].mean()
    x = np.arange(len(features))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))
    for position, (species, values) in enumerate(means.iterrows()):
        ax.bar(x + (position - 1) * width, values.to_numpy(), width,
               label=species)
    ax.set_xticks(x, labels=[f.replace("_", "\n") for f in features])
    ax.set(ylabel="Mean measurement (cm)",
           title="Average Iris measurements by species")
    ax.legend(title="Species")
    fig.tight_layout()
    fig.savefig(FIGURES / "11_iris_species_means.png", dpi=180)
    plt.close(fig)


def main():
    FIGURES.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    table, features = load_dataframe()
    print("Dataset shape (rows, columns including species):", table.shape)
    print("Missing entries by column:\n", table.isna().sum().to_string())
    print("Flower counts by species:\n", table["species"].value_counts().to_string())

    summary = make_summary(table, features)
    path = RESULTS / "11_iris_descriptive_statistics.csv"
    summary.to_csv(path, index=False, float_format="%.4f")
    print("\nPetal length statistics (cm):")
    print(summary.loc[summary["feature"] == "petal_length",
                      ["group", "count", "mean", "median", "std"]]
          .round(3).to_string(index=False))

    save_distribution(table)
    save_feature_boxplots(table, features)
    save_correlation_heatmap(table, features)
    save_group_means(table, features)
    print(f"\nSaved summary: {path}")
    print(f"Saved four Week 13 figures in: {FIGURES}")
    print("Interpretation: Look for separation AND overlap; do not claim causation.")


if __name__ == "__main__":
    main()


# STUDENT EXERCISES (use your own data where possible):
# 1. Change the histogram to sepal_width. Do species still look distinct?
# 2. Find the feature with the largest overall standard deviation using CSV.
#    Does the answer change when you compare species separately? Explain.
# 3. Select two strongly correlated features from the heatmap. Make a scatter
#    plot colored by species; write a 2-sentence caption about what it shows
#    and what it cannot prove. Do not treat correlation as causation.
