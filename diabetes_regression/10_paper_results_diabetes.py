"""Week 12 self-study: generate evidence for a Diabetes paper draft.

Run from the repository root:
    python diabetes_regression/10_paper_results_diabetes.py

This is a programming example, not a clinical prediction tool.
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent


def main():
    diabetes = load_diabetes()
    X_train, X_test, y_train, y_test = train_test_split(
        diabetes.data, diabetes.target, test_size=0.20, random_state=42
    )

    # The Linear Regression matches files 04 and 08. The baseline predicts
    # the mean target from TRAINING examples, avoiding test-data leakage.
    models = [
        ("Training-mean baseline", DummyRegressor(strategy="mean")),
        ("Linear Regression", LinearRegression()),
    ]
    scores = []
    linear_predictions = None
    for name, model in models:
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        r_squared = r2_score(y_test, predictions)
        scores.append((name, mae, r_squared))
        if name == "Linear Regression":
            linear_predictions = predictions

    results_dir = HERE / "results"
    figures_dir = HERE / "figures"
    results_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)

    csv_path = results_dir / "10_diabetes_paper_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "train_samples", "test_samples", "test_MAE", "test_R_squared"])
        for name, mae, r_squared in scores:
            writer.writerow([name, len(y_train), len(y_test), f"{mae:.4f}", f"{r_squared:.4f}"])

    # Each dot is one held-out test example; dashed line is perfect prediction.
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_test, linear_predictions, alpha=0.7)
    lower = min(y_test.min(), linear_predictions.min())
    upper = max(y_test.max(), linear_predictions.max())
    ax.plot([lower, upper], [lower, upper], "k--", label="Perfect prediction")
    ax.set_xlabel("Actual target")
    ax.set_ylabel("Predicted target")
    ax.set_title("Diabetes Linear Regression: held-out test predictions")
    ax.legend()
    fig.tight_layout()
    figure_path = figures_dir / "10_diabetes_actual_vs_predicted.png"
    fig.savefig(figure_path, dpi=200)
    plt.close(fig)

    txt_path = results_dir / "10_diabetes_paper_notes.txt"
    lines = [
        "WEEK 12 DIABETES: EDUCATIONAL PAPER NOTES (NOT A PUBLISHED PAPER)",
        "Dataset: scikit-learn Diabetes; 442 samples, 10 preprocessed numeric features.",
        "Task: predict a continuous target (regression).",
        "Split: 80% train / 20% test; random_state=42.",
        f"Train/test counts: {len(y_train)}/{len(y_test)}.",
        "Compared methods: training-mean baseline and Linear Regression.",
        *[f"{name}: test MAE={mae:.4f}; test R-squared={r_squared:.4f}"
          for name, mae, r_squared in scores],
        "MAE: average absolute prediction error; smaller is better.",
        "R-squared: relative fit compared with a test-set mean reference; larger is better.",
        "Results table source: results/10_diabetes_paper_metrics.csv",
        "Figure source: figures/10_diabetes_actual_vs_predicted.png",
        "Discussion prompt: Where are predictions farthest from the dashed diagonal?",
        "Limitation: One test split; predictive patterns are not causal findings.",
        "For education only; not suitable for diagnosis or clinical decisions.",
    ]
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nSaved: {csv_path}\nSaved: {figure_path}\nSaved: {txt_path}")


if __name__ == "__main__":
    main()
