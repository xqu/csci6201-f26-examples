"""Week 11 self-study: save Diabetes prediction errors to CSV + a figure.

Educational dataset only; do not use these predictions for clinical decisions.
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

HERE = Path(__file__).resolve().parent


def main():
    diabetes = load_diabetes()
    row_ids = list(range(len(diabetes.target)))
    X_train, X_test, y_train, y_test, _, test_ids = train_test_split(
        diabetes.data, diabetes.target, row_ids,
        test_size=0.2, random_state=42
    )
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    errors = y_test - predictions  # Residual = actual minus predicted.

    output = HERE / "results"
    figures = HERE / "figures"
    output.mkdir(exist_ok=True)
    figures.mkdir(exist_ok=True)

    csv_path = output / "08_diabetes_predictions.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["row_id", "actual", "predicted", "residual", "absolute_error"])
        for row_id, actual, predicted, residual in zip(test_ids, y_test, predictions, errors):
            writer.writerow([
                row_id, f"{actual:.6f}", f"{predicted:.6f}",
                f"{residual:.6f}", f"{abs(residual):.6f}"
            ])

    print(f"Diabetes held-out test examples: {len(y_test)}")
    print(f"Mean absolute error (MAE): {mean_absolute_error(y_test, predictions):.2f}")
    print(f"R-squared: {r2_score(y_test, predictions):.3f}")
    print(f"Saved CSV: {csv_path}")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(predictions, errors, alpha=0.7)
    ax.axhline(0, linestyle="--", color="black", linewidth=1)
    ax.set_xlabel("Predicted target")
    ax.set_ylabel("Residual (actual - predicted)")
    ax.set_title("Diabetes Linear Regression: residuals")
    fig.tight_layout()
    chart_path = figures / "08_diabetes_residuals.png"
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)
    print(f"Saved figure: {chart_path}")
    print("Try it: which test examples have the largest absolute errors?")
    print("This is a programming demonstration, not a medical tool.")


if __name__ == "__main__":
    main()
