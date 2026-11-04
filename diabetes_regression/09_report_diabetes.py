"""Week 11 self-study: read file 08 CSV and write a regression report.

Practices reading files, numeric conversion, loops, and report writing.
"""
import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent
INPUT = HERE / "results" / "08_diabetes_predictions.csv"
OUTPUT = HERE / "results" / "09_diabetes_project_report.txt"


def main():
    if not INPUT.exists():
        print("First run: python diabetes_regression/08_evaluate_diabetes.py")
        return

    with INPUT.open("r", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        print("The CSV has no prediction rows.")
        return

    actual = [float(row["actual"]) for row in rows]
    predicted = [float(row["predicted"]) for row in rows]
    absolute_errors = [abs(a - p) for a, p in zip(actual, predicted)]
    mae = sum(absolute_errors) / len(rows)
    actual_mean = sum(actual) / len(actual)
    sum_squared_errors = sum((a - p) ** 2 for a, p in zip(actual, predicted))
    total_variation = sum((a - actual_mean) ** 2 for a in actual)
    r_squared = 1 - sum_squared_errors / total_variation if total_variation else float("nan")

    largest = max(range(len(rows)), key=lambda i: absolute_errors[i])
    lines = [
        "Week 11 Diabetes Regression: Progress Report",
        "Research question: Can baseline measurements predict the dataset target?",
        "Dataset: scikit-learn Diabetes, 442 examples, 10 preprocessed features.",
        "Model: Linear Regression; 80/20 split; random_state=42.",
        f"Held-out test examples: {len(rows)}",
        f"Mean absolute error (MAE): {mae:.2f}",
        f"R-squared: {r_squared:.3f}",
        f"Largest absolute error: {absolute_errors[largest]:.2f} (row {rows[largest]['row_id']})",
        "Evidence: results/08_diabetes_predictions.csv and figures/08_diabetes_residuals.png",
        "Interpretation: residuals show where the model over- and under-predicts.",
        "Limitations: one test split; prediction errors can be substantial.",
        "Next steps: inspect high-error rows and compare a simple baseline.",
        "For education only; not for diagnosis or clinical decisions.",
    ]
    report = "\n".join(lines) + "\n"
    OUTPUT.parent.mkdir(exist_ok=True)
    OUTPUT.write_text(report, encoding="utf-8")
    print(report)
    print(f"Saved report: {OUTPUT}")
    print("Try it: explain whether the MAE seems large relative to the target values.")


if __name__ == "__main__":
    main()
