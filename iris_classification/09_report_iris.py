"""Week 11: read the Iris CSV from file 08 and make a project report.

Demonstrates file input, dictionaries, loops, and writing text output.
"""
import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent
INPUT = HERE / "results" / "08_iris_predictions.csv"
OUTPUT = HERE / "results" / "09_iris_project_report.txt"


def main():
    if not INPUT.exists():
        print("First run: python iris_classification/08_evaluate_iris.py")
        return

    with INPUT.open("r", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        print("The CSV has no prediction rows.")
        return

    species = ["setosa", "versicolor", "virginica"]
    totals = {name: 0 for name in species}
    correct_by_species = {name: 0 for name in species}
    mistakes = []

    for row in rows:
        actual = row["actual"]
        predicted = row["predicted"]
        totals[actual] += 1
        if actual == predicted:
            correct_by_species[actual] += 1
        else:
            mistakes.append(f"  row {row['row_id']}: {actual} -> {predicted}")

    total_correct = sum(correct_by_species.values())
    accuracy = total_correct / len(rows)
    lines = [
        "Week 11 Iris Classification: Progress Report",
        "Research question: Can flower measurements predict Iris species?",
        "Dataset: scikit-learn Iris, 150 examples, 4 features, 3 classes.",
        "Model: Decision Tree; 80/20 stratified split; random_state=42.",
        f"Held-out test examples: {len(rows)}",
        f"Test accuracy: {accuracy:.3f} ({total_correct}/{len(rows)} correct)",
        "Results by actual species:",
    ]
    for name in species:
        lines.append(f"  {name}: {correct_by_species[name]}/{totals[name]} correct")
    lines.append("Misclassified examples:")
    lines.extend(mistakes if mistakes else ["  None in this test split."])
    lines += [
        "Evidence: results/08_iris_predictions.csv and figures/08_iris_correct_vs_incorrect.png",
        "Limitations: one split; accuracy may change with other samples or models.",
        "Next steps: compare another split or classifier; explain confusing species.",
    ]
    report = "\n".join(lines) + "\n"
    OUTPUT.parent.mkdir(exist_ok=True)
    OUTPUT.write_text(report, encoding="utf-8")
    print(report)
    print(f"Saved report: {OUTPUT}")
    print("Try it: add one sentence explaining the most common mistake.")


if __name__ == "__main__":
    main()
