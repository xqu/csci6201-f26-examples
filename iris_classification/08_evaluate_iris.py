"""Week 11: inspect Iris prediction errors and save a CSV + figure.

Run after 04_classify_iris.py, or by itself. Uses the same model/split.
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Save figures without requiring a display window.
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

HERE = Path(__file__).resolve().parent


def main():
    iris = load_iris()
    # Keep row numbers so we can identify examples in the original dataset.
    row_ids = list(range(len(iris.target)))
    X_train, X_test, y_train, y_test, _, test_ids = train_test_split(
        iris.data, iris.target, row_ids,
        test_size=0.2, random_state=42, stratify=iris.target
    )
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    output = HERE / "results"
    figures = HERE / "figures"
    output.mkdir(exist_ok=True)
    figures.mkdir(exist_ok=True)

    csv_path = output / "08_iris_predictions.csv"
    errors = 0
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["row_id", "actual", "predicted", "correct"])
        for row_id, actual, predicted in zip(test_ids, y_test, predictions):
            correct = int(actual == predicted)
            if not correct:
                errors += 1
            writer.writerow([
                row_id, iris.target_names[actual],
                iris.target_names[predicted], correct
            ])

    accuracy = accuracy_score(y_test, predictions)
    print(f"Iris held-out test examples: {len(y_test)}")
    print(f"Correct: {len(y_test) - errors}; incorrect: {errors}")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Saved CSV: {csv_path}")

    # A small visual summary; notice that counts alone hide WHICH species failed.
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["Correct", "Incorrect"], [len(y_test) - errors, errors])
    ax.set_ylabel("Number of test examples")
    ax.set_title("Iris Decision Tree: held-out predictions")
    fig.tight_layout()
    chart_path = figures / "08_iris_correct_vs_incorrect.png"
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)
    print(f"Saved figure: {chart_path}")
    print("Try it: open the CSV and locate rows where correct equals 0.")
    print("Remember: one 80/20 split does not establish model stability.")


if __name__ == "__main__":
    main()
