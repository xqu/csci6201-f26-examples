"""Week 12: turn the Iris class example into reproducible paper evidence.

Run from the repository root:
    python iris_classification/10_paper_results_iris.py

Creates one figure and two small results files for an educational paper draft.
No new research contribution is claimed.
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

HERE = Path(__file__).resolve().parent


def main():
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.20, random_state=42,
        stratify=iris.target
    )

    # The Decision Tree matches files 04 and 08. The dummy baseline always
    # predicts the most common class in the TRAINING data, not the test data.
    models = [
        ("Majority-class baseline", DummyClassifier(strategy="most_frequent")),
        ("Decision Tree", DecisionTreeClassifier(random_state=42)),
    ]
    scores = []
    tree_predictions = None
    for name, model in models:
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        scores.append((name, accuracy))
        if name == "Decision Tree":
            tree_predictions = predictions

    results_dir = HERE / "results"
    figures_dir = HERE / "figures"
    results_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)

    # A compact CSV becomes the source of the paper's results table.
    csv_path = results_dir / "10_iris_paper_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "train_samples", "test_samples", "test_accuracy"])
        for name, accuracy in scores:
            writer.writerow([name, len(y_train), len(y_test), f"{accuracy:.4f}"])

    # Confusion matrix: rows = true species; columns = predicted species.
    matrix = confusion_matrix(y_test, tree_predictions, labels=[0, 1, 2])
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix, cmap="Blues", vmin=0)
    fig.colorbar(image, ax=ax, label="Number of test examples")
    species = list(iris.target_names)
    ax.set_xticks(range(3), labels=species)
    ax.set_yticks(range(3), labels=species)
    ax.set_xlabel("Predicted species")
    ax.set_ylabel("Actual species")
    ax.set_title("Iris Decision Tree: test-set confusion matrix")
    for row in range(3):
        for col in range(3):
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center",
                    color="white" if matrix[row, col] > matrix.max() / 2 else "black")
    fig.tight_layout()
    figure_path = figures_dir / "10_iris_confusion_matrix.png"
    fig.savefig(figure_path, dpi=200)
    plt.close(fig)

    # The text can be consulted while drafting, but should not be copied blindly.
    txt_path = results_dir / "10_iris_paper_notes.txt"
    lines = [
        "WEEK 12 IRIS: EDUCATIONAL PAPER NOTES (NOT A PUBLISHED PAPER)",
        "Dataset: scikit-learn Iris; 150 flowers, 4 measurements, 3 species.",
        "Task: predict species (classification).",
        "Split: stratified 80% train / 20% test; random_state=42.",
        f"Train/test counts: {len(y_train)}/{len(y_test)}.",
        "Compared methods: majority-class baseline and Decision Tree (random_state=42).",
        *[f"Test accuracy, {name}: {accuracy:.4f}" for name, accuracy in scores],
        "Confusion matrix (rows=actual, columns=predicted; setosa, versicolor, virginica):",
        *[str(row.tolist()) for row in matrix],
        "Results table source: results/10_iris_paper_metrics.csv",
        "Figure source: figures/10_iris_confusion_matrix.png",
        "Discussion prompt: Which species, if any, were confused?",
        "Limitation: One small held-out test split; results may change with a different split.",
        "Avoid claims about novelty or dependable performance on new populations.",
    ]
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nSaved: {csv_path}\nSaved: {figure_path}\nSaved: {txt_path}")


if __name__ == "__main__":
    main()
