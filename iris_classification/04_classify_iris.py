"""Week 9: baseline classification with a Decision Tree."""
from pathlib import Path
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

OUTPUT = Path(__file__).resolve().parent / "results"


def train_and_evaluate():
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    matrix = confusion_matrix(y_test, predictions)
    assert len(predictions) == len(y_test)
    assert 0 <= accuracy <= 1
    report = (f"Iris Decision Tree baseline\n"
              f"Training samples: {len(y_train)}\nTest samples: {len(y_test)}\n"
              f"Accuracy: {accuracy:.3f}\n"
              f"Confusion matrix (setosa, versicolor, virginica):\n{matrix}\n"
              "Interpretation: preliminary single train/test split; performance can vary.\n")
    OUTPUT.mkdir(exist_ok=True)
    (OUTPUT / "preliminary_results.txt").write_text(report, encoding="utf-8")
    print(report)
    return accuracy


if __name__ == "__main__":
    train_and_evaluate()
