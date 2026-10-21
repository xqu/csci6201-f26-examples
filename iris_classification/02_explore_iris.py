"""Week 9: inspect the Iris data using reusable functions."""
import numpy as np
from sklearn.datasets import load_iris


def summarize_column(values):
    """Return basic descriptive statistics for one numeric feature."""
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        raise ValueError("Cannot summarize an empty column")
    return {"min": float(values.min()), "max": float(values.max()),
            "mean": float(values.mean())}


def count_values_above(values, threshold):
    """Count data values strictly greater than threshold (practice loops)."""
    count = 0
    for value in values:
        if value > threshold:
            count += 1
    return count


def main():
    iris = load_iris()
    X, y = iris.data, iris.target
    print("Iris samples/features:", X.shape)
    print("Feature names:", iris.feature_names)
    print("Missing numeric entries:", int(np.isnan(X).sum()))
    print("First label:", iris.target_names[y[0]])
    for index, name in enumerate(iris.target_names):
        print(f"Count of {name}:", int(np.sum(y == index)))
    petal_length = X[:, 2]
    print("Petal-length statistics:", summarize_column(petal_length))
    print("Petal length > 5 cm:", count_values_above(petal_length, 5))
    assert X.shape == (150, 4)
    assert len(y) == 150


if __name__ == "__main__":
    main()
