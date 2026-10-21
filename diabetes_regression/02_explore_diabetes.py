"""Week 9: explore Diabetes regression data with simple functions."""
import numpy as np
from sklearn.datasets import load_diabetes


def summarize_column(values):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        raise ValueError("Cannot summarize an empty column")
    return {"min": float(values.min()), "max": float(values.max()),
            "mean": float(values.mean())}


def count_above(values, threshold):
    count = 0
    for value in values:
        if value > threshold:
            count += 1
    return count


def main():
    diabetes = load_diabetes()
    X, y = diabetes.data, diabetes.target
    print("Diabetes samples/features:", X.shape)
    print("Feature names:", diabetes.feature_names)
    print("Missing feature values:", int(np.isnan(X).sum()))
    print("Missing target values:", int(np.isnan(y).sum()))
    print("Target statistics:", summarize_column(y))
    print("Target values over 200:", count_above(y, 200))
    print("BMI feature statistics:", summarize_column(X[:, 2]))
    assert X.shape == (442, 10)
    assert len(y) == 442


if __name__ == "__main__":
    main()
