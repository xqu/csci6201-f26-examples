"""Week 9: Linear Regression baseline on the Diabetes dataset."""
from pathlib import Path
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

OUTPUT = Path(__file__).resolve().parent / "results"


def train_and_evaluate():
    diabetes = load_diabetes()
    X, y = diabetes.data, diabetes.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    assert len(predictions) == len(y_test)
    assert mae >= 0
    report = (f"Diabetes Linear Regression baseline\n"
              f"Training samples: {len(y_train)}\nTest samples: {len(y_test)}\n"
              f"Mean absolute error (MAE): {mae:.2f}\nR-squared: {r2:.3f}\n"
              "Interpretation: preliminary test-set evaluation; not for clinical use.\n")
    OUTPUT.mkdir(exist_ok=True)
    (OUTPUT / "preliminary_results.txt").write_text(report, encoding="utf-8")
    print(report)
    return mae


if __name__ == "__main__":
    train_and_evaluate()
