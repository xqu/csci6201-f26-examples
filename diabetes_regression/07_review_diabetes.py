"""Week 10: revisit errors from the Week 9 Linear Regression baseline."""
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


def main():
    diabetes = load_diabetes()
    X_train, X_test, y_train, y_test = train_test_split(
        diabetes.data, diabetes.target, test_size=0.2, random_state=42
    )
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print("Five held-out Diabetes examples (not training examples):")
    for actual, predicted in zip(y_test[:5], predictions[:5]):
        error = abs(actual - predicted)
        print(f"Actual: {actual:6.1f} Predicted: {predicted:7.1f} Absolute error: {error:6.1f}")
    print(f"Test mean absolute error (MAE): {mean_absolute_error(y_test, predictions):.2f}")
    print("Educational dataset example only; not for clinical decisions.")
    # Try it: change 5 to 10 above to examine more prediction errors.


if __name__ == "__main__":
    main()
