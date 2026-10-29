"""Week 10: revisit predictions from the Week 9 Decision Tree baseline."""
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


def main():
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42, stratify=iris.target
    )
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print("Five held-out Iris examples (not training examples):")
    for actual, predicted in zip(y_test[:5], predictions[:5]):
        actual_name = iris.target_names[actual]
        predicted_name = iris.target_names[predicted]
        status = "correct" if actual == predicted else "incorrect"
        print(f"Actual: {actual_name:10s} Predicted: {predicted_name:10s} ({status})")
    print(f"Test accuracy: {accuracy_score(y_test, predictions):.3f}")
    print("One test split is a starting point, not a guarantee of future performance.")
    # Try it: change 5 to 10 above to view more predictions.


if __name__ == "__main__":
    main()
