"""Week 10: searching and sorting numerical target values."""
from sklearn.datasets import load_diabetes


def main():
    diabetes = load_diabetes()
    targets = diabetes.target
    threshold = 200

    # Linear search: examine each target value once.
    matches = []
    for index, value in enumerate(targets):
        if value > threshold:
            matches.append(index)

    sorted_targets = sorted(float(value) for value in targets)
    print(f"Records with target > {threshold}: {len(matches)}")
    print("First five matching row numbers:", matches[:5])
    print("Five lowest targets:", sorted_targets[:5])
    print("Five highest targets:", sorted_targets[-5:])
    print(f"Minimum: {min(targets):.0f}; maximum: {max(targets):.0f}")
    print("These are dataset target values, not clinical recommendations.")
    # Try it: change threshold to 150. How does the match count change?


if __name__ == "__main__":
    main()
