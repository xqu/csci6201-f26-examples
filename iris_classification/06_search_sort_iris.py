"""Week 10: searching and sorting flower measurements."""
from sklearn.datasets import load_iris


def main():
    iris = load_iris()
    petal_lengths = iris.data[:, 2]
    threshold = 5.0

    # Linear search: check each flower one at a time.
    matches = []
    for index, length in enumerate(petal_lengths):
        if length > threshold:
            matches.append(index)

    sorted_lengths = sorted(float(value) for value in petal_lengths)
    print(f"Flowers with petal length > {threshold} cm: {len(matches)}")
    print("First five matching row numbers:", matches[:5])
    print("Five shortest petal lengths:", sorted_lengths[:5])
    print("Five longest petal lengths:", sorted_lengths[-5:])
    print(f"Minimum: {min(petal_lengths):.1f} cm; maximum: {max(petal_lengths):.1f} cm")
    # Try it: change threshold to 4.0. How does the match count change?


if __name__ == "__main__":
    main()
