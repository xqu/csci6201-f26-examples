
# CSCI 6201 - Fall 2026
# Week 8: Introduction to the Iris Dataset

from sklearn.datasets import load_iris


def main():
    # Step 1: Load the built-in Iris dataset
    iris = load_iris()

    # Step 2: Explore the dataset
    print("Number of samples:", len(iris.data))
    print("Number of features:", len(iris.feature_names))

    print("\nFeature names:")
    print(iris.feature_names)

    print("\nFlower species:")
    print(iris.target_names)

    # Step 3: Examine the first sample
    print("\nFirst flower measurements:")
    print(iris.data[0])

    print("\nFirst flower label:")
    print(iris.target[0])


if __name__ == "__main__":
    main()
