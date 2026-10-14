
# CSCI 6201 - Fall 2026
# Week 8: Introduction to the Diabetes Dataset
# Example 2: Regression

from sklearn.datasets import load_diabetes


def main():
    # Step 1: Load the dataset
    diabetes = load_diabetes()

    # Step 2: Explore the dataset
    print("Number of samples:", len(diabetes.data))
    print("Number of features:", len(diabetes.feature_names))

    print("\nFeature names:")
    print(diabetes.feature_names)

    # Step 3: Examine the first sample
    print("\nFirst sample measurements:")
    print(diabetes.data[0])

    print("\nFirst target value:")
    print(diabetes.target[0])

    # Step 4: Understand the target
    print("\nTarget minimum:", diabetes.target.min())
    print("Target maximum:", diabetes.target.max())


if __name__ == "__main__":
    main()
