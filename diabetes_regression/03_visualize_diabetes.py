"""Week 9: create two visualizations relevant to regression."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes

OUTPUT = Path(__file__).resolve().parent / "figures"


def create_plots():
    diabetes = load_diabetes()
    X, y = diabetes.data, diabetes.target
    OUTPUT.mkdir(exist_ok=True)

    # Question: What is the distribution of the progression target?
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(y, bins=18, edgecolor="black")
    ax.set(title="Distribution of diabetes progression target",
           xlabel="One-year disease progression target", ylabel="Number of observations")
    fig.tight_layout()
    fig.savefig(OUTPUT / "01_target_distribution.png", dpi=160)
    plt.close(fig)

    # Question: Is standardized BMI associated with target values?
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(X[:, 2], y, alpha=0.65)
    ax.set(title="Standardized BMI feature versus progression target",
           xlabel="BMI feature (standardized)", ylabel="One-year disease progression target")
    fig.tight_layout()
    fig.savefig(OUTPUT / "02_bmi_vs_target.png", dpi=160)
    plt.close(fig)
    print("Saved two Diabetes figures to:", OUTPUT)


if __name__ == "__main__":
    create_plots()
