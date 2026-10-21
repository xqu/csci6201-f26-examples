"""Week 9: create two Iris figures matched to research questions."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # Save images even without a GUI window
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

OUTPUT = Path(__file__).resolve().parent / "figures"


def create_plots():
    iris = load_iris()
    X, y = iris.data, iris.target
    OUTPUT.mkdir(exist_ok=True)

    # Research question: How do petal lengths differ by species?
    fig, ax = plt.subplots(figsize=(8, 5))
    grouped = [X[y == i, 2] for i in range(3)]
    ax.boxplot(grouped, tick_labels=iris.target_names)
    ax.set(title="Iris petal length by species", ylabel="Petal length (cm)")
    fig.tight_layout()
    fig.savefig(OUTPUT / "01_petal_length_by_species.png", dpi=160)
    plt.close(fig)

    # Research question: Can petal measurements separate species?
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, species in enumerate(iris.target_names):
        ax.scatter(X[y == i, 2], X[y == i, 3], label=species, alpha=0.75)
    ax.set(title="Iris: petal length versus petal width",
           xlabel="Petal length (cm)", ylabel="Petal width (cm)")
    ax.legend(title="Species")
    fig.tight_layout()
    fig.savefig(OUTPUT / "02_petal_length_vs_width.png", dpi=160)
    plt.close(fig)
    print("Saved two Iris figures to:", OUTPUT)


if __name__ == "__main__":
    create_plots()
