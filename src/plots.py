"""Draw the figures used in the README from the CSVs in results/."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from data import dataset_summary

RESULTS = Path(__file__).resolve().parent.parent / "results"
FIGURES = RESULTS / "figures"


def defect_rates():
    summary = dataset_summary()
    summary.to_csv(RESULTS / "dataset_summary.csv", index=False)
    labels = summary["project"] + "-" + summary["version"]
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(labels, summary["defect_rate"] * 100, color="#4C72B0")
    ax.set_ylabel("Defective classes (%)")
    ax.set_title("Share of defective classes in each release")
    ax.tick_params(axis="x", rotation=90, labelsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "defect_rates.png", dpi=150)
    plt.close(fig)


def within_project():
    results = pd.read_csv(RESULTS / "within_project.csv")
    order = ["LogReg", "LogReg (balanced)", "RandomForest", "RandomForest (balanced)"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric in zip(axes, ["AUC", "MCC"]):
        data = [results.loc[results["model"] == m, metric].dropna() for m in order]
        ax.boxplot(data)
        ax.set_xticks(range(1, len(order) + 1), [m.replace(" (", "\n(") for m in order], fontsize=8)
        ax.set_title(f"Within-project {metric}")
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "within_project.png", dpi=150)
    plt.close(fig)


def cross_project():
    results = pd.read_csv(RESULTS / "cross_project.csv")
    model = "RandomForest (balanced)"
    sub = results[results["model"] == model]
    table = sub.pivot(index="target", columns="setup", values="AUC")[["within", "all-other", "similar-3"]]
    ax = table.plot.bar(figsize=(11, 4), color=["#4C72B0", "#DD8452", "#55A868"])
    ax.set_ylabel("AUC")
    ax.set_ylim(0.4, 1.0)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    ax.set_title(f"Cross-project prediction of each project's latest release ({model})")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(title="Trained on")
    plt.tight_layout()
    plt.savefig(FIGURES / "cross_project.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    FIGURES.mkdir(parents=True, exist_ok=True)
    defect_rates()
    within_project()
    cross_project()
    print(f"Figures saved in {FIGURES}")
