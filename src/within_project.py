"""Experiment 1: within-project defect prediction.

Realistic setup: a team trains a model on the previous release of their own
project, then uses it to predict which classes of the next release are defective.
For every project we make pairs of consecutive releases (1.3 -> 1.4, 1.4 -> 1.5, ...).
"""

from pathlib import Path

import pandas as pd

from data import list_releases, load_release
from models import build_models, evaluate

RESULTS = Path(__file__).resolve().parent.parent / "results"


def run():
    rows = []
    for project, versions in list_releases().items():
        for train_v, test_v in zip(versions[:-1], versions[1:]):
            X_train, y_train = load_release(project, train_v)
            X_test, y_test = load_release(project, test_v)
            for name, model in build_models().items():
                model.fit(X_train, y_train)
                prob = model.predict_proba(X_test)[:, 1]
                rows.append({
                    "project": project,
                    "train": train_v,
                    "test": test_v,
                    "model": name,
                    **evaluate(y_test, prob),
                })
    results = pd.DataFrame(rows)
    RESULTS.mkdir(exist_ok=True)
    results.to_csv(RESULTS / "within_project.csv", index=False)
    return results


if __name__ == "__main__":
    results = run()
    print(f"{results[['project', 'train', 'test']].drop_duplicates().shape[0]} release pairs\n")
    print("Mean score per model (within-project):")
    print(results.groupby("model")[["AUC", "Recall", "F1", "MCC"]].mean().round(3))
