"""Experiment 2: cross-project defect prediction (CPDP).

Problem: a new project has no bug history, so it cannot train its own model.
Idea: train on other projects that do have history.

For each target project we predict its latest release using three training sets:
  1. within    : the target's own previous release (the normal case, for comparison)
  2. all-other : the latest release of ALL other projects, pooled together
  3. similar-3 : only the 3 other projects whose code metrics look most like the
                 target's (no target labels are used to pick them)

Question: can similar projects replace the target's own history?
"""

from pathlib import Path

import numpy as np
import pandas as pd

from data import list_releases, load_release
from models import build_models, evaluate, log_transform

RESULTS = Path(__file__).resolve().parent.parent / "results"
K_SIMILAR = 3


def profile(X):
    """Describe a release by the median and spread of each (log) metric.
    Uses only the metrics, never the bug labels."""
    logged = log_transform(X.values)
    return np.concatenate([np.median(logged, axis=0), logged.std(axis=0)])


def most_similar(target, candidates):
    """Return the K candidate projects whose profile is closest to the target."""
    profiles = {p: profile(X) for p, (X, _) in candidates.items()}
    all_profiles = np.vstack(list(profiles.values()) + [profile(target)])
    scale = all_profiles.std(axis=0) + 1e-9  # put every metric on the same scale
    t = profile(target) / scale
    distance = {p: np.linalg.norm(v / scale - t) for p, v in profiles.items()}
    return sorted(distance, key=distance.get)[:K_SIMILAR]


def pool(datasets):
    X = pd.concat([d[0] for d in datasets], ignore_index=True)
    y = pd.concat([d[1] for d in datasets], ignore_index=True)
    return X, y


def run():
    releases = list_releases()
    latest = {p: load_release(p, v[-1]) for p, v in releases.items()}
    rows, chosen = [], []

    for target, versions in releases.items():
        X_test, y_test = latest[target]
        others = {p: d for p, d in latest.items() if p != target}
        similar = most_similar(X_test, others)
        chosen.append({"target": f"{target}-{versions[-1]}", "similar projects": ", ".join(similar)})

        training_sets = {
            "within": load_release(target, versions[-2]),
            "all-other": pool(others.values()),
            "similar-3": pool([others[p] for p in similar]),
        }
        for setup, (X_train, y_train) in training_sets.items():
            for name, model in build_models().items():
                model.fit(X_train, y_train)
                prob = model.predict_proba(X_test)[:, 1]
                rows.append({
                    "target": f"{target}-{versions[-1]}",
                    "setup": setup,
                    "model": name,
                    **evaluate(y_test, prob),
                })

    results = pd.DataFrame(rows)
    RESULTS.mkdir(exist_ok=True)
    results.to_csv(RESULTS / "cross_project.csv", index=False)
    pd.DataFrame(chosen).to_csv(RESULTS / "similar_projects.csv", index=False)
    return results, pd.DataFrame(chosen)


if __name__ == "__main__":
    results, chosen = run()
    print("Projects picked as most similar to each target:")
    print(chosen.to_string(index=False), "\n")
    print("Mean score per training setup and model (cross-project):")
    table = results.groupby(["setup", "model"])[["AUC", "Recall", "F1", "MCC"]].mean()
    print(table.round(3))
