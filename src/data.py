"""Load the PROMISE defect datasets.

Each CSV file is one release of one Java project. Every row is one Java class,
described by 20 static code metrics (size, complexity, coupling, cohesion...).
The "bug" column counts the bugs found in that class after the release.
We turn it into a yes/no label: defective (bug > 0) or clean (bug == 0).
"""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# The 20 metrics used as input features (the "name" column is just the class name).
FEATURES = [
    "wmc", "dit", "noc", "cbo", "rfc", "lcom", "ca", "ce", "npm", "lcom3",
    "loc", "dam", "moa", "mfa", "cam", "ic", "cbm", "amc", "max_cc", "avg_cc",
]


def version_key(version: str):
    """Sort versions like 1.4 < 1.4.4 < 1.10 correctly."""
    return tuple(int(part) for part in version.split("."))


def list_releases():
    """Return {project: [versions in order]} for every CSV in data/."""
    releases = {}
    for csv in DATA_DIR.glob("*/*.csv"):
        project, version = csv.stem.split("-", 1)
        releases.setdefault(project, []).append(version)
    return {p: sorted(v, key=version_key) for p, v in sorted(releases.items())}


def load_release(project: str, version: str):
    """Return (X, y) for one release: X = metrics table, y = 1 if defective."""
    df = pd.read_csv(DATA_DIR / project / f"{project}-{version}.csv")
    X = df[FEATURES].astype(float)
    y = (df["bug"] > 0).astype(int)
    return X, y


def dataset_summary():
    """One row per release: number of classes and share of defective ones."""
    rows = []
    for project, versions in list_releases().items():
        for version in versions:
            X, y = load_release(project, version)
            rows.append({
                "project": project,
                "version": version,
                "classes": len(y),
                "defective": int(y.sum()),
                "defect_rate": round(y.mean(), 3),
            })
    return pd.DataFrame(rows)
