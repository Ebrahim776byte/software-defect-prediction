"""The classifiers we compare, and how we score them."""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, matthews_corrcoef, recall_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

SEED = 42


def log_transform(X):
    """Code metrics are very skewed (a few huge classes), so we use log(1 + x)."""
    return np.log1p(np.clip(X, 0, None))


def build_models():
    """Return {name: model}. 'balanced' versions give more weight to the
    rare defective classes, which helps when only a few classes have bugs."""
    def logistic(balanced):
        return make_pipeline(
            FunctionTransformer(log_transform),
            StandardScaler(),
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced" if balanced else None,
            ),
        )

    def forest(balanced):
        return RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced" if balanced else None,
            random_state=SEED,
            n_jobs=-1,
        )

    return {
        "LogReg": logistic(balanced=False),
        "LogReg (balanced)": logistic(balanced=True),
        "RandomForest": forest(balanced=False),
        "RandomForest (balanced)": forest(balanced=True),
    }


def evaluate(y_true, y_prob, threshold=0.5):
    """Score predictions.

    AUC    : how well the model ranks defective classes above clean ones (0.5 = random).
    Recall : share of the real defective classes that the model caught.
    F1     : balance between catching defects and not raising false alarms.
    MCC    : correlation between prediction and truth (-1..1, 0 = random),
             fair even when defective classes are rare.
    """
    y_pred = (y_prob >= threshold).astype(int)
    auc = roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else float("nan")
    return {
        "AUC": auc,
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }
