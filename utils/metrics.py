"""
metrics.py
----------
A single, shared `evaluate_subset` function used by the baseline
classifier notebook AND every metaheuristic optimizer (GA, PSO, GWO, WOA).

Using one evaluation function everywhere guarantees the comparison in
07_Comparison.ipynb is fair: every algorithm is scored with identical
classifiers, identical train/test splits, and identical metrics.
"""

import time
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


def get_classifier(name="svm", random_state=42):
    """Factory for the classifiers used throughout the project."""
    name = name.lower()
    if name == "svm":
        return SVC(kernel="rbf", probability=True, random_state=random_state)
    elif name == "random_forest" or name == "rf":
        return RandomForestClassifier(n_estimators=200, random_state=random_state)
    else:
        raise ValueError(f"Unknown classifier: {name}")


def _apply_mask(X_train, X_test, mask):
    """
    Select columns of X_train / X_test according to a binary feature mask.

    mask : array-like of 0/1 (or bool) values, one per feature.
    If the mask selects zero features, we fall back to using all features
    to avoid degenerate optimizer solutions (fitness will still penalize
    this via the feature-count term in fitness.py).
    """
    mask = np.asarray(mask).astype(bool)
    if mask.sum() == 0:
        mask = np.ones_like(mask, dtype=bool)
    return X_train[:, mask], X_test[:, mask], mask


def evaluate_subset(
    mask,
    X_train,
    X_test,
    y_train,
    y_test,
    classifier="svm",
    random_state=42,
):
    """
    Train `classifier` on the features selected by `mask` and compute
    standard classification metrics on the held-out test set.

    Parameters
    ----------
    mask : array-like binary vector selecting features (len == n_features)
    classifier : "svm" | "random_forest"

    Returns
    -------
    dict with keys:
        accuracy, precision, recall, f1, roc_auc,
        n_features, runtime, y_pred, mask
    """
    X_train_sel, X_test_sel, mask_bool = _apply_mask(X_train, X_test, mask)

    clf = get_classifier(classifier, random_state=random_state)

    start = time.time()
    clf.fit(X_train_sel, y_train)
    y_pred = clf.predict(X_test_sel)

    # roc_auc needs probability / decision scores
    try:
        y_score = clf.predict_proba(X_test_sel)[:, 1]
        roc_auc = roc_auc_score(y_test, y_score)
    except Exception:
        roc_auc = np.nan
    runtime = time.time() - start

    results = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc,
        "n_features": int(mask_bool.sum()),
        "runtime": runtime,
        "y_pred": y_pred,
        "mask": mask_bool,
    }
    return results


def get_confusion_matrix(y_test, y_pred):
    return confusion_matrix(y_test, y_pred)
