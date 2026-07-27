import time
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.svm import SVC


def get_classifier(
    name="svm",
    random_state=42,
    use_gpu=False,
):
    name = name.lower()

    if name == "svm":
        return SVC(
            kernel="rbf",
            probability=False,
            random_state=random_state,
        )

    if name in {"random_forest", "rf"}:
        return RandomForestClassifier(
            n_estimators=100,
            random_state=random_state,
            n_jobs=-1,
        )

    if name in {"xgboost", "xgb"}:
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            tree_method="hist",
            device="cuda" if use_gpu else "cpu",
            random_state=random_state,
            n_jobs=-1,
            verbosity=0,
        )

    raise ValueError(f"Unknown classifier: {name}")


def evaluate_subset(
    mask,
    X_train,
    X_eval,
    y_train,
    y_eval,
    classifier="svm",
    random_state=42,
    use_gpu=False,
):
    mask = np.asarray(mask, dtype=bool)

    if mask.sum() == 0:
        raise ValueError("The mask selects zero features")

    X_train_selected = X_train[:, mask]
    X_eval_selected = X_eval[:, mask]

    model = get_classifier(
        classifier,
        random_state=random_state,
        use_gpu=use_gpu,
    )

    start = time.perf_counter()

    model.fit(X_train_selected, y_train)
    y_pred = model.predict(X_eval_selected)

    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_eval_selected)[:, 1]
    else:
        y_score = model.decision_function(X_eval_selected)

    runtime = time.perf_counter() - start

    return {
        "accuracy": accuracy_score(y_eval, y_pred),
        "precision": precision_score(
            y_eval, y_pred, zero_division=0
        ),
        "recall": recall_score(
            y_eval, y_pred, zero_division=0
        ),
        "f1": f1_score(
            y_eval, y_pred, zero_division=0
        ),
        "roc_auc": roc_auc_score(y_eval, y_score),
        "n_features": int(mask.sum()),
        "runtime": runtime,
        "y_pred": y_pred,
        "y_score": y_score,
        "mask": mask,
    }


def get_confusion_matrix(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)