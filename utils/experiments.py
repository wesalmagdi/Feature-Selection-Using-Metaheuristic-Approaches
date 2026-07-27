import json
import random
import time

import numpy as np
import pandas as pd

from utils.config import (
    ROOT,
    DATASETS,
    CLASSIFIERS,
    CLASSIFIER_SEED,
    OPTIMIZER_SEEDS,
    POP_SIZE,
    ITERATIONS,
    ALPHA,
    USE_XGB_GPU,
    RESULTS_DIR,
    ARTIFACTS_DIR,
)
from utils.preprocessing import load_processed_data
from utils.fitness import make_fitness_fn
from utils.metrics import evaluate_subset


def run_feature_selector(algorithm_name, runner):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    result_path = (
        RESULTS_DIR
        / f"{algorithm_name.lower()}_results.csv"
    )

    if result_path.exists():
        rows = pd.read_csv(result_path).to_dict("records")
    else:
        rows = []

    completed = {
        (
            row["Dataset"],
            row["Classifier"],
            int(row["Seed"]),
        )
        for row in rows
    }

    for dataset_name in DATASETS:
        (
            X_train,
            X_val,
            X_test,
            y_train,
            y_val,
            y_test,
            feature_names,
        ) = load_processed_data(dataset_name)

        # After feature selection, train the final model on
        # training + validation.
        X_development = np.vstack([X_train, X_val])
        y_development = np.concatenate([y_train, y_val])

        n_features = X_train.shape[1]

        for classifier_name in CLASSIFIERS:
            for seed in OPTIMIZER_SEEDS:
                key = (
                    dataset_name,
                    classifier_name,
                    seed,
                )

                if key in completed:
                    print("Skipping completed:", key)
                    continue

                random.seed(seed)
                np.random.seed(seed)

                use_gpu = (
                    USE_XGB_GPU
                    and classifier_name == "xgboost"
                )

                objective = make_fitness_fn(
                    X_train,
                    X_val,
                    y_train,
                    y_val,
                    classifier=classifier_name,
                    alpha=ALPHA,
                    random_state=CLASSIFIER_SEED,
                    use_gpu=use_gpu,
                )

                start = time.perf_counter()

                best_mask, validation_fitness, curve = (
                    runner(
                        objective,
                        n_features,
                        POP_SIZE,
                        ITERATIONS,
                    )
                )

                selection_runtime = (
                    time.perf_counter() - start
                )

                best_mask = np.asarray(
                    best_mask,
                    dtype=int,
                )

                # First and only use of the test set for this run.
                test_result = evaluate_subset(
                    best_mask,
                    X_development,
                    X_test,
                    y_development,
                    y_test,
                    classifier=classifier_name,
                    random_state=CLASSIFIER_SEED,
                    use_gpu=use_gpu,
                )

                stem = (
                    f"{dataset_name}__"
                    f"{classifier_name}__"
                    f"{algorithm_name.lower()}__"
                    f"seed{seed}"
                )

                mask_path = (
                    ARTIFACTS_DIR / f"{stem}__mask.npy"
                )
                curve_path = (
                    ARTIFACTS_DIR
                    / f"{stem}__convergence.npy"
                )

                np.save(mask_path, best_mask)
                np.save(curve_path, np.asarray(curve))

                selected_names = [
                    feature
                    for feature, selected
                    in zip(feature_names, best_mask)
                    if selected
                ]

                rows.append({
                    "Dataset": dataset_name,
                    "Classifier": classifier_name,
                    "Algorithm": algorithm_name,
                    "Seed": seed,
                    "ValidationFitness": (
                        validation_fitness
                    ),
                    "Accuracy": test_result["accuracy"],
                    "Precision": test_result["precision"],
                    "Recall": test_result["recall"],
                    "F1": test_result["f1"],
                    "ROC_AUC": test_result["roc_auc"],
                    "Features": test_result["n_features"],
                    "SelectionRuntime": (
                        selection_runtime
                    ),
                    "TestRuntime": (
                        test_result["runtime"]
                    ),
                    "SelectedFeatureNames": json.dumps(
                        selected_names
                    ),
                    "MaskFile": str(
                        mask_path.relative_to(ROOT)
                    ),
                    "ConvergenceFile": str(
                        curve_path.relative_to(ROOT)
                    ),
                })

                # Save immediately so an interrupted Kaggle
                # session can resume.
                pd.DataFrame(rows).to_csv(
                    result_path,
                    index=False,
                )

                completed.add(key)
                print("Saved:", key)

    return pd.DataFrame(rows)