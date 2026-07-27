import numpy as np
from utils.metrics import evaluate_subset


def binarize(solution, threshold=0.5):
    solution = np.asarray(solution)
    return (solution > threshold).astype(int)


def fitness(
    solution,
    X_train,
    X_val,
    y_train,
    y_val,
    classifier="svm",
    alpha=0.99,
    random_state=42,
    use_gpu=False,
):
    mask = binarize(solution)

    if mask.sum() == 0:
        return 1.0

    validation_result = evaluate_subset(
        mask,
        X_train,
        X_val,
        y_train,
        y_val,
        classifier=classifier,
        random_state=random_state,
        use_gpu=use_gpu,
    )

    error_rate = 1.0 - validation_result["accuracy"]
    feature_ratio = mask.sum() / len(mask)

    return (
        alpha * error_rate
        + (1.0 - alpha) * feature_ratio
    )


def make_fitness_fn(
    X_train,
    X_val,
    y_train,
    y_val,
    classifier="svm",
    alpha=0.99,
    random_state=42,
    use_gpu=False,
):
    def objective(solution):
        return fitness(
            solution,
            X_train,
            X_val,
            y_train,
            y_val,
            classifier=classifier,
            alpha=alpha,
            random_state=random_state,
            use_gpu=use_gpu,
        )

    return objective