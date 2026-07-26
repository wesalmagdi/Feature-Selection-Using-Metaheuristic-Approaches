"""
fitness.py
----------
A single fitness function shared by GA, PSO, GWO, and WOA.

Every optimizer treats a candidate solution as a real-valued vector in
[0, 1]^n_features; it is binarized (>0.5 -> selected) before evaluation.
The fitness combines classification error with a small penalty on the
number of selected features, which is standard practice in
feature-selection metaheuristics (Emary et al., 2016; Mirjalili & Lewis, 2016).

Because every algorithm calls this exact function with the exact same
train/test split, results in 07_Comparison.ipynb are directly comparable.
"""

import numpy as np
from utils.metrics import evaluate_subset

# Weight on accuracy vs. feature-count reduction. alpha close to 1 heavily
# favors classification accuracy; (1 - alpha) rewards using fewer features.
ALPHA = 0.99


def binarize(solution, threshold=0.5):
    """Convert a continuous solution vector into a binary feature mask."""
    solution = np.asarray(solution)
    return (solution > threshold).astype(int)


def fitness(
    solution,
    X_train,
    X_test,
    y_train,
    y_test,
    classifier="svm",
    alpha=ALPHA,
    minimize=True,
):
    """
    Compute the fitness of a candidate feature-selection solution.

    Parameters
    ----------
    solution : array-like, continuous vector in [0, 1] (as produced by
        GA/PSO/GWO/WOA) OR an already-binary mask.
    minimize : bool
        If True (default), returns a value to MINIMIZE:
            fitness = alpha * error_rate + (1 - alpha) * (n_selected / n_total)
        Most metaheuristic libraries (including mealpy) minimize by default.
        If False, returns a value to MAXIMIZE (1 - the above).

    Returns
    -------
    float fitness score
    """
    mask = binarize(solution)
    n_total = len(mask)

    if mask.sum() == 0:
        # No features selected -> worst possible fitness
        return 1.0 if minimize else 0.0

    result = evaluate_subset(mask, X_train, X_test, y_train, y_test, classifier=classifier)

    error_rate = 1.0 - result["accuracy"]
    feature_ratio = mask.sum() / n_total

    score = alpha * error_rate + (1 - alpha) * feature_ratio

    return score if minimize else (1.0 - score)


def make_fitness_fn(X_train, X_test, y_train, y_test, classifier="svm", alpha=ALPHA):
    """
    Convenience factory: returns a fitness(solution) -> float closure with
    the dataset baked in. Useful for passing directly into mealpy's
    Problem definition or a custom GA/PSO/GWO/WOA loop.
    """

    def _fn(solution):
        return fitness(
            solution, X_train, X_test, y_train, y_test, classifier=classifier, alpha=alpha
        )

    return _fn
