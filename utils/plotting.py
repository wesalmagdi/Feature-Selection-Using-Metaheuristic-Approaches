"""
plotting.py
-----------
Shared plotting utilities for 07_Comparison.ipynb (and any notebook that
wants a quick visual). Kept separate from analysis logic so plot styling
stays consistent across the project.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay


def plot_convergence(convergence_curves: dict, title="Convergence Comparison", save_path=None):
    """
    Plot fitness-vs-iteration curves for multiple optimizers on one axis.

    convergence_curves : dict[str, list[float]]
        e.g. {"GA": [...], "PSO": [...], "GWO": [...], "WOA": [...]}
    """
    plt.figure(figsize=(8, 5))
    for name, curve in convergence_curves.items():
        plt.plot(curve, label=name, linewidth=2)
    plt.xlabel("Iteration")
    plt.ylabel("Best Fitness (lower = better)")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_feature_counts(results: dict, save_path=None):
    """
    Bar chart of number of selected features per algorithm.

    results : dict[str, int]  e.g. {"Baseline": 30, "GA": 14, "PSO": 12, ...}
    """
    names = list(results.keys())
    counts = list(results.values())

    plt.figure(figsize=(7, 5))
    sns.barplot(x=names, y=counts, palette="viridis")
    plt.ylabel("Number of Selected Features")
    plt.title("Feature Count Comparison")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_runtime_comparison(results: dict, save_path=None):
    """
    Bar chart of runtime (seconds) per algorithm.

    results : dict[str, float]  e.g. {"Baseline": 0.2, "GA": 12.4, ...}
    """
    names = list(results.keys())
    times = list(results.values())

    plt.figure(figsize=(7, 5))
    sns.barplot(x=names, y=times, palette="magma")
    plt.ylabel("Runtime (seconds)")
    plt.title("Runtime Comparison")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_confusion_matrices(cms: dict, class_names=None, save_path=None):
    """
    Grid of confusion matrices, one subplot per algorithm.

    cms : dict[str, np.ndarray]  e.g. {"Baseline": cm1, "GA": cm2, ...}
    """
    n = len(cms)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (name, cm) in zip(axes, cms.items()):
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(name)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_metric_comparison(df, metric="accuracy", save_path=None):
    """
    Bar chart comparing a single metric (e.g. accuracy, f1) across algorithms.

    df : pandas.DataFrame with an "Algorithm" column and a column named `metric`.
    """
    plt.figure(figsize=(7, 5))
    sns.barplot(data=df, x="Algorithm", y=metric, palette="crest")
    plt.title(f"{metric.capitalize()} Comparison")
    plt.ylim(0, 1.05) if df[metric].max() <= 1.0 else None
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.show()
