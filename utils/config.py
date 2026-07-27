from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_DIR = ROOT / "datasets"
PROCESSED_DIR = RAW_DATA_DIR / "processed"

RUN_MODE = "final"  # change to "final" after testing

DATASETS = ["breast", "heart"]
CLASSIFIERS = ["svm", "random_forest", "xgboost"]

# Keep the classifier fixed so only the metaheuristic randomness changes.
CLASSIFIER_SEED = 42

ALPHA = 0.99
USE_XGB_GPU = True

if RUN_MODE == "smoke":
    OPTIMIZER_SEEDS = [0]
    POP_SIZE = 5
    ITERATIONS = 3

elif RUN_MODE == "final":
    OPTIMIZER_SEEDS = [0, 1, 2, 3, 4]
    POP_SIZE = 10
    ITERATIONS = 20

else:
    raise ValueError("RUN_MODE must be 'smoke' or 'final'")

RESULTS_DIR = ROOT / "results" / RUN_MODE
ARTIFACTS_DIR = RESULTS_DIR / "artifacts"
PLOTS_DIR = RESULTS_DIR / "plots"