# Feature Selection for Medical Data — GA vs PSO vs GWO vs WOA vs CSA

This project compares five metaheuristic feature-selection algorithms against a
full-feature baseline:

- Genetic Algorithm (GA)
- Binary Particle Swarm Optimization (PSO)
- Binary Grey Wolf Optimizer (GWO)
- Binary Whale Optimization Algorithm (WOA)
- Binary Crow Search Algorithm (CSA)

Every feature selector is evaluated with three classifiers:

- Support Vector Machine (SVM)
- Random Forest
- XGBoost

The experiments use two medical classification datasets:

- Wisconsin Diagnostic Breast Cancer
- UCI Heart Disease, converted into binary classification:
  `0 = no heart disease` and `1 = heart disease`

The project is designed for Kaggle Notebooks, but it can run in any Python
environment with the required packages installed.

## Experimental design

Each dataset is split using stratified sampling into:

- 60% training data
- 20% validation data
- 20% test data

The preprocessing pipeline is fitted only on the training data. The validation
and test sets are transformed using that already-fitted pipeline.

During feature selection, every candidate feature mask is evaluated using only
the training and validation sets:

```text
fitness = 0.99 × validation error
        + 0.01 × selected-feature ratio
```

The test set is never used to guide the optimizers. It is used only after an
optimizer has fixed its final feature mask. The final classifier is then trained
using the combined training and validation data and evaluated once on the test
set.

This separation prevents test-data leakage and keeps the final evaluation fair.

## Repeated runs

The feature selectors are stochastic, so each optimizer is repeated with
multiple optimizer seeds. In final mode, the configured seeds are:

```python
OPTIMIZER_SEEDS = [0, 1, 2, 3, 4]
```

Results are summarized using the mean and standard deviation across these
repeated runs. With five selectors, three classifiers, two datasets, and five
seeds, the final experiment contains:

```text
5 selectors × 3 classifiers × 2 datasets × 5 seeds
= 150 feature-selection runs
```

## Why multiple notebooks?

Splitting the work by stage keeps the project easier to understand, rerun, and
debug. All optimizer notebooks use the same shared preprocessing, fitness,
evaluation, and experiment-running functions from `utils/`. Therefore, all
five algorithms are evaluated under the same conditions.

## Notebook structure

```text
notebooks/
├── 01_Data_Preprocessing.ipynb
│   └── Loads, explores, preprocesses, splits, and saves both datasets.
│
├── 02_Baseline_Classifier.ipynb
│   └── Evaluates SVM, Random Forest, and XGBoost using all features.
│
├── 03_Genetic_Algorithm_FS.ipynb
│   └── GA feature selection with repeated optimizer seeds.
│
├── 04_PSO_FS.ipynb
│   └── Binary PSO feature selection with repeated optimizer seeds.
│
├── 05_GWO_FS.ipynb
│   └── Binary GWO feature selection with repeated optimizer seeds.
│
├── 06_WOA_FS.ipynb
│   └── Binary WOA feature selection with repeated optimizer seeds.
│
├── 07_Crow_Search_FS.ipynb
│   └── Binary CSA feature selection with repeated optimizer seeds.
│
└── 08_Comparison.ipynb
    └── Combines all saved results and creates the final tables and plots.
```

`07_Crow_Search_FS.ipynb` uses Binary CSA because feature selection requires a
binary decision for every feature:

- `1` means the feature is selected.
- `0` means the feature is not selected.

The project includes only Binary CSA because it is the version needed for the
feature-selection experiment.

## Folder structure

```text
Feature-Selection-Using-Metaheuristic-Approaches/
├── datasets/
│   ├── Breast Cancer Wisconsin Diagnostic Data Set.csv
│   ├── heart_disease_uci.csv
│   └── processed/
│       ├── breast/
│       │   ├── X_train.npy
│       │   ├── X_val.npy
│       │   ├── X_test.npy
│       │   ├── y_train.npy
│       │   ├── y_val.npy
│       │   ├── y_test.npy
│       │   └── feature_names.txt
│       └── heart/
│           └── Same processed files
│
├── notebooks/
│   └── Notebooks 01–08 described above
│
├── utils/
│   ├── config.py
│   │   └── Datasets, classifiers, seeds, run mode, and experiment settings
│   ├── preprocessing.py
│   │   └── Dataset loading and leakage-safe preprocessing
│   ├── metrics.py
│   │   └── Classifier creation and common evaluation metrics
│   ├── fitness.py
│   │   └── Validation-only feature-selection fitness
│   ├── experiments.py
│   │   └── Shared repeated-run and result-saving logic
│   └── plotting.py
│       └── Shared plotting utilities, if used
│
├── results/
│   ├── smoke/
│   └── final/
│
├── requirements.txt
└── README.md
```

## Run modes

Use the setting in `utils/config.py`:

```python
RUN_MODE = "smoke"
```

Smoke mode performs a small test run so errors can be found quickly. After all
notebooks finish successfully, change it to:

```python
RUN_MODE = "final"
```

Smoke and final outputs are stored separately in `results/smoke/` and
`results/final/`.

## Run order

1. Run `01_Data_Preprocessing.ipynb` first.
2. Run `02_Baseline_Classifier.ipynb`.
3. Run notebooks `03`–`07` in any order.
4. Run `08_Comparison.ipynb` last.

Always complete this order in smoke mode before starting the full final runs.

## Shared evaluation

Every optimizer uses the same objective created by
`utils.fitness.make_fitness_fn()`:

```python
objective = make_fitness_fn(
    X_train,
    X_val,
    y_train,
    y_val,
    classifier=classifier_name,
)
```

After the final mask is selected, the shared experiment runner performs the
first and only test evaluation for that optimizer run:

```python
test_result = evaluate_subset(
    best_mask,
    X_development,
    X_test,
    y_development,
    y_test,
    classifier=classifier_name,
)
```

This common workflow ensures that GA, PSO, GWO, WOA, and CSA are compared using
the same data splits, fitness definition, classifiers, metrics, and repeated
seeds.

## Evaluation outputs

The project records:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Number of selected features
- Feature-selection runtime
- Final test runtime
- Best validation fitness
- Selected feature mask
- Convergence history

`08_Comparison.ipynb` produces mean ± standard-deviation summaries, performance
plots, feature-count and runtime comparisons, convergence plots, and
feature-selection frequency visualizations.

## Setup

Install the repository requirements:

```bash
pip install -r requirements.txt
```

On Kaggle, a GPU can be enabled for XGBoost. The SVM, Random Forest, and Python
metaheuristic loops mainly use the CPU.
