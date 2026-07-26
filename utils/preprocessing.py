"""
preprocessing.py
----------------
Reusable data-loading and preprocessing utilities shared across notebooks.

Keeping this logic in one place guarantees that every optimizer
(GA, PSO, GWO, WOA) and the baseline classifier train/test on
*exactly* the same processed data, which is essential for a fair
comparison.
"""

import os
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


def load_dataset(source="breast_cancer", csv_path=None, target_column=None):
    """
    Load a medical dataset as a pandas DataFrame + target Series.

    Parameters
    ----------
    source : str
        "breast_cancer" -> uses sklearn's built-in Wisconsin Breast Cancer
        dataset (good default, no download required, works offline on Kaggle).
        "csv" -> loads a CSV file from `csv_path`, with `target_column` as
        the label column. Use this to swap in any other medical dataset
        (e.g. a Kaggle dataset added to the notebook's input directory).

    Returns
    -------
    X : pd.DataFrame
    y : pd.Series
    """
    if source == "breast_cancer":
        data = load_breast_cancer(as_frame=True)
        X = data.data.copy()
        y = data.target.copy()
        y.name = "target"
        return X, y

    elif source == "csv":
        if csv_path is None or target_column is None:
            raise ValueError("csv_path and target_column are required when source='csv'")
        df = pd.read_csv(csv_path)
        y = df[target_column]
        X = df.drop(columns=[target_column])
        return X, y

    else:
        raise ValueError(f"Unknown source: {source}")


def explore_dataset(X: pd.DataFrame, y: pd.Series):
    """Print a quick data-quality / shape summary. Returns nothing, side-effect only."""
    print("Shape:", X.shape)
    print("\nClass balance:")
    print(y.value_counts(normalize=True))
    print("\nMissing values per column:")
    missing = X.isnull().sum()
    print(missing[missing > 0] if missing.sum() > 0 else "No missing values.")
    print("\nDescribe:")
    print(X.describe().T)


def handle_missing_values(X: pd.DataFrame, strategy="median"):
    """
    Impute missing values column-wise.

    strategy : "median" | "mean" | "most_frequent" | "drop"
    """
    X = X.copy()
    if strategy == "drop":
        return X.dropna()

    for col in X.columns:
        if X[col].isnull().any():
            if strategy == "median":
                fill_value = X[col].median()
            elif strategy == "mean":
                fill_value = X[col].mean()
            elif strategy == "most_frequent":
                fill_value = X[col].mode()[0]
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            X[col] = X[col].fillna(fill_value)
    return X


def encode_labels(y: pd.Series):
    """Encode target labels as integers 0..n_classes-1. Returns (y_encoded, encoder)."""
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    return y_encoded, encoder


def scale_features(X_train, X_test):
    """Standardize features (zero mean, unit variance) fit on train, applied to test."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def preprocess_and_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size=0.2,
    random_state=42,
    missing_strategy="median",
):
    """
    Full pipeline: handle missing values -> encode labels -> split -> scale.

    Returns
    -------
    X_train, X_test, y_train, y_test, feature_names, scaler, encoder
    """
    X_clean = handle_missing_values(X, strategy=missing_strategy)
    y_encoded, encoder = encode_labels(y)

    feature_names = list(X_clean.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X_clean.values,
        y_encoded,
        test_size=test_size,
        random_state=random_state,
        stratify=y_encoded,
    )

    X_train, X_test, scaler = scale_features(X_train, X_test)

    return X_train, X_test, y_train, y_test, feature_names, scaler, encoder


def save_processed_data(X_train, X_test, y_train, y_test, feature_names, out_dir="../datasets"):
    """Save the processed train/test arrays + feature names to disk as .npy files."""
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "X_train.npy"), X_train)
    np.save(os.path.join(out_dir, "X_test.npy"), X_test)
    np.save(os.path.join(out_dir, "y_train.npy"), y_train)
    np.save(os.path.join(out_dir, "y_test.npy"), y_test)
    with open(os.path.join(out_dir, "feature_names.txt"), "w") as f:
        f.write("\n".join(feature_names))
    print(f"Saved processed data to {out_dir}")


def load_processed_data(in_dir="../datasets"):
    """Load previously saved processed train/test arrays + feature names."""
    X_train = np.load(os.path.join(in_dir, "X_train.npy"))
    X_test = np.load(os.path.join(in_dir, "X_test.npy"))
    y_train = np.load(os.path.join(in_dir, "y_train.npy"))
    y_test = np.load(os.path.join(in_dir, "y_test.npy"))
    with open(os.path.join(in_dir, "feature_names.txt")) as f:
        feature_names = f.read().splitlines()
    return X_train, X_test, y_train, y_test, feature_names
