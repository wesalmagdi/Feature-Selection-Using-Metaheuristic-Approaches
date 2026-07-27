from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from utils.config import RAW_DATA_DIR, PROCESSED_DIR


def load_raw_dataset(dataset_name, raw_dir=RAW_DATA_DIR):
    raw_dir = Path(raw_dir)

    if dataset_name == "breast":
        path = raw_dir / "Breast Cancer Wisconsin Diagnostic Data Set.csv"
        df = pd.read_csv(path)

        # Remove the identifier and the completely empty last column.
        df = df.drop(columns=["id"], errors="ignore")
        df = df.dropna(axis=1, how="all")

        # Malignant is the medically important positive class.
        y = df.pop("diagnosis").map({"B": 0, "M": 1})

        if y.isna().any():
            raise ValueError("Unexpected diagnosis label")

        return df, y.astype(int)

    if dataset_name == "heart":
        path = raw_dir / "heart_disease_uci.csv"
        df = pd.read_csv(path)

        # Convert the 0–4 severity target into:
        # 0 = no heart disease, 1 = heart disease present.
        y = (df.pop("num") > 0).astype(int)

        # id is only an identifier.
        # dataset identifies the source hospital/cohort and may produce
        # dataset-specific bias.
        X = df.drop(columns=["id", "dataset"], errors="ignore")

        return X, y

    raise ValueError(f"Unknown dataset: {dataset_name}")


def preprocess_and_split(dataset_name, random_state=42):
    X, y = load_raw_dataset(dataset_name)

    # 60% train, 20% validation, 20% test.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.40,
        random_state=random_state,
        stratify=y,
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=random_state,
        stratify=y_temp,
    )

    numeric_columns = (
        X_train.select_dtypes(include=np.number).columns.tolist()
    )
    categorical_columns = [
        column for column in X_train.columns
        if column not in numeric_columns
    ]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            ),
        ),
    ])

    preprocessor = ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numeric_columns),
            (
                "categorical",
                categorical_pipeline,
                categorical_columns,
            ),
        ],
        verbose_feature_names_out=False,
    )

    # Fit only on training data.
    X_train_processed = preprocessor.fit_transform(X_train)

    # Validation and test only use transform.
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)

    feature_names = preprocessor.get_feature_names_out().tolist()

    return (
        X_train_processed,
        X_val_processed,
        X_test_processed,
        y_train.to_numpy(),
        y_val.to_numpy(),
        y_test.to_numpy(),
        feature_names,
    )


def save_processed_data(
    dataset_name,
    arrays,
    out_root=PROCESSED_DIR,
):
    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        feature_names,
    ) = arrays

    out_dir = Path(out_root) / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    values = {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
    }

    for name, value in values.items():
        np.save(out_dir / f"{name}.npy", value)

    (out_dir / "feature_names.txt").write_text(
        "\n".join(feature_names),
        encoding="utf-8",
    )


def load_processed_data(
    dataset_name,
    in_root=PROCESSED_DIR,
):
    in_dir = Path(in_root) / dataset_name

    arrays = [
        np.load(in_dir / f"{name}.npy")
        for name in [
            "X_train",
            "X_val",
            "X_test",
            "y_train",
            "y_val",
            "y_test",
        ]
    ]

    feature_names = (
        in_dir / "feature_names.txt"
    ).read_text(encoding="utf-8").splitlines()

    return (*arrays, feature_names)