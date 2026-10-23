import json
import time
import hashlib
import platform
import sys

import joblib
import sklearn
from pathlib import Path
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

from telco_churn.dataset import load_data

SEED = 42

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_FEATURES = ["Contract", "InternetService", "PaymentMethod"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def prepare_features(df):
    """Validate and convert raw input columns without fitting transformations."""
    X = df[FEATURES].copy()

    for column in NUMERIC_FEATURES:
        # Treat empty or whitespace-only strings as genuinely missing.
        values = X[column].astype("string").str.strip()
        values = values.mask(values.eq(""), pd.NA)

        # Raise on malformed nonblank values instead of hiding data errors.
        X[column] = pd.to_numeric(values, errors="raise")

    for column in CATEGORICAL_FEATURES:
        values = X[column].astype("string").str.strip()
        X[column] = values.mask(values.eq(""), pd.NA)

    return X


def prepare_target(df):
    """Encode No churn as 0 and Yes churn as 1."""
    y = df["Churn"].map({"No": 0, "Yes": 1})

    if y.isna().any():
        raise ValueError("Missing or unexpected Churn labels.")

    return y.astype(int)


def split_data(df):
    """Create reproducible stratified 60/20/20 splits."""
    train_df, remaining_df = train_test_split(
        df,
        test_size=0.4,
        stratify=df["Churn"],
        random_state=SEED,
    )

    valid_df, test_df = train_test_split(
        remaining_df,
        test_size=0.5,
        stratify=remaining_df["Churn"],
        random_state=SEED,
    )

    # Keep customer IDs out of the feature matrix and verify disjoint splits.
    id_sets = [
        set(part["customerID"])
        for part in (train_df, valid_df, test_df)
    ]
    if (
        id_sets[0] & id_sets[1]
        or id_sets[0] & id_sets[2]
        or id_sets[1] & id_sets[2]
    ):
        raise ValueError("Customer IDs overlap between splits.")

    return train_df, valid_df, test_df


def build_pipeline(classifier):
    """Return a fresh, unfitted preprocessing and classification pipeline."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )

def save_selected_model(model, selected_metrics, train_df, valid_df, test_df):
    """Save the fitted pipeline and metadata without retraining it."""
    artifact_dir = Path("models")
    artifact_dir.mkdir(parents=True, exist_ok=True)

    artifact_path = artifact_dir / "telco_churn_pipeline.joblib"
    metadata_path = artifact_dir / "metadata.json"
    predictions_path = artifact_dir / "validation_predictions.json"

    # Hash the original dataset file.
    dataset_path = Path("data/raw/datasets.csv")
    dataset_hash = hashlib.sha256(dataset_path.read_bytes()).hexdigest()

    # Save the entire fitted pipeline, including preprocessing.
    joblib.dump(model, artifact_path)

    # Preserve predictions to verify reload consistency later.
    # The predictions are generated from validation data, not the test set.
    X_valid = prepare_features(valid_df)
    validation_predictions = model.predict(X_valid).tolist()
    (predictions_path).write_text(
        json.dumps(validation_predictions, indent=2) + "\n",
        encoding="utf-8",
    )

    metadata = {
        "selected_model": selected_metrics["model"],
        "selection_metric": "validation_average_precision",
        "positive_class": {"label": "Yes", "encoded_value": 1},
        "classification_threshold": 0.5,
        "random_seed": SEED,
        "split_sizes": {
            "train": len(train_df),
            "validation": len(valid_df),
            "test": len(test_df),
        },
        "features": {
            "numeric": NUMERIC_FEATURES,
            "categorical": CATEGORICAL_FEATURES,
        },
        "preprocessing": {
            "numeric_imputation": "median",
            "numeric_scaling": "StandardScaler",
            "categorical_imputation": "most_frequent",
            "categorical_encoding": "OneHotEncoder(handle_unknown='ignore')",
        },
        "model_parameters": model.named_steps["classifier"].get_params(),
        "validation_metrics": selected_metrics,
        "dataset_sha256": dataset_hash,
        "python_version": sys.version,
        "platform": platform.platform(),
        "scikit_learn_version": sklearn.__version__,
        "joblib_version": joblib.__version__,
        "artifact_path": str(artifact_path),
    }

    metadata_path.write_text(
        json.dumps(metadata, indent=2, default=str) + "\n",
        encoding="utf-8",
    )

    print(f"\nSaved fitted pipeline: {artifact_path}")
    print(f"Saved metadata: {metadata_path}")
    print(f"Saved validation predictions: {predictions_path}")

def main():
    df = load_data()

    if df["customerID"].duplicated().any():
        raise ValueError("Duplicate customer IDs detected.")

    train_df, valid_df, test_df = split_data(df)

    X_train = prepare_features(train_df)
    X_valid = prepare_features(valid_df)
    y_train = prepare_target(train_df)
    y_valid = prepare_target(valid_df)

    candidates = {
        "majority_baseline": None,
        "logistic_regression": LogisticRegression(max_iter=1000),
        "decision_tree": DecisionTreeClassifier(
            max_depth=5, random_state=SEED
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100, max_depth=8, random_state=SEED,
            n_jobs=-1,
        ),
    }

    results = []
    fitted_models = {}

    for name, classifier in candidates.items():
        if classifier is None:
            model = DummyClassifier(strategy="most_frequent")
        else:
            model = build_pipeline(clone(classifier))

        start = time.perf_counter()
        model.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - start

        predictions = model.predict(X_valid)

        # All selected classifiers support predict_proba.
        positive_index = list(model.classes_).index(1)
        scores = model.predict_proba(X_valid)[:, positive_index]

        metrics = {
            "model": name,
            "fit_seconds": round(fit_seconds, 4),
            "accuracy": accuracy_score(y_valid, predictions),
            "precision": precision_score(
                y_valid, predictions, zero_division=0
            ),
            "recall": recall_score(
                y_valid, predictions, zero_division=0
            ),
            "f1": f1_score(y_valid, predictions, zero_division=0),
            "average_precision": average_precision_score(y_valid, scores),
            "confusion_matrix": confusion_matrix(
                y_valid, predictions, labels=[0, 1]
            ).tolist(),
        }

        results.append(metrics)
        fitted_models[name] = model

    # Primary metric chosen before comparison: validation average precision.
    results.sort(key=lambda item: item["average_precision"], reverse=True)

    print(
        f"Split sizes: train={len(train_df)}, "
        f"validation={len(valid_df)}, test={len(test_df)}"
    )
    print("Primary selection metric: validation average precision")
    for result in results:
        print(f"\n{result['model']}")
        for key, value in result.items():
            if key != "model":
                print(f"  {key}: {value}")

    report_dir = Path("reports")
    report_dir.mkdir(exist_ok=True)
    (report_dir / "model_comparison.json").write_text(
        json.dumps(results, indent=2) + "\n",
        encoding="utf-8",
    )
    print("\nSaved comparison to reports/model_comparison.json")

    (report_dir / "model_comparison.json").write_text(
        json.dumps(results, indent=2) + "\n",
        encoding="utf-8",
    )
    print("\nSaved comparison to reports/model_comparison.json")

    selected_metrics = next(
        result for result in results
        if result["model"] == "random_forest"
    )

    save_selected_model(
        model=fitted_models["random_forest"],
        selected_metrics=selected_metrics,
        train_df=train_df,
        valid_df=valid_df,
        test_df=test_df,
    )


if __name__ == "__main__":
    main()