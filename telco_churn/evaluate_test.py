
import json
from pathlib import Path

import joblib
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from telco_churn.dataset import load_data
from telco_churn.train import prepare_features, prepare_target, split_data

ARTIFACT_PATH = Path("models/telco_churn_pipeline.joblib")
OUTPUT_PATH = Path("reports/final_test_metrics.json")


def main():
    # Load the selected, fitted pipeline. Do not retrain it.
    model = joblib.load(ARTIFACT_PATH)

    # Recreate the fixed split and use only the reserved test partition.
    df = load_data()
    _, _, test_df = split_data(df)

    X_test = prepare_features(test_df)
    y_test = prepare_target(test_df)

    predictions = model.predict(X_test)
    positive_index = list(model.classes_).index(1)
    scores = model.predict_proba(X_test)[:, positive_index]

    metrics = {
        "model": "random_forest",
        "evaluation_split": "test",
        "sample_count": len(test_df),
        "positive_class": "Yes",
        "classification_threshold": 0.5,
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "average_precision": average_precision_score(y_test, scores),
        "confusion_matrix_labels": ["No", "Yes"],
        "confusion_matrix": confusion_matrix(
            y_test, predictions, labels=[0, 1]
        ).tolist(),
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )

    print("Final test-set evaluation")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    print(f"\nSaved metrics to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
