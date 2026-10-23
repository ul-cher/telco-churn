
import json
from pathlib import Path

import joblib
import numpy as np

from telco_churn.dataset import load_data
from telco_churn.train import prepare_features, split_data

ARTIFACT_PATH = Path("models/telco_churn_pipeline.joblib")
PREDICTIONS_PATH = Path("models/validation_predictions.json")


def main():
    # Load the saved pipeline; do not fit or retrain it.
    model = joblib.load(ARTIFACT_PATH)

    # Recreate the same validation split.
    df = load_data()
    _, valid_df, _ = split_data(df)
    X_valid = prepare_features(valid_df)

    # Compare predictions with those saved during training.
    expected = np.array(json.loads(PREDICTIONS_PATH.read_text()))
    actual = model.predict(X_valid)

    if not np.array_equal(expected, actual):
        raise AssertionError("Reloaded predictions do not match!")

    print("Reload verification: PASSED")
    print(f"Validation predictions checked: {len(actual)}")
    print("The saved pipeline reproduces the original predictions.")
    print("No retraining was performed.")


if __name__ == "__main__":
    main()
