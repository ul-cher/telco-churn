
from pathlib import Path

import joblib
import numpy as np

from telco_churn.dataset import load_data
from telco_churn.train import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    prepare_features,
    split_data,
)

ARTIFACT_PATH = Path("models/telco_churn_pipeline.joblib")


def main():
    model = joblib.load(ARTIFACT_PATH)

    df = load_data()
    train_df, valid_df, _ = split_data(df)

    X_train = prepare_features(train_df)
    X_valid = prepare_features(valid_df)

    # 1. Verify that numeric imputation statistics came from training data.
    preprocessor = model.named_steps["preprocessor"]
    numeric_pipeline = preprocessor.named_transformers_["num"]
    imputer = numeric_pipeline.named_steps["imputer"]
    scaler = numeric_pipeline.named_steps["scaler"]

    expected_medians = np.nanmedian(
        X_train[NUMERIC_FEATURES].to_numpy(dtype=float),
        axis=0,
    )
    np.testing.assert_allclose(imputer.statistics_, expected_medians)
    print("PASS: Numeric medians were learned from training data only.")

    # Verify the scaler mean uses the training data after median imputation.
    train_numeric = X_train[NUMERIC_FEATURES].to_numpy(dtype=float)
    missing_rows, missing_cols = np.where(np.isnan(train_numeric))
    train_numeric[missing_rows, missing_cols] = expected_medians[missing_cols]

    expected_means = train_numeric.mean(axis=0)
    np.testing.assert_allclose(scaler.mean_, expected_means)
    print("PASS: Scaler means were learned from imputed training data only.")

    # Save learned statistics to ensure prediction doesn't alter them.
    medians_before = imputer.statistics_.copy()
    means_before = scaler.mean_.copy()

    # 2. Missing numeric values should be handled by the fitted imputer.
    missing_X = X_valid.copy()
    missing_X.iloc[0, missing_X.columns.get_loc("tenure")] = np.nan
    missing_X.iloc[1, missing_X.columns.get_loc("TotalCharges")] = np.nan

    missing_predictions = model.predict(missing_X)
    assert len(missing_predictions) == len(missing_X)
    print("PASS: Missing numeric values are handled.")

    # Unseen categories should be ignored by the fitted encoder.
    unseen_X = X_valid.copy()
    unseen_X.iloc[0, unseen_X.columns.get_loc("Contract")] = (
        "PreviouslyUnseenContract"
    )

    unseen_predictions = model.predict(unseen_X)
    assert len(unseen_predictions) == len(unseen_X)
    print("PASS: Unseen categories are handled.")

    # 3. Predictions must not modify learned statistics.
    np.testing.assert_array_equal(imputer.statistics_, medians_before)
    np.testing.assert_array_equal(scaler.mean_, means_before)
    print("PASS: Prediction does not change fitted preprocessing statistics.")

    # 4. Missing required features must fail rather than be silently accepted.
    try:
        model.predict(X_valid.drop(columns=["tenure"]))
    except ValueError:
        print("PASS: Missing required feature raises ValueError.")
    else:
        raise AssertionError(
            "The pipeline unexpectedly accepted input without 'tenure'."
        )

    # Check that customer IDs are disjoint between the splits.
    valid_ids = set(valid_df["customerID"])
    train_ids = set(train_df["customerID"])
    assert train_ids.isdisjoint(valid_ids)
    print("PASS: Training and validation customer IDs do not overlap.")

    print("\nAll pipeline checks passed.")


if __name__ == "__main__":
    main()
