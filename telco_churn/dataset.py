import csv
import hashlib
import json
from pathlib import Path
import platform

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE = PROJECT_ROOT / "data" / "raw" / "datasets.csv"
REPORT = PROJECT_ROOT / "reports" / "data_overview.json"


def load_data(source: Path = SOURCE) -> pd.DataFrame:
    """Load the original Telco CSV without modifying the source file."""
    df = pd.read_csv(
        source,
        encoding="utf-8-sig",
        quoting=csv.QUOTE_NONE,
    )

    # Remove the outer quotes left by this CSV's unusual format.
    df.columns = df.columns.str.strip('"')
    df.iloc[:, 0] = df.iloc[:, 0].str.strip('"')
    df.iloc[:, -1] = df.iloc[:, -1].str.strip('"')

    return df


def inspect_data(source: Path = SOURCE) -> dict:
    """Inspect the dataset and return a JSON-serializable report."""
    df = load_data(source)

    # Count actual missing values and empty/whitespace-only strings.
    missing_counts = {}
    for column in df.columns:
        values = df[column]
        blank_values = values.astype("string").str.strip().eq("").fillna(False)
        missing_counts[column] = int((values.isna() | blank_values).sum())

    churn_counts = df["Churn"].value_counts(dropna=False)
    churn_proportions = df["Churn"].value_counts(dropna=False, normalize=True)

    return {
        "source": {
            "path": str(source.relative_to(PROJECT_ROOT)),
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        },
        "environment": {
            "python_version": platform.python_version(),
            "pandas_version": pd.__version__,
        },
        "dimensions": {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
        },
        "column_types": {column: str(dtype) for column, dtype in df.dtypes.items()},
        "target": {
            "column": "Churn",
            "counts": {str(key): int(value) for key, value in churn_counts.items()},
            "proportions": {str(key): float(value) for key, value in churn_proportions.items()},
        },
        "duplicate_customer_ids": int(df["customerID"].duplicated().sum()),
        "missing_values": missing_counts,
    }


def main() -> None:
    report = inspect_data()
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Inspection report saved to: {REPORT.relative_to(PROJECT_ROOT)}")
    print(json.dumps(report["dimensions"], indent=2))


if __name__ == "__main__":
    main()
