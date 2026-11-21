"""
Fraud Detection with PyOD AutoEncoder
Author: Harika Surineni
Course: MSCS 633 - Advanced Artificial Intelligence
Purpose: Unsupervised anomaly detection on an anonymized credit-card dataset.

Run:
    python fraud_detection_autoencoder.py --data data/creditcard.csv

Supported target columns: Class, class, is_fraud, isFraud, fraud.
The target is kept out of model training and used only for evaluation.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from pyod.models.auto_encoder import AutoEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42
TARGET_CANDIDATES = ("Class", "class", "is_fraud", "isFraud", "fraud")
IDENTIFIER_HINTS = (
    "id", "cc_num", "card_number", "trans_num", "transaction_id",
    "name", "first", "last", "street", "dob", "ssn"
)


def configure_logging() -> None:
    """Configure readable console messages."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )


def locate_target(data: pd.DataFrame) -> Optional[str]:
    """Return the first recognized fraud-label column, if present."""
    lookup = {column.lower(): column for column in data.columns}
    for candidate in TARGET_CANDIDATES:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    return None


def prepare_features(data: pd.DataFrame, target_column: Optional[str]) -> pd.DataFrame:
    """
    Keep numeric behavioral features and remove labels/identifiers.
    This avoids feeding personal identifiers or the answer into the model.
    """
    excluded = set()
    if target_column:
        excluded.add(target_column)

    for column in data.columns:
        normalized = str(column).strip().lower()
        if any(hint in normalized for hint in IDENTIFIER_HINTS):
            excluded.add(column)

    numeric = data.drop(columns=list(excluded), errors="ignore").select_dtypes(
        include=[np.number]
    )
    # Remove constant columns because they add no useful information.
    numeric = numeric.loc[:, numeric.nunique(dropna=True) > 1]

    if numeric.empty:
        raise ValueError(
            "No usable numeric features were found. Check the CSV columns "
            "and update prepare_features() for this dataset."
        )

    # Replace infinite values before imputation.
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    return numeric


def main() -> None:
    configure_logging()

    parser = argparse.ArgumentParser(
        description="Train a PyOD AutoEncoder for credit-card fraud anomaly detection."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/creditcard.csv"),
        help="Path to the Kaggle CSV file (default: data/creditcard.csv).",
    )
    parser.add_argument(
        "--contamination",
        type=float,
        default=0.002,
        help="Expected anomaly fraction; adjust to the dataset (default: 0.002).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of AutoEncoder training epochs (default: 20).",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=100000,
        help="Maximum rows to use for a manageable first experiment (default: 100000).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs"),
        help="Folder for prediction and metric files.",
    )
    args = parser.parse_args()

    if not 0.0 < args.contamination < 0.5:
        parser.error("--contamination must be between 0 and 0.5.")

    if not args.data.exists():
        raise FileNotFoundError(
            f"Dataset not found: {args.data}. Download the Kaggle CSV and place it "
            "at this path, or pass its location with --data."
        )

    logging.info("Loading dataset: %s", args.data)
    raw = pd.read_csv(args.data, low_memory=False)
    if len(raw) > args.sample:
        raw = raw.sample(n=args.sample, random_state=RANDOM_STATE)
        logging.info("Using a reproducible sample of %s rows.", args.sample)

    target_column = locate_target(raw)
    y = None
    if target_column:
        y = pd.to_numeric(raw[target_column], errors="coerce").fillna(0).astype(int)
        logging.info("Found evaluation label column: %s", target_column)
    else:
        logging.warning(
            "No recognized fraud label was found. The model will produce anomaly "
            "scores only; supervised evaluation metrics will be skipped."
        )

    X = prepare_features(raw, target_column)
    logging.info("Rows: %s | Numeric features: %s", X.shape[0], X.shape[1])

    # Keep a separate evaluation set when labels are available.
    if y is not None and y.nunique() > 1:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
        )
    else:
        X_train, X_test = train_test_split(
            X, test_size=0.25, random_state=RANDOM_STATE
        )
        y_train = y_test = None

    # Fit preprocessing only on training data to prevent test-set leakage.
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_train_ready = scaler.fit_transform(imputer.fit_transform(X_train))
    X_test_ready = scaler.transform(imputer.transform(X_test))

    detector = AutoEncoder(
        contamination=args.contamination,
        preprocessing=False,  # Scaling is handled explicitly above.
        lr=1e-3,
        epoch_num=args.epochs,
        batch_size=256,
        hidden_neuron_list=[32, 16],
        hidden_activation_name="relu",
        batch_norm=True,
        dropout_rate=0.1,
        random_state=RANDOM_STATE,
        verbose=1,
    )

    logging.info("Training PyOD AutoEncoder...")
    detector.fit(X_train_ready)

    scores = detector.decision_function(X_test_ready)
    predictions = (scores > detector.threshold_).astype(int)

    args.output.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(
        {
            "row_index": X_test.index,
            "anomaly_score": scores,
            "predicted_anomaly": predictions,
        }
    )
    if y_test is not None:
        results["actual_fraud"] = y_test.to_numpy()

    result_path = args.output / "fraud_detection_predictions.csv"
    results.to_csv(result_path, index=False)

    summary = {
        "dataset": str(args.data),
        "rows_used": int(len(raw)),
        "training_rows": int(len(X_train)),
        "testing_rows": int(len(X_test)),
        "features_used": list(X.columns),
        "contamination": args.contamination,
        "epochs": args.epochs,
        "threshold": float(detector.threshold_),
        "predicted_anomalies": int(predictions.sum()),
        "prediction_file": str(result_path),
    }

    if y_test is not None:
        summary.update(
            {
                "precision": float(precision_score(y_test, predictions, zero_division=0)),
                "recall": float(recall_score(y_test, predictions, zero_division=0)),
                "f1_score": float(f1_score(y_test, predictions, zero_division=0)),
                "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
            }
        )
        try:
            summary["roc_auc"] = float(roc_auc_score(y_test, scores))
        except ValueError:
            summary["roc_auc"] = None

        print("\nEvaluation report")
        print(classification_report(y_test, predictions, zero_division=0))
        print("Confusion matrix:")
        print(confusion_matrix(y_test, predictions))
    else:
        print("\nNo ground-truth labels available; reporting anomaly scores only.")

    summary_path = args.output / "experiment_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nExperiment completed.")
    print(f"Anomaly threshold: {detector.threshold_:.6f}")
    print(f"Predicted anomalies: {predictions.sum()} of {len(predictions)} test rows")
    print(f"Predictions saved to: {result_path}")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
