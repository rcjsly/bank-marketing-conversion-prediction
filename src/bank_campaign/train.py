"""Train two baselines, select on validation data, evaluate once on test data."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from bank_campaign.data import CATEGORICAL, NUMERIC, download_data, load_data

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "bank-additional-full.csv"
ARTIFACT_PATH = ROOT / "artifacts" / "model.joblib"
REPORT_PATH = ROOT / "artifacts" / "metrics.json"


def make_pipeline(estimator) -> Pipeline:
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    transform = ColumnTransformer([
        ("numeric", numeric, [*NUMERIC, "previously_contacted"]),
        ("categorical", categorical, list(CATEGORICAL)),
    ])
    return Pipeline([("preprocess", transform), ("classifier", estimator)])


def choose_threshold(y_true, probabilities) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(
        precision[:-1] + recall[:-1], 1e-12
    )
    return float(thresholds[np.argmax(f1)])


def scores(y_true, probabilities, threshold: float) -> dict:
    predictions = probabilities >= threshold
    return {
        "average_precision": round(float(average_precision_score(y_true, probabilities)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, probabilities)), 4),
        "precision": round(float(precision_score(y_true, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, predictions, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, predictions, zero_division=0)), 4),
    }


def train(data_path: Path = DATA_PATH, artifact_path: Path = ARTIFACT_PATH,
          report_path: Path = REPORT_PATH) -> dict:
    if not data_path.exists():
        download_data(data_path)
    x, y = load_data(data_path)
    if len(x) < 100 or y.nunique() != 2:
        raise ValueError("Training requires at least 100 rows and both target classes")

    # 70/15/15 stratified split. The raw dataset has no year field for a true time split.
    x_train, x_holdout, y_train, y_holdout = train_test_split(
        x, y, test_size=0.30, random_state=42, stratify=y
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_holdout, y_holdout, test_size=0.50, random_state=42, stratify=y_holdout
    )
    candidates = {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=120, max_leaf_nodes=15, random_state=42
        ),
    }
    results = {}
    trained = {}
    for name, estimator in candidates.items():
        pipeline = make_pipeline(estimator)
        pipeline.fit(x_train, y_train)
        probability = pipeline.predict_proba(x_val)[:, 1]
        threshold = choose_threshold(y_val, probability)
        results[name] = {
            "threshold": round(threshold, 6),
            "validation": scores(y_val, probability, threshold),
        }
        trained[name] = pipeline

    winner = max(results, key=lambda name: results[name]["validation"]["average_precision"])
    threshold = results[winner]["threshold"]
    test_probabilities = trained[winner].predict_proba(x_test)[:, 1]
    report = {
        "dataset": "UCI Bank Marketing / bank-additional-full.csv",
        "source": "https://archive.ics.uci.edu/dataset/222/bank+marketing",
        "rows": len(x),
        "positive_rate": round(float(y.mean()), 4),
        "split": {"train": len(x_train), "validation": len(x_val), "test": len(x_test)},
        "leakage_control": "Excluded call duration; all preprocessors fitted on train only.",
        "selection_metric": "validation average_precision",
        "candidates": results,
        "winner": winner,
        "test": scores(y_test, test_probabilities, threshold),
    }
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temp_artifact = artifact_path.with_suffix(".tmp")
    joblib.dump({"pipeline": trained[winner], "threshold": threshold,
                 "model_name": winner}, temp_artifact)
    temp_artifact.replace(artifact_path)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--artifact", type=Path, default=ARTIFACT_PATH)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    args = parser.parse_args()
    print(json.dumps(train(args.data, args.artifact, args.report), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
