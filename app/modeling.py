import csv
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


ROOT_DIR = Path(__file__).parents[1]
DATASET_PATH = ROOT_DIR / "data" / "agricultural_product_quality.csv"
ARTIFACT_DIR = ROOT_DIR / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "svm_model.joblib"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"


def load_dataset(path: Path = DATASET_PATH) -> tuple[np.ndarray, np.ndarray, list[str]]:
    with path.open(newline="", encoding="utf-8") as dataset_file:
        rows = list(csv.DictReader(dataset_file))

    if not rows:
        raise ValueError("dataset is empty")

    feature_names = [name for name in rows[0] if name != "quality"]
    features = np.asarray([[float(row[name]) for name in feature_names] for row in rows], dtype=float)
    labels = np.asarray([int(row["quality"]) for row in rows])
    return features, labels, feature_names


def build_model() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", SVC(probability=True)),
    ])


def train_and_evaluate(
    features: np.ndarray,
    labels: np.ndarray,
    feature_names: list[str],
) -> dict[str, Any]:
    if len(features) < 10 or np.min(np.unique(labels, return_counts=True)[1]) < 2:
        model = build_model().set_params(classifier__kernel="rbf", classifier__C=1.0, classifier__gamma="scale")
        model.fit(features, labels)
        metrics = {
            "samples": int(len(features)),
            "features": feature_names,
            "classes": [int(value) if isinstance(value, (int, np.integer)) else str(value) for value in sorted(np.unique(labels))],
            "train_samples": int(len(features)),
            "test_samples": 0,
            "accuracy": None,
            "precision_macro": None,
            "recall_macro": None,
            "f1_macro": None,
            "confusion_matrix": [],
            "classification_report": {},
            "support_vectors": 0,
            "best_params": {"classifier__kernel": "rbf", "classifier__C": 1.0, "classifier__gamma": "scale"},
            "cv_best_accuracy": None,
        }
        save_artifacts(model, feature_names, metrics)
        return metrics

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )
    search = GridSearchCV(
        build_model(),
        {
            "classifier__kernel": ["rbf", "linear"],
            "classifier__C": [0.1, 1.0, 10.0],
            "classifier__gamma": ["scale", "auto"],
        },
        cv=3,
        scoring="accuracy",
        n_jobs=-1,
    )
    search.fit(x_train, y_train)
    predictions = search.predict(x_test)
    metrics = {
        "samples": int(len(features)),
        "features": feature_names,
        "classes": [int(value) if isinstance(value, (int, np.integer)) else str(value) for value in sorted(np.unique(labels))],
        "train_samples": int(len(x_train)),
        "test_samples": int(len(x_test)),
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "precision_macro": round(float(precision_score(y_test, predictions, average="macro", zero_division=0)), 4),
        "recall_macro": round(float(recall_score(y_test, predictions, average="macro", zero_division=0)), 4),
        "f1_macro": round(float(f1_score(y_test, predictions, average="macro", zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "classification_report": classification_report(y_test, predictions, output_dict=True, zero_division=0),
        "support_vectors": int(len(search.best_estimator_.named_steps["classifier"].support_vectors_)),
        "best_params": search.best_params_,
        "cv_best_accuracy": round(float(search.best_score_), 4),
    }
    save_artifacts(search.best_estimator_, feature_names, metrics)
    return metrics


def save_artifacts(model: Pipeline, feature_names: list[str], metrics: dict[str, Any]) -> None:
    ARTIFACT_DIR.mkdir(exist_ok=True)
    joblib.dump({"model": model, "feature_names": feature_names}, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def load_artifacts() -> tuple[Pipeline, list[str], dict[str, Any]] | None:
    if not MODEL_PATH.exists():
        return None
    bundle = joblib.load(MODEL_PATH)
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8")) if METRICS_PATH.exists() else {}
    return bundle["model"], bundle["feature_names"], metrics