from typing import Any

import numpy as np
from fastapi import Body, FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.modeling import DATASET_PATH, load_artifacts, load_dataset, train_and_evaluate


app = FastAPI(title="Local SVM AI Server", version="2.0.0")
model: Any = None
feature_names: list[str] = []
metrics: dict[str, Any] = {}


class TrainingRequest(BaseModel):
    features: list[list[float]] = Field(..., min_length=2)
    labels: list[str | int | float] = Field(..., min_length=2)


class PredictionRequest(BaseModel):
    """Dữ liệu đầu vào theo thứ tự: weight, size, moisture, sugar_level, firmness, color_score."""

    features: list[list[float]] = Field(
        ...,
        min_length=1,
        description=(
            "Danh sách nông sản cần phân loại. Mỗi dòng phải có đúng 6 giá trị "
            "theo thứ tự: weight, size, moisture, sugar_level, firmness, color_score."
        ),
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "features": [[212.07, 8.51, 92.49, 13.84, 8.24, 9.16]],
                }
            ]
        }
    }


def load_saved_model() -> None:
    global model, feature_names, metrics
    artifacts = load_artifacts()
    if artifacts is not None:
        model, feature_names, metrics = artifacts


load_saved_model()


@app.get("/health")
def health() -> dict[str, bool]:
    return {"status": True, "model_ready": model is not None}


@app.get("/metrics")
def get_metrics() -> dict[str, Any]:
    if not metrics:
        raise HTTPException(status_code=404, detail="model metrics are not available")
    return metrics


@app.post("/train")
def train(request: TrainingRequest) -> dict[str, Any]:
    global model, feature_names, metrics
    if len(request.features) != len(request.labels):
        raise HTTPException(status_code=422, detail="features and labels must have the same length")
    if not request.features or any(len(row) != len(request.features[0]) for row in request.features):
        raise HTTPException(status_code=422, detail="all feature rows must have the same width")
    if len(set(request.labels)) < 2:
        raise HTTPException(status_code=422, detail="SVM training requires at least two classes")

    values = np.asarray(request.features, dtype=float)
    labels = np.asarray(request.labels)
    feature_names = [f"feature_{index + 1}" for index in range(values.shape[1])]
    metrics = train_and_evaluate(values, labels, feature_names)
    loaded = load_artifacts()
    if loaded is None:
        raise HTTPException(status_code=500, detail="model artifact was not created")
    model, feature_names, _ = loaded
    return {"trained": True, **metrics}


@app.post("/train-dataset")
def train_dataset() -> dict[str, Any]:
    global model, feature_names, metrics
    try:
        features, labels, names = load_dataset(DATASET_PATH)
        metrics = train_and_evaluate(features, labels, names)
        loaded = load_artifacts()
        if loaded is None:
            raise ValueError("model artifact was not created")
        model, feature_names, _ = loaded
    except (OSError, ValueError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return {"trained": True, "dataset": DATASET_PATH.name, "feature_names": feature_names, **metrics}


@app.post(
    "/predict",
    summary="Phân loại chất lượng nông sản",
    description=(
        "Nhập 6 giá trị theo đúng thứ tự: "
    ),
)
def predict(
    request: PredictionRequest = Body(
        ...,
        openapi_examples={
            "mau_nong_san": {
                "summary": "Mẫu dữ liệu nông sản",
                "description": "Một dòng gồm đúng 6 giá trị theo thứ tự feature được mô tả "
                "weight (khối lượng), size (kích thước), moisture (độ ẩm),"
                "sugar_level (độ ngọt), firmness (độ cứng), color_score (điểm màu sắc)." ,
                "value": {
                    "features": [[212.07, 8.51, 92.49, 13.84, 8.24, 9.16]],
                },
            }
        },
    ),
) -> dict[str, list[Any]]:
    if model is None:
        raise HTTPException(status_code=409, detail="train the model before predicting")
    try:
        values = np.asarray(request.features, dtype=float)
        if values.shape[1] != len(feature_names):
            raise ValueError(f"expected {len(feature_names)} features: {', '.join(feature_names)}")
        predictions = model.predict(values).tolist()
        probabilities = model.predict_proba(values).tolist()
    except (IndexError, ValueError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    quality_names = {0: "Thấp", 1: "Trung bình", 2: "Cao"}
    labels = [quality_names.get(int(label), str(label)) for label in predictions]
    confidences = [max(row) for row in probabilities]
    conclusions = [f"Nông sản được phân loại: {label}" for label in labels]
    return {
        "predictions": predictions,
        "quality_labels": labels,
        "probabilities": probabilities,
        "confidence": confidences,
        "conclusions": conclusions,
    }
