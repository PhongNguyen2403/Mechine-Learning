from fastapi.testclient import TestClient

from app.modeling import METRICS_PATH, MODEL_PATH

MODEL_PATH.unlink(missing_ok=True)
METRICS_PATH.unlink(missing_ok=True)

from app.main import app


client = TestClient(app)


def test_health_starts_without_model():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] is True


def test_predict_requires_training():
    response = client.post("/predict", json={"features:": [[0.1, 0.2]]})

    assert response.status_code == 409


def test_train_and_predict():
    response = client.post(
        "/train",
        json={
            "features": [[0, 0], [0, 1], [1, 0], [1, 1]],
            "labels": ["low", "high", "high", "low"],
            "kernel": "rbf",
        },
    )

    assert response.status_code == 200
    assert response.json()["trained"] is True

    response = client.post("/predict", json={"features": [[0, 0], [1, 1]]})

    assert response.status_code == 200
    assert len(response.json()["predictions"]) == 2
    assert len(response.json()["probabilities"]) == 2


def test_train_supplied_dataset_and_predict():
    response = client.post("/train-dataset")

    assert response.status_code == 200
    assert response.json()["dataset"] == "agricultural_product_quality.csv"
    assert response.json()["feature_names"] == [
        "weight",
        "size",
        "moisture",
        "sugar_level",
        "firmness",
        "color_score",
    ]

    response = client.post(
        "/predict",
        json={"features": [[212.07, 8.51, 92.49, 13.84, 8.24, 9.16]]},
    )

    assert response.status_code == 200
    assert response.json()["predictions"] == [2]


def test_train_rejects_mismatched_data():
    response = client.post(
        "/train",
        json={"features": [[0, 0], [1, 1]], "labels": ["one"]},
    )

    assert response.status_code == 422