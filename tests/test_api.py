from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_endpoint():
    payload = {
        "cpu_usage": 52.0,
        "memory_usage": 66.0,
        "disk_io": 140.0,
        "network_in": 310.0,
        "network_out": 290.0,
        "request_count": 1650,
        "hour": 10,
        "day_of_week": 3,
        "is_weekend": 0,
        "cpu_lag_1": 50.0,
        "memory_lag_1": 64.0,
        "request_count_lag_1": 1600,
        "cpu_lag_2": 48.0,
        "memory_lag_2": 63.0,
        "request_count_lag_2": 1550,
        "cpu_rolling_mean_2": 51.0,
        "memory_rolling_mean_2": 65.0,
        "cpu_rolling_mean_3": 49.5,
        "memory_rolling_mean_3": 64.3
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200

    response_json = response.json()
    assert "predicted_future_cpu_usage" in response_json
    assert "predicted_future_memory_usage" in response_json
    assert "scaling_action" in response_json