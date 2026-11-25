from __future__ import annotations

import io
from unittest.mock import MagicMock

from conftest import API_ROOT, FRAUD_FEATURES, LEGIT_FEATURES


AUTH_HEADERS = {"X-API-Key": "test-secret-key"}


def test_health_endpoint_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["inference_backend"] == "local"
    assert "X-Process-Time-Ms" in response.headers


def test_model_info_returns_29_features_and_threshold(client):
    response = client.get("/model-info")
    assert response.status_code == 200
    body = response.json()
    assert body["features_count"] == 29
    assert body["features"][0] == "V1"
    assert body["features"][-1] == "Amount"
    assert 0.0 < body["threshold"] < 0.05


def test_predict_requires_valid_api_key(client):
    payload = {"transaction_id": "TX-NO-AUTH", "features": LEGIT_FEATURES}
    assert client.post("/v1/predict", json=payload).status_code == 401
    assert (
        client.post(
            "/v1/predict", json=payload, headers={"X-API-Key": "wrong-key"}
        ).status_code
        == 401
    )


def test_predict_rejects_missing_or_extra_features(client):
    incomplete = dict(LEGIT_FEATURES)
    incomplete.pop("Amount")
    res_missing = client.post(
        "/v1/predict",
        json={"transaction_id": "TX-MISSING", "features": incomplete},
        headers=AUTH_HEADERS,
    )
    assert res_missing.status_code == 422
    assert "missing_features" in res_missing.json()["detail"]

    extra = dict(LEGIT_FEATURES)
    extra["Time"] = 10.0
    res_extra = client.post(
        "/v1/predict",
        json={"transaction_id": "TX-EXTRA", "features": extra},
        headers=AUTH_HEADERS,
    )
    assert res_extra.status_code == 422
    assert "unknown_features" in res_extra.json()["detail"]


def test_predict_rejects_boolean_feature_value(client):
    invalid = dict(LEGIT_FEATURES)
    invalid["Amount"] = True
    response = client.post(
        "/v1/predict",
        json={"transaction_id": "TX-BOOL", "features": invalid},
        headers=AUTH_HEADERS,
    )
    assert response.status_code == 422


def test_predict_legitimate_transaction_authorizes(client):
    response = client.post(
        "/v1/predict",
        json={"transaction_id": "TX-LEGITIMA-001", "features": LEGIT_FEATURES},
        headers=AUTH_HEADERS,
    )
    assert response.status_code == 200
    body = response.json()
    assert body["transaction_id"] == "TX-LEGITIMA-001"
    assert body["predicted_class"] == 0
    assert body["decision"] == "AUTORIZAR"
    assert body["fraud_probability"] < body["threshold"]
    assert body["inference_backend"] == "local"


def test_predict_fraudulent_transaction_flags_review(client):
    response = client.post(
        "/v1/predict",
        json={"transaction_id": "TX-FRAUDE-001", "features": FRAUD_FEATURES},
        headers=AUTH_HEADERS,
    )
    assert response.status_code == 200
    body = response.json()
    assert body["transaction_id"] == "TX-FRAUDE-001"
    assert body["predicted_class"] == 1
    assert body["decision"] == "REVISAR_O_RECHAZAR"
    assert body["fraud_probability"] >= body["threshold"]


def test_predict_sagemaker_backend_invokes_endpoint_and_calibrates(monkeypatch):
    monkeypatch.setenv("USE_SAGEMAKER_ENDPOINT", "true")
    monkeypatch.setenv("SAGEMAKER_ENDPOINT_NAME", "pagoseguro-mock-endpoint")
    monkeypatch.setenv("AWS_REGION", "us-east-1")

    fake_runtime = MagicMock()
    fake_runtime.invoke_endpoint.return_value = {
        "Body": io.BytesIO(b"0.9999252557754517\n")
    }
    monkeypatch.setattr("app.model_service.boto3.client", lambda *_a, **_kw: fake_runtime)

    from app.model_service import FraudModelService

    service = FraudModelService(API_ROOT / "artifacts")
    result = service.predict("TX-SM-001", FRAUD_FEATURES)

    assert result["inference_backend"] == "sagemaker"
    assert result["predicted_class"] == 1
    assert result["decision"] == "REVISAR_O_RECHAZAR"
    fake_runtime.invoke_endpoint.assert_called_once()
    call_kwargs = fake_runtime.invoke_endpoint.call_args.kwargs
    assert call_kwargs["EndpointName"] == "pagoseguro-mock-endpoint"
    assert call_kwargs["ContentType"] == "text/csv"


def test_explain_endpoint_returns_shap_top_features(client):
    response = client.post(
        "/v1/explain",
        json={
            "transaction_id": "TX-EXPLAIN-001",
            "features": FRAUD_FEATURES,
            "top_n": 5,
        },
        headers=AUTH_HEADERS,
    )
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"]["decision"] == "REVISAR_O_RECHAZAR"
    explanation = body["explanation"]
    assert explanation["method"] == "SHAP TreeExplainer"
    assert len(explanation["top_features"]) == 5
    assert explanation["additivity_error"] < 1e-3


def test_predict_batch_and_limits(client):
    response = client.post(
        "/v1/predict/batch",
        json={
            "transactions": [
                {"transaction_id": "TX-B1", "features": LEGIT_FEATURES},
                {"transaction_id": "TX-B2", "features": FRAUD_FEATURES},
            ]
        },
        headers=AUTH_HEADERS,
    )
    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 2
    assert body["results"][0]["decision"] == "AUTORIZAR"
    assert body["results"][1]["decision"] == "REVISAR_O_RECHAZAR"

    empty_res = client.post(
        "/v1/predict/batch", json={"transactions": []}, headers=AUTH_HEADERS
    )
    assert empty_res.status_code == 422


def test_kafka_publisher_and_worker_process_event(monkeypatch):
    from app.event_publisher import KafkaEventPublisher
    from app.model_service import FraudModelService
    from worker.audit_worker import process_event

    monkeypatch.setenv("USE_SAGEMAKER_ENDPOINT", "false")
    monkeypatch.setenv("SAGEMAKER_ENDPOINT_NAME", "")

    sent_events = []

    class FakeFuture:
        def add_errback(self, _cb):
            return None

    class FakeProducer:
        def send(self, topic, key, value):
            sent_events.append({"topic": topic, "key": key, "value": value})
            return FakeFuture()

        def flush(self, timeout=2):
            return None

        def close(self, timeout=2):
            return None

    monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
    publisher = KafkaEventPublisher()
    publisher._producer = FakeProducer()

    prediction = {
        "transaction_id": "TX-EVT-001",
        "fraud_probability": 0.98,
        "raw_fraud_probability": 0.99,
        "predicted_class": 1,
        "decision": "REVISAR_O_RECHAZAR",
        "threshold": 0.01333367,
        "inference_backend": "local",
    }
    status = publisher.publish_prediction("TX-EVT-001", FRAUD_FEATURES, prediction)
    assert status == "queued"
    assert len(sent_events) == 1

    saved_records = []

    class FakeRepository:
        def save_event(self, event, explanation, *, status="completed", error_message=None):
            saved_records.append(
                {
                    "event_id": event["event_id"],
                    "status": status,
                    "explanation": explanation,
                    "error_message": error_message,
                }
            )

    model_service = FraudModelService(API_ROOT / "artifacts")
    process_event(sent_events[0]["value"], model_service, FakeRepository())
    assert len(saved_records) == 1
    assert saved_records[0]["status"] == "completed"
    assert len(saved_records[0]["explanation"]["top_features"]) == 10


def test_metrics_endpoint_and_rate_limiting(client, monkeypatch):
    monkeypatch.setenv("RATE_LIMIT", "2/minute")
    payload = {"transaction_id": "TX-RL", "features": LEGIT_FEATURES}
    assert client.post("/v1/predict", json=payload, headers=AUTH_HEADERS).status_code == 200
    assert client.post("/v1/predict", json=payload, headers=AUTH_HEADERS).status_code == 200
    third = client.post("/v1/predict", json=payload, headers=AUTH_HEADERS)
    assert third.status_code == 429

    metrics_res = client.get("/metrics")
    assert metrics_res.status_code == 200
    assert "pagoseguro_http_requests_total" in metrics_res.text
    assert "pagoseguro_http_request_duration_ms" in metrics_res.text
    assert "pagoseguro_predictions_decision_total" in metrics_res.text
