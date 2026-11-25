import hmac
import json
import logging
import os
import threading
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.model_service import FraudModelService
from app.event_publisher import KafkaEventPublisher
from app.audit_repository import AuditRepository
from app.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    ExplainRequest,
    ExplainResponse,
    PredictionRequest,
    PredictionResponse,
)

load_dotenv()
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("pagoseguro.api")

ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", "artifacts"))
API_KEY = os.getenv("API_KEY", "")
RATE_LIMIT = os.getenv("RATE_LIMIT", "60/minute")
CORS_ORIGINS = [item.strip() for item in os.getenv("CORS_ORIGINS", "").split(",") if item.strip()]
limiter = Limiter(key_func=get_remote_address)


def get_rate_limit(*_args: Any, **_kwargs: Any) -> str:
    """Lee RATE_LIMIT dinamicamente para permitir ajustes en pruebas de carga."""
    return os.getenv("RATE_LIMIT", RATE_LIMIT)


class RuntimeMetrics:
    """Recolector ligero de latencia y conteos para observabilidad Prometheus."""

    def __init__(self, window_size: int = 2000) -> None:
        self._lock = threading.Lock()
        self._requests_total: dict[tuple[str, str, int], int] = defaultdict(int)
        self._latencies_ms: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=window_size)
        )
        self._decisions_total: dict[str, int] = defaultdict(int)

    def record_http(self, method: str, path: str, status_code: int, duration_ms: float) -> None:
        with self._lock:
            self._requests_total[(method, path, status_code)] += 1
            self._latencies_ms[path].append(duration_ms)

    def record_decision(self, decision: str) -> None:
        with self._lock:
            self._decisions_total[decision] += 1

    @staticmethod
    def _percentile(sorted_values: list[float], pct: float) -> float:
        if not sorted_values:
            return 0.0
        idx = (len(sorted_values) - 1) * pct
        low = int(idx)
        high = min(low + 1, len(sorted_values) - 1)
        weight = idx - low
        return sorted_values[low] * (1.0 - weight) + sorted_values[high] * weight

    def render_prometheus(self) -> str:
        lines = [
            "# HELP pagoseguro_http_requests_total Total de solicitudes HTTP procesadas.",
            "# TYPE pagoseguro_http_requests_total counter",
        ]
        with self._lock:
            for (method, path, status_code), count in sorted(self._requests_total.items()):
                lines.append(
                    f'pagoseguro_http_requests_total{{method="{method}",path="{path}",status="{status_code}"}} {count}'
                )
            lines.extend(
                [
                    "# HELP pagoseguro_http_request_duration_ms Latencia HTTP por ruta en milisegundos.",
                    "# TYPE pagoseguro_http_request_duration_ms summary",
                ]
            )
            for path, values in sorted(self._latencies_ms.items()):
                ordered = sorted(values)
                if not ordered:
                    continue
                for q_label, q_val in (("0.5", 0.5), ("0.95", 0.95), ("0.99", 0.99)):
                    p_val = self._percentile(ordered, q_val)
                    lines.append(
                        f'pagoseguro_http_request_duration_ms{{path="{path}",quantile="{q_label}"}} {p_val:.3f}'
                    )
                lines.append(
                    f'pagoseguro_http_request_duration_ms_count{{path="{path}"}} {len(ordered)}'
                )
            lines.extend(
                [
                    "# HELP pagoseguro_predictions_decision_total Predicciones emitidas por decision.",
                    "# TYPE pagoseguro_predictions_decision_total counter",
                ]
            )
            for decision, count in sorted(self._decisions_total.items()):
                lines.append(f'pagoseguro_predictions_decision_total{{decision="{decision}"}} {count}')
        return "\n".join(lines) + "\n"


runtime_metrics = RuntimeMetrics()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Intenta cargar los artefactos al iniciar la API."""
    app.state.event_publisher = KafkaEventPublisher()
    try:
        app.state.model_service = FraudModelService(ARTIFACTS_DIR)
        app.state.model_error = None
        logger.info("Modelo PagoSeguro cargado desde %s", ARTIFACTS_DIR.resolve())
    except Exception as exc:
        app.state.model_service = None
        app.state.model_error = str(exc)
        logger.exception("No se pudo cargar el modelo")
    yield
    publisher = getattr(app.state, "event_publisher", None)
    if publisher is not None:
        publisher.close()


app = FastAPI(
    title="PagoSeguro - API de deteccion de fraude",
    description=(
        "API REST para evaluar transacciones con XGBoost. Publica eventos de prediccion a Kafka para "
        "explicabilidad SHAP asincrona y auditoria PostgreSQL. Uso academico y experimental."
    ),
    version="1.2.0",
    lifespan=lifespan,
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

if CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "X-API-Key"],
    )


@app.middleware("http")
async def track_latency_middleware(request: Request, call_next):
    started = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - started) * 1000.0
    runtime_metrics.record_http(
        request.method, request.url.path, response.status_code, duration_ms
    )
    response.headers["X-Process-Time-Ms"] = f"{duration_ms:.3f}"
    return response


def verify_api_key(x_api_key: str = Header(default="", alias="X-API-Key")) -> None:
    """Protege los endpoints de prediccion mediante una API key."""
    configured_key = os.getenv("API_KEY", "")
    if not configured_key:
        raise HTTPException(
            status_code=503,
            detail="API_KEY no esta configurada en el servidor.",
        )
    if not hmac.compare_digest(x_api_key, configured_key):
        raise HTTPException(status_code=401, detail="API key invalida.")


def publish_audit_event(
    request: Request,
    transaction_id: str | None,
    features: dict[str, Any],
    prediction: dict[str, Any],
) -> str:
    publisher = getattr(request.app.state, "event_publisher", None)
    if publisher is None:
        return "disabled"
    return publisher.publish_prediction(transaction_id, features, prediction)


def get_model_service(request: Request) -> FraudModelService:
    service = getattr(request.app.state, "model_service", None)
    if service is None:
        reason = getattr(request.app.state, "model_error", "Modelo no disponible")
        raise HTTPException(status_code=503, detail=f"Modelo no disponible: {reason}")
    return service


def run_prediction(service: FraudModelService, payload: PredictionRequest) -> dict[str, Any]:
    try:
        return service.predict(payload.transaction_id, payload.features)
    except ValueError as exc:
        message = str(exc)
        try:
            parsed = json.loads(message)
            raise HTTPException(status_code=422, detail=parsed) from exc
        except json.JSONDecodeError:
            raise HTTPException(status_code=422, detail=message) from exc
    except RuntimeError as exc:
        logger.exception("Fallo la inferencia del modelo")
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def run_explanation(
    service: FraudModelService, payload: ExplainRequest
) -> dict[str, Any]:
    try:
        return service.explain(
            payload.transaction_id, payload.features, top_n=payload.top_n
        )
    except ValueError as exc:
        message = str(exc)
        try:
            parsed = json.loads(message)
            raise HTTPException(status_code=422, detail=parsed) from exc
        except json.JSONDecodeError:
            raise HTTPException(status_code=422, detail=message) from exc
    except RuntimeError as exc:
        logger.exception("No se pudo generar la explicacion SHAP")
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/", tags=["General"])
def root() -> dict[str, str]:
    return {"service": "PagoSeguro Fraud Detection API", "docs": "/docs", "health": "/health"}


@app.get("/health", tags=["General"])
def health(request: Request) -> dict[str, Any]:
    service = getattr(request.app.state, "model_service", None)
    if service is None:
        raise HTTPException(
            status_code=503,
            detail={
                "status": "degraded",
                "model_loaded": False,
                "reason": getattr(request.app.state, "model_error", "Modelo no disponible"),
            },
        )
    publisher = getattr(request.app.state, "event_publisher", None)
    return {
        "status": "ok",
        "model_loaded": True,
        "kafka_publisher_enabled": bool(publisher and publisher.enabled),
        "audit_database_configured": bool(os.getenv("DATABASE_URL", "").strip()),
        "inference_backend": service.inference_backend,
        "sagemaker_endpoint_configured": bool(service.sagemaker_endpoint_name),
    }


@app.get("/metrics", tags=["Observabilidad"], summary="Metricas Prometheus de latencia y decisiones")
def prometheus_metrics() -> PlainTextResponse:
    return PlainTextResponse(
        runtime_metrics.render_prometheus(),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )


@app.get("/model-info", tags=["Modelo"])
def model_info(service: FraudModelService = Depends(get_model_service)) -> dict[str, Any]:
    return {
        "project": service.metadata.get("project", "PagoSeguro"),
        "model_type": service.metadata.get("model_type", "XGBoost"),
        "features_count": len(service.features),
        "features": service.features,
        "threshold": service.threshold,
        "calibration_method": service.metadata.get("calibration_method", "No informado"),
        "inference_backend": service.inference_backend,
        "sagemaker_endpoint_name": service.sagemaker_endpoint_name or None,
    }


@app.post(
    "/v1/predict",
    response_model=PredictionResponse,
    tags=["Predicciones"],
    summary="Evaluar una transaccion",
)
@limiter.limit(get_rate_limit)
def predict(
    request: Request,
    payload: PredictionRequest,
    _api_key: None = Depends(verify_api_key),
    service: FraudModelService = Depends(get_model_service),
) -> dict[str, Any]:
    """Devuelve la probabilidad estimada y la decision experimental."""
    result = run_prediction(service, payload)
    runtime_metrics.record_decision(result["decision"])
    result["audit_event_status"] = publish_audit_event(
        request, payload.transaction_id, payload.features, result
    )
    return result


@app.post(
    "/v1/explain",
    response_model=ExplainResponse,
    tags=["Explicabilidad"],
    summary="Explicar una prediccion con SHAP",
)
@limiter.limit(get_rate_limit)
def explain_prediction(
    request: Request,
    payload: ExplainRequest,
    _api_key: None = Depends(verify_api_key),
    service: FraudModelService = Depends(get_model_service),
) -> dict[str, Any]:
    """Genera una explicacion local sin agregar SHAP al flujo normal /predict."""
    return run_explanation(service, payload)


@app.post(
    "/v1/predict/batch",
    response_model=BatchPredictionResponse,
    tags=["Predicciones"],
    summary="Evaluar varias transacciones",
)
@limiter.limit(get_rate_limit)
def predict_batch(
    request: Request,
    payload: BatchPredictionRequest,
    _api_key: None = Depends(verify_api_key),
    service: FraudModelService = Depends(get_model_service),
) -> dict[str, Any]:
    results = []
    for item in payload.transactions:
        result = run_prediction(service, item)
        runtime_metrics.record_decision(result["decision"])
        result["audit_event_status"] = publish_audit_event(
            request, item.transaction_id, item.features, result
        )
        results.append(result)
    return {"total": len(results), "results": results}


@app.get(
    "/v1/audit/recent",
    tags=["Auditoria"],
    summary="Consultar auditorias procesadas por el worker",
)
def recent_audit_events(
    limit: int = 50,
    _api_key: None = Depends(verify_api_key),
) -> dict[str, Any]:
    database_url = os.getenv("DATABASE_URL", "").strip()
    if not database_url:
        raise HTTPException(status_code=503, detail="DATABASE_URL no esta configurada.")
    try:
        repository = AuditRepository(database_url)
        events = repository.recent_events(limit)
        return {"total": len(events), "events": events}
    except Exception as exc:
        logger.exception("No se pudo consultar la auditoria distribuida")
        raise HTTPException(
            status_code=503,
            detail=f"Auditoria no disponible o tabla aun no creada: {type(exc).__name__}: {exc}",
        ) from exc
