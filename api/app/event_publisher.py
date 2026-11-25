from __future__ import annotations

import json
import logging
import os
import threading
import uuid
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger("pagoseguro.events")


class KafkaEventPublisher:
    """Publica eventos de prediccion sin esperar al procesamiento SHAP."""

    def __init__(self) -> None:
        self.bootstrap_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "").strip()
        self.topic = os.getenv("KAFKA_TOPIC", "pagoseguro.transaction.predicted")
        self._producer = None
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return bool(self.bootstrap_servers)

    def _get_producer(self):
        if not self.enabled:
            return None
        if self._producer is None:
            with self._lock:
                if self._producer is None:
                    from kafka import KafkaProducer

                    self._producer = KafkaProducer(
                        bootstrap_servers=[s.strip() for s in self.bootstrap_servers.split(",") if s.strip()],
                        key_serializer=lambda value: value.encode("utf-8"),
                        value_serializer=lambda value: json.dumps(
                            value, ensure_ascii=False, allow_nan=False
                        ).encode("utf-8"),
                        acks="all",
                        retries=3,
                        linger_ms=5,
                        max_block_ms=700,
                        request_timeout_ms=2000,
                    )
        return self._producer

    def publish_prediction(
        self,
        transaction_id: str | None,
        features: dict[str, Any],
        prediction: dict[str, Any],
    ) -> str:
        if not self.enabled:
            return "disabled"
        event_id = str(uuid.uuid4())
        event = {
            "schema_version": 1,
            "event_id": event_id,
            "event_type": "transaction.predicted",
            "occurred_at": datetime.now(timezone.utc).isoformat(),
            "source": "pagoseguro-api",
            "transaction_id": transaction_id,
            "features": features,
            "prediction": prediction,
        }
        try:
            producer = self._get_producer()
            if producer is None:
                return "disabled"
            future = producer.send(
                self.topic,
                key=transaction_id or event_id,
                value=event,
            )
            future.add_errback(
                lambda exc: logger.error(
                    "Fallo posterior al encolar evento %s: %s", event_id, exc
                )
            )
            return "queued"
        except Exception as exc:
            logger.warning("No se pudo encolar evento de auditoria: %s", exc)
            return "unavailable"

    def close(self) -> None:
        if self._producer is not None:
            try:
                self._producer.flush(timeout=2)
                self._producer.close(timeout=2)
            except Exception:
                logger.exception("Error cerrando productor Kafka")
