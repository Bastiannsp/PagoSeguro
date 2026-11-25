from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from kafka import KafkaConsumer

from app.audit_repository import AuditRepository
from app.model_service import FraudModelService

load_dotenv()
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger("pagoseguro.audit-worker")

BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
TOPIC = os.getenv("KAFKA_TOPIC", "pagoseguro.transaction.predicted")
GROUP_ID = os.getenv("KAFKA_GROUP_ID", "pagoseguro-audit-workers")
DATABASE_URL = os.getenv("DATABASE_URL", "")
ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", "/app/artifacts"))


def build_consumer() -> KafkaConsumer:
    return KafkaConsumer(
        TOPIC,
        bootstrap_servers=[item.strip() for item in BOOTSTRAP_SERVERS.split(",") if item.strip()],
        group_id=GROUP_ID,
        enable_auto_commit=False,
        auto_offset_reset="earliest",
        value_deserializer=lambda raw: json.loads(raw.decode("utf-8")),
        consumer_timeout_ms=1000,
        max_poll_records=1,
    )


def process_event(
    event: dict[str, Any],
    model_service: FraudModelService,
    repository: AuditRepository,
) -> None:
    required = {"event_id", "occurred_at", "features", "prediction"}
    missing = sorted(required - set(event))
    if missing:
        raise ValueError(f"Evento incompleto. Faltan campos: {missing}")

    transaction_id = event.get("transaction_id")
    try:
        result = model_service.explain(
            transaction_id=transaction_id,
            features=event["features"],
            top_n=10,
        )
        repository.save_event(event, result["explanation"], status="completed")
        logger.info(
            "Auditoria completada: event_id=%s transaction_id=%s decision=%s",
            event["event_id"],
            transaction_id,
            event.get("prediction", {}).get("decision"),
        )
    except Exception as exc:
        # Se persiste el fallo para que el mensaje no bloquee indefinidamente la particion.
        repository.save_event(
            event,
            None,
            status="failed",
            error_message=f"{type(exc).__name__}: {exc}",
        )
        logger.exception("Fallo SHAP para event_id=%s", event.get("event_id"))


def main() -> None:
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL no esta configurada para el worker.")
    if not BOOTSTRAP_SERVERS.strip():
        raise RuntimeError("KAFKA_BOOTSTRAP_SERVERS no esta configurada.")

    logger.info("Cargando modelo para el worker desde %s", ARTIFACTS_DIR.resolve())
    model_service = FraudModelService(ARTIFACTS_DIR)
    repository = AuditRepository(DATABASE_URL)

    while True:
        consumer = None
        try:
            repository.ensure_schema()
            consumer = build_consumer()
            logger.info("Worker escuchando topic=%s en %s", TOPIC, BOOTSTRAP_SERVERS)
            while True:
                records = consumer.poll(timeout_ms=1000, max_records=1)
                for _partition, messages in records.items():
                    for message in messages:
                        process_event(message.value, model_service, repository)
                        # El offset se confirma despues de persistir el resultado en PostgreSQL.
                        consumer.commit()
        except KeyboardInterrupt:
            logger.info("Worker detenido por el usuario.")
            break
        except Exception:
            logger.exception("Error del worker; se intentara reconectar en 4 segundos.")
            time.sleep(4)
        finally:
            if consumer is not None:
                try:
                    consumer.close()
                except Exception:
                    logger.exception("Error cerrando consumidor Kafka")


if __name__ == "__main__":
    main()
