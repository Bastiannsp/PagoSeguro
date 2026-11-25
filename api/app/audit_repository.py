from __future__ import annotations

import os
from typing import Any

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb


CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS audit_events (
    event_id UUID PRIMARY KEY,
    transaction_id TEXT,
    event_type TEXT NOT NULL,
    occurred_at TIMESTAMPTZ NOT NULL,
    processed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    source TEXT NOT NULL,
    fraud_probability DOUBLE PRECISION,
    raw_fraud_probability DOUBLE PRECISION,
    predicted_class SMALLINT,
    decision TEXT,
    threshold DOUBLE PRECISION,
    features JSONB NOT NULL,
    explanation JSONB,
    processing_status TEXT NOT NULL DEFAULT 'completed',
    error_message TEXT
)
"""



class AuditRepository:
    def __init__(self, database_url: str | None = None) -> None:
        self.database_url = database_url or os.getenv("DATABASE_URL", "").strip()
        if not self.database_url:
            raise ValueError("DATABASE_URL no esta configurada.")

    def ensure_schema(self) -> None:
        with psycopg.connect(self.database_url) as conn:
            with conn.cursor() as cur:
                cur.execute(CREATE_TABLE_SQL)
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_audit_events_processed_at "
                    "ON audit_events (processed_at DESC)"
                )
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_audit_events_transaction_id "
                    "ON audit_events (transaction_id)"
                )

    def save_event(
        self,
        event: dict[str, Any],
        explanation: dict[str, Any] | None,
        *,
        status: str = "completed",
        error_message: str | None = None,
    ) -> None:
        prediction = event.get("prediction") or {}
        features = event.get("features") or {}
        with psycopg.connect(self.database_url) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO audit_events (
                        event_id, transaction_id, event_type, occurred_at, source,
                        fraud_probability, raw_fraud_probability, predicted_class,
                        decision, threshold, features, explanation,
                        processing_status, error_message
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                        %s, %s, %s, %s
                    )
                    ON CONFLICT (event_id) DO NOTHING
                    """,
                    (
                        event["event_id"],
                        event.get("transaction_id"),
                        event.get("event_type", "transaction.predicted"),
                        event["occurred_at"],
                        event.get("source", "unknown"),
                        prediction.get("fraud_probability"),
                        prediction.get("raw_fraud_probability"),
                        prediction.get("predicted_class"),
                        prediction.get("decision"),
                        prediction.get("threshold"),
                        Jsonb(features),
                        Jsonb(explanation) if explanation is not None else None,
                        status,
                        error_message[:1000] if error_message else None,
                    ),
                )

    def recent_events(self, limit: int = 50) -> list[dict[str, Any]]:
        safe_limit = max(1, min(int(limit), 250))
        with psycopg.connect(self.database_url, row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT event_id::text AS event_id, transaction_id, event_type,
                           occurred_at, processed_at, source, fraud_probability,
                           raw_fraud_probability, predicted_class, decision, threshold,
                           explanation, processing_status, error_message
                    FROM audit_events
                    ORDER BY processed_at DESC
                    LIMIT %s
                    """,
                    (safe_limit,),
                )
                rows = cur.fetchall()
        return [dict(row) for row in rows]
