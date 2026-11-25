from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


def _connect(db_path: str | Path) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=10)
    connection.row_factory = sqlite3.Row
    return connection


def init_db(db_path: str | Path = "data/audit.db") -> None:
    with _connect(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS audit_events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                transaction_id TEXT,
                source TEXT NOT NULL,
                operation TEXT NOT NULL,
                fraud_probability REAL NOT NULL,
                raw_fraud_probability REAL,
                predicted_class INTEGER NOT NULL,
                decision TEXT NOT NULL,
                threshold REAL NOT NULL,
                features_json TEXT NOT NULL,
                explanation_json TEXT
            )
            """
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_audit_created_at ON audit_events(created_at DESC)"
        )


def save_event(
    *,
    db_path: str | Path,
    transaction_id: str,
    source: str,
    operation: str,
    features: dict[str, Any],
    prediction: dict[str, Any],
    explanation: dict[str, Any] | None = None,
) -> None:
    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with _connect(db_path) as connection:
        connection.execute(
            """
            INSERT INTO audit_events (
                created_at, transaction_id, source, operation,
                fraud_probability, raw_fraud_probability, predicted_class,
                decision, threshold, features_json, explanation_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                created_at,
                transaction_id,
                source,
                operation,
                float(prediction["fraud_probability"]),
                float(prediction.get("raw_fraud_probability", 0.0)),
                int(prediction["predicted_class"]),
                str(prediction["decision"]),
                float(prediction["threshold"]),
                json.dumps(features, ensure_ascii=False, allow_nan=False),
                json.dumps(explanation, ensure_ascii=False, allow_nan=False) if explanation else None,
            ),
        )


def load_recent_events(
    db_path: str | Path = "data/audit.db", limit: int = 50
) -> pd.DataFrame:
    with _connect(db_path) as connection:
        rows = connection.execute(
            """
            SELECT event_id, created_at, transaction_id, source, operation,
                   fraud_probability, raw_fraud_probability, predicted_class,
                   decision, threshold
            FROM audit_events
            ORDER BY event_id DESC
            LIMIT ?
            """,
            (int(limit),),
        ).fetchall()
    return pd.DataFrame([dict(row) for row in rows])
