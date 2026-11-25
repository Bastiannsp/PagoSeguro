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
);
CREATE INDEX IF NOT EXISTS idx_audit_events_processed_at ON audit_events (processed_at DESC);
CREATE INDEX IF NOT EXISTS idx_audit_events_transaction_id ON audit_events (transaction_id);
