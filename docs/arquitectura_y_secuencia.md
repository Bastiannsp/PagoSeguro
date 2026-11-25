# Arquitectura y Flujo de Comunicación — PagoSeguro (EP3)

## 1. Arquitectura Desplegada (MVP EP3)

```mermaid
flowchart TB
    User["Usuario / Cliente REST / Locust"] -->|HTTPS TLS :8443| Nginx["Nginx Reverse Proxy (TLS 1.2/1.3)"]
    Nginx -->|"/"| Dash["Dashboard Streamlit (:8501)"]
    Nginx -->|"/v1/*, /docs, /health, /metrics"| API["FastAPI (:8000) + X-API-Key + SlowAPI"]
    Dash -->|HTTP REST + X-API-Key| API
    API -->|"USE_SAGEMAKER_ENDPOINT=true (text/csv)"| SM["AWS SageMaker Endpoint (XGBoost 3.2-0, ml.m5.large)"]
    API -->|"Platt Scaling + Threshold (0.013334)"| Resp["Decisión: AUTORIZAR / REVISAR_O_RECHAZAR"]
    API -->|"Evento transaction.predicted"| Kafka["Apache Kafka (:9092)"]
    Kafka -->|"Consumer Group + Commit Manual"| Worker["Worker Asíncrono SHAP (TreeExplainer)"]
    Worker -->|"INSERT ON CONFLICT DO NOTHING"| PG[("PostgreSQL (audit_events)")]
    API -->|"GET /v1/audit/recent"| PG
    SM -.->|"Logs y Métricas"| CW["AWS CloudWatch"]
```

## 2. Diagrama de Secuencia (Flujo Completo en Tiempo Real)

```mermaid
sequenceDiagram
    autonumber
    actor C as Cliente / Dashboard
    participant N as Nginx (TLS)
    participant A as FastAPI (/v1/predict)
    participant S as AWS SageMaker (XGBoost)
    participant K as Apache Kafka
    participant W as Worker SHAP
    participant DB as PostgreSQL

    C->>N: POST /v1/predict (X-API-Key + 29 features)
    N->>A: Reenvía solicitud HTTP interna
    A->>A: Valida API Key, Rate Limit y esquema Pydantic
    A->>S: invoke_endpoint(text/csv)
    S-->>A: raw_fraud_probability
    A->>A: Calibración Platt + comparación umbral (0.013334)
    A->>K: Publica evento transaction.predicted (async)
    A-->>N: 200 OK (decision + audit_event_status="queued" + X-Process-Time-Ms)
    N-->>C: Respuesta TLS al cliente

    K->>W: Entrega mensaje (max_poll_records=1)
    W->>W: Calcula SHAP TreeExplainer (top 10 features + error aditividad)
    W->>DB: INSERT audit_events (event_id PK idempotente)
    DB-->>W: Confirmación persistencia
    W->>K: consumer.commit()
    C->>A: GET /v1/audit/recent
    A->>DB: SELECT ... ORDER BY processed_at DESC
    DB-->>A: Registros con explicación SHAP
    A-->>C: Historial de auditoría distribuida
```

## 3. Mapeo MVP Implementado vs Arquitectura Target Caso 3 (Oportunidades de Optimización)

| Aspecto | Implementación Actual (EP3) | Arquitectura Target Caso 3 | Brecha y Optimización Propuesta |
| :--- | :--- | :--- | :--- |
| **Inferencia Cloud** | AWS SageMaker `ml.m5.large` (1 instancia) + `configure_autoscaling.py` | EKS / SageMaker Multi-AZ (4–12 nodos `c6i.xlarge`) con HPA/Auto-Scaling | Activar mínimo 2 instancias Multi-AZ para HA `99.99%` y mover calibrador Platt al contenedor de inferencia para evitar roundtrips. |
| **Ingestion / Streaming** | Apache Kafka 1 nodo en Docker (`PLAINTEXT`) | Amazon MSK (3 brokers Multi-AZ, TLS + SASL/IAM) | Migrar broker a Amazon MSK con replicación factor 3 y cifrado en tránsito. |
| **Feature Store** | Variables precalculadas en payload (`V1..V28`, `Amount`) | Feast + Amazon ElastiCache Redis Multi-Región (`< 10 ms`) | Incorporar Redis para calcular agregaciones en ventana móvil (ej. monto últimos 5 min). |
| **Seguridad** | `X-API-Key`, `slowapi`, TLS en Nginx (`HTTPS :8443`) | mTLS interno, KMS CMK en reposo, IAM Least Privilege, masking PAN | Terminar TLS en ALB con ACM, rotar secretos con Secrets Manager y cifrar RDS/MSK con KMS. |
| **Observabilidad & MLOps** | Middleware latencia, `/metrics` Prometheus, script CloudWatch | Prometheus + Grafana + PagerDuty/Slack + MLflow + Drift detection | Desplegar alertas automatizadas sobre `P99 > 90 ms`, Data/Concept Drift y F1 en producción, más tracking MLflow. |

