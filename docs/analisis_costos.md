# Análisis de Costos AWS — PagoSeguro (Caso 3)

## 1. Escenario 1: Entorno Desplegado en AWS Academy (MVP EP3)

Configuración utilizada para validar el sistema completo y ejecutar la defensa técnica dentro de los límites de AWS Academy Learner Lab:

| Recurso AWS | Especificación | Uso Mensual Estimado (730 h) | Costo Horario Ref. (us-east-1) | Costo Mensual Ref. (USD) |
| :--- | :--- | :---: | :---: | :---: |
| **Amazon SageMaker Real-Time Endpoint** | 1x `ml.m5.large` (2 vCPU, 8 GB RAM) | 730 h (continuo) / 20 h (demo) | \$0.115 / h | \$83.95 (24/7) · \$2.30 (solo pruebas 20h) |
| **Amazon S3** | Almacenamiento de `model.tar.gz` (~1 MB) y requests | < 1 GB | \$0.023 / GB-mes | < \$0.05 |
| **Amazon EC2 (Host Docker Compose)** *(opcional nube)* | 1x `t3.medium` (2 vCPU, 4 GB RAM, 30 GB EBS gp3) | 730 h (continuo) / 20 h (demo) | \$0.0416 / h + EBS \$2.40 | \$32.77 (24/7) · \$3.23 (20h) |
| **Amazon CloudWatch** | Métricas estándar SageMaker + Logs básicos (< 5 GB) | Incluido en capa/uso bajo | — | ~\$1.00 - \$3.00 |
| **Total MVP AWS Academy** | **Operación continua 24/7 / Solo laboratorio (20h)** | — | — | **~\$118 USD/mes** *(o ~\$6 - \$9 USD en sesiones de prueba)* |

> **Recomendación operativa:** Ejecutar `python sagemaker/delete_endpoint.py` después de las pruebas y volver a desplegar antes de la presentación para minimizar consumo de créditos en AWS Academy.

---

## 2. Escenario 2: Arquitectura Target de Producción (Caso 3 — 1.000.000 tx/día, picos 1.000 TPS)

Comparación referencial contra el dimensionamiento solicitado en el Caso 3 (Sección 5) y el presupuesto referencial (**\$3.000 – \$5.000 USD/mes** / estimado **\$1.500 – \$2.800 USD/mes**):

| Componente del Caso 3 | Servicio AWS Target | Dimensionamiento Referencial | Costo Mensual Estimado (USD) |
| :--- | :--- | :--- | :---: |
| **Inference Engine** | Amazon EKS / SageMaker Multi-AZ | 4 a 12 nodos `c6i.xlarge` (4 vCPU, 8 GB RAM, promedio 6 nodos) | \$745 – \$1.490 |
| **Streaming & Event Broker** | Amazon MSK (Managed Kafka) | 3 brokers `kafka.m5.large` Multi-AZ + almacenamiento EBS | \$480 – \$650 |
| **Feature Store (< 10 ms)** | Amazon ElastiCache for Redis | Cluster Multi-AZ / replicación (`cache.r6g.large`) | \$280 – \$450 |
| **Auditoría XAI (PostgreSQL / DynamoDB)** | Amazon RDS PostgreSQL Multi-AZ o DynamoDB | `db.t4g.medium` Multi-AZ + 100 GB gp3 + backups | \$130 – \$220 |
| **Balanceo, Seguridad y Red** | ALB + KMS (CMK) + Secrets Manager + NAT Gateway | TLS/mTLS + llaves administradas + tráfico | \$120 – \$220 |
| **Observabilidad y MLOps** | CloudWatch + Amazon Managed Prometheus/Grafana | Métricas P99, Drift, logs y alertas | \$80 – \$150 |
| **Total Target Producción (Caso 3)** | **Infraestructura Multi-AZ de Alta Disponibilidad** | — | **~\$1.835 – \$3.180 USD/mes** |

## 3. Conclusión Financiera

- El **MVP implementado** cuesta **~\$118 USD/mes** si corre 24/7 (y menos de **\$10 USD** durante ventanas de prueba), demostrando la integración real con **AWS SageMaker** dentro del presupuesto de AWS Academy.
- La **arquitectura target escalable** se ubica en **~\$1.835 – \$3.180 USD/mes** (alineado al rango estimado de \$1.500–\$2.800 USD/mes y debajo del techo presupuestario de \$3.000–\$5.000 USD/mes del Caso 3), logrando un ahorro estimado frente a los **\$500.000 USD/año (\$41.667 USD/mes)** en pérdidas por fraude reportados en el problema inicial.

