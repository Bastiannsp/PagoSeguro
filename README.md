# PagoSeguro - Kafka, Worker SHAP y auditoria PostgreSQL

Esta etapa agrega una cola de eventos y un worker asincrono al proyecto PagoSeguro. FastAPI publica el resultado de cada prediccion; Apache Kafka transporta el evento; un worker independiente vuelve a calcular SHAP y persiste la explicacion y los datos de la decision en PostgreSQL. El dashboard consulta esa auditoria distribuida.

**Alcance:** esta es una implementacion local de desarrollo con Docker Compose. Apache Kafka local se reemplazara por Amazon MSK en AWS. No demuestra todavia TLS/mTLS, autenticacion SASL/IAM, disponibilidad 99,99%, latencia <=100 ms ni 1.000 TPS.

## 1. Preparar artefactos del modelo

Copia desde tu carpeta anterior `PagoSeguro_FastAPI_SHAP/artifacts/` a `api/artifacts/` estos tres archivos:

- `xgboost_fraud_model.json`
- `calibration_model.joblib`
- `model_metadata.json`

La carpeta `api/artifacts/` debe contener los tres archivos antes de construir los contenedores.

## 2. Configurar secretos

En PowerShell, desde la carpeta `PagoSeguro_Streaming_Auditoria`:

```powershell
Copy-Item .env.example .env
py -c "import secrets; print(secrets.token_urlsafe(32))"
py -c "import secrets; print(secrets.token_hex(24))"
```

Usa el primer valor como `API_KEY` y el segundo como `POSTGRES_PASSWORD` dentro de `.env`. Conserva esos valores en un lugar seguro y no subas `.env` a GitHub.

## 3. Levantar Kafka, PostgreSQL, API y worker

Necesitas Docker Desktop iniciado. Desde PowerShell en la carpeta raíz:

```powershell
docker compose up --build -d
docker compose ps
```

Servicios:

| Servicio | Funcion | Acceso |
|---|---|---|
| Kafka | Broker de eventos local | Solo red interna Docker (`kafka:9092`) |
| PostgreSQL | Almacena auditorias | `localhost:5432` |
| FastAPI | Prediccion y consulta de auditoria | `http://localhost:8000/docs` |
| Worker | Consume eventos y genera SHAP | Proceso interno |

La primera construccion puede tardar porque instala XGBoost, SHAP y dependencias de Python.

## 4. Ejecutar el dashboard

Abre otra terminal PowerShell en `dashboard/`. Puedes usar tu entorno virtual previo, pero instala las dependencias de esta carpeta:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run streamlit_app.py
```

En la barra lateral utiliza `http://127.0.0.1:8000` y la misma `API_KEY` de `.env`.

## 5. Verificar el flujo completo

1. Comprueba `http://localhost:8000/health`; debe mostrar `model_loaded: true`, `kafka_publisher_enabled: true` y `audit_database_configured: true`.
2. Ejecuta una solicitud `POST /v1/predict` desde Swagger o desde el dashboard.
3. La respuesta incluye `audit_event_status: "queued"`. Significa que el productor acepto el evento para envio; el SHAP se calcula posteriormente en el worker.
4. Revisa los logs del worker:

   ```powershell
   docker compose logs -f worker
   ```

5. Consulta `GET /v1/audit/recent?limit=20` en Swagger con el encabezado `X-API-Key`, o abre la pestaña de historial del dashboard y revisa `Auditoria distribuida Kafka + PostgreSQL`.

El worker debe registrar `Auditoria completada` y la tabla debe mostrar `processing_status=completed`. El worker confirma el offset de Kafka despues de persistir en PostgreSQL. Los eventos tienen `event_id` y la escritura es idempotente ante una reentrega.

## 6. Pruebas rapidas

```powershell
docker compose ps
docker compose logs --tail 100 api
docker compose logs --tail 100 worker
docker compose exec postgres psql -U pagoseguro_app -d pagoseguro -c "SELECT transaction_id, decision, processing_status, processed_at FROM audit_events ORDER BY processed_at DESC LIMIT 10;"
```

Para detener los servicios:

```powershell
docker compose down
```

Para eliminar tambien los datos locales de Kafka y PostgreSQL (esto borra auditorias guardadas):

```powershell
docker compose down -v
```

## 7. Arquitectura implementada

```text
Cliente / Dashboard
       |
       v
FastAPI: POST /v1/predict
       |
       +---- respuesta inmediata de prediccion
       |
       v
Kafka topic: pagoseguro.transaction.predicted
       |
       v
Worker asincrono
       |
       +---- carga modelo XGBoost
       +---- calcula SHAP
       |
       v
PostgreSQL: audit_events
       |
       v
FastAPI: GET /v1/audit/recent
       |
       v
Dashboard: auditoria distribuida
```

## 8. Consideraciones importantes

- El endpoint `/v1/predict` no calcula SHAP en linea. Publica el evento y devuelve el resultado del modelo con `audit_event_status`.
- `/v1/explain` sigue disponible para solicitar explicabilidad inmediata bajo demanda. El worker genera la explicacion asincrona para las predicciones que llegan a Kafka.
- El productor Kafka tiene un buffer asincrono. `queued` no significa que el worker ya haya persistido; confirma la persistencia consultando la auditoria distribuida.
- Si Kafka no esta disponible, la API mantiene la prediccion, informa `audit_event_status: unavailable` y deja un error en los logs. Para produccion haria falta un patron outbox/reintentos durables.
- En el entorno local Kafka no esta autenticado ni cifrado. No expongas estos puertos a Internet ni uses datos reales de tarjetas.
- Para AWS, usa Amazon MSK con TLS y SASL/IAM o SCRAM segun la configuracion del cluster, credenciales en Secrets Manager, subredes privadas y security groups restrictivos. Mantén IAM de minimo privilegio.
- PostgreSQL local sirve como base de desarrollo. En AWS se puede usar Amazon RDS/Aurora PostgreSQL o DynamoDB segun los requisitos de auditoria.
- La estrategia de reintentos del consumidor y la clave `event_id` permiten al menos una vez con escritura idempotente; no es una garantia de exactly-once extremo a extremo.
- La base local usa volumen Docker. Si ejecutas `docker compose down -v`, se eliminan los datos almacenados.

## 9. Estructura

```text
PagoSeguro_Streaming_Auditoria/
├── api/
│   ├── app/
│   ├── artifacts/
│   ├── Dockerfile
│   └── requirements.txt
├── worker/
│   ├── audit_worker.py
│   └── Dockerfile
├── dashboard/
├── db/001_init.sql
├── docker-compose.yml
└── .env.example
```


## 10. Despliegue del modelo en AWS SageMaker

Esta version permite mantener el flujo local validado y cambiar la inferencia a SageMaker mediante variables de entorno. FastAPI conserva localmente el JSON de XGBoost para SHAP; cuando `USE_SAGEMAKER_ENDPOINT=true`, las probabilidades raw se solicitan al endpoint administrado de SageMaker. FastAPI aplica despues el mismo calibrador Platt y el umbral guardado en `model_metadata.json`. El worker SHAP continua usando la misma copia local del modelo.

### 10.1 Requisitos AWS

- Cuenta AWS Academy/Learner Lab o cuenta AWS con permisos habilitados.
- AWS CLI v2 o credenciales temporales disponibles para Boto3.
- Un rol de ejecucion que SageMaker pueda asumir y que tenga acceso de lectura al artefacto en S3. El usuario que ejecuta el despliegue debe tener permisos para S3, SageMaker y `iam:PassRole`.
- Confirma la region habilitada en tu laboratorio.

En AWS Academy, abre **AWS Details / AWS CLI** para obtener las credenciales temporales (`AccessKeyId`, `SecretAccessKey`, `SessionToken`). No publiques esos valores ni los subas a GitHub. El rol de laboratorio puede llamarse `LabRole`, pero confirma el ARN real en IAM; no lo inventes.

### 10.2 Instalar herramientas de despliegue

Desde la carpeta raiz del proyecto:

```powershell
py -m venv .venv-sagemaker
.\.venv-sagemaker\Scripts\Activate.ps1
pip install -r sagemaker/requirements.txt
```

En la misma terminal establece las credenciales temporales y region. Sustituye los valores entre comillas por los de AWS Academy:

```powershell
$env:AWS_ACCESS_KEY_ID="..."
$env:AWS_SECRET_ACCESS_KEY="..."
$env:AWS_SESSION_TOKEN="..."
$env:AWS_DEFAULT_REGION="us-east-1"
$env:AWS_REGION="us-east-1"
$env:SAGEMAKER_EXECUTION_ROLE_ARN="arn:aws:iam::TU_CUENTA:role/LabRole"
```

Comprueba identidad y region:

```powershell
aws sts get-caller-identity
aws configure get region
```

Si el laboratorio entrega otro rol, region o nombre, usa los datos reales del entorno.

### 10.3 Crear el endpoint

Asegurate de que los tres artefactos del modelo estan en `api/artifacts/`:

- `xgboost_fraud_model.json`
- `calibration_model.joblib`
- `model_metadata.json`

Luego ejecuta:

```powershell
python sagemaker/deploy.py --artifacts-dir api/artifacts --instance-type ml.m5.large
```

El script crea un bucket S3 con nombre unico (o puedes entregar `--bucket NOMBRE_BUCKET`), empaqueta el modelo JSON como `xgboost-model`, crea el modelo, endpoint config y endpoint real-time. Espera a `InService` y realiza una invocacion de prueba con un vector sintetico de ceros (solo smoke test; no es una transaccion real). El resultado queda en `sagemaker/deployment_info.json`.

Se utiliza la imagen XGBoost SageMaker version `3.2-0` de framework mode. Si el laboratorio o la region no ofrecen esa imagen, revisa las versiones soportadas en la documentacion oficial y cambia `--xgboost-image-version` a una version disponible y compatible. El modelo se guarda en JSON nativo de XGBoost; el script conserva el contenido y lo empaqueta con el nombre `xgboost-model` esperado por el contenedor.

### 10.4 Conectar FastAPI local con SageMaker

Copia el endpoint que imprimio el script a `.env` (el `.env` esta ignorado por Git). Mantén los secretos de AWS solo en ese archivo local/temporal y no en el repositorio:

```dotenv
USE_SAGEMAKER_ENDPOINT=true
SAGEMAKER_ENDPOINT_NAME=nombre-real-del-endpoint
AWS_REGION=us-east-1
AWS_DEFAULT_REGION=us-east-1
AWS_ACCESS_KEY_ID=credencial-temporal
AWS_SECRET_ACCESS_KEY=credencial-temporal
AWS_SESSION_TOKEN=token-temporal
```

No reemplaces `API_KEY`, `POSTGRES_PASSWORD`, `KAFKA_TOPIC` ni otros valores actuales. Conserva tambien la carpeta `api/artifacts/`, porque el servicio local la necesita para generar SHAP. Luego reconstruye solo la API:

```powershell
docker compose up -d --build api
docker compose logs -f api
```

Verifica `GET /health`: debe mostrar `inference_backend: "sagemaker"` y `sagemaker_endpoint_configured: true`. `GET /model-info` debe mantener el umbral del modelo. Realiza una prediccion por Swagger/dashboard y confirma que el campo `inference_backend` de la respuesta sea `sagemaker`.

La calibracion Platt y el umbral permanecen en FastAPI; SageMaker entrega la probabilidad raw del XGBoost. SHAP se sigue calculando localmente sobre el mismo archivo de modelo porque es un proceso separado de la inferencia online.

### 10.5 Volver al modo local

Cambia `USE_SAGEMAKER_ENDPOINT=false` en `.env` y reconstruye la API. Se conserva el mismo endpoint de prediccion local y no se invoca AWS.

### 10.6 Eliminar recursos y detener costos

Un endpoint de inferencia provisionado tiene costo mientras permanece activo, aunque no reciba solicitudes. Cuando termines las pruebas, elimina el endpoint:

```powershell
python sagemaker/delete_endpoint.py
```

El script elimina endpoint, endpoint config y modelo, pero conserva el artefacto en S3 por seguridad. Para eliminar tambien el objeto y el bucket generado (solo si estas seguro):

```powershell
python sagemaker/delete_endpoint.py --delete-s3
```

Revisa AWS Console para confirmar que no queden endpoints activos. No uses un endpoint provisionado para probar durante dias si no lo necesitas.

### 10.7 Limitaciones de esta etapa

- El tipo `ml.m5.large` con una instancia es una configuracion de desarrollo, no valida por si sola 1.000 TPS, autoscaling ni disponibilidad 99,99%.
- La latencia end-to-end depende de red, FastAPI, SageMaker, instancia y carga; debe medirse en la etapa de testing.
- Las credenciales temporales AWS Academy expiran; renuevalas y reinicia/reconstruye la API si corresponde.
- La integracion local de Docker con credenciales temporales es para demostracion. En produccion, la API debe usar un rol IAM adjunto al servicio (ECS/EKS/EC2), no access keys guardadas en `.env`.
- El modo SageMaker invoca el endpoint de manera sincrona; Kafka sigue procesando SHAP y auditoria de forma asincrona.

## 11. Testing, Pruebas de Carga, TLS y Evidencias CloudWatch

- **Pruebas unitarias y de integracion (15 casos):**
  ```powershell
  .\.venv-sagemaker\Scripts\python.exe -m pytest tests/ sagemaker/test_response_parser.py -v
  ```
- **Pruebas de carga con Locust (p50, p95, p99, throughput):** revisa [load_tests/README.md](load_tests/README.md).
- **Observabilidad Prometheus:** consulta `GET /metrics` (o el header `X-Process-Time-Ms` en cada respuesta HTTP).
- **TLS/HTTPS con Nginx y Dashboard en Docker:** `docker compose up --build -d` levanta `dashboard` (`:8501`) y `nginx` (`https://localhost:8443`).
- **Auto-Scaling y CloudWatch en SageMaker:**
  ```powershell
  python sagemaker/configure_autoscaling.py
  python sagemaker/export_cloudwatch_evidence.py
  ```
- **Documentacion y borrador del informe:** revisa `docs/analisis_costos.md`, `docs/arquitectura_y_secuencia.md` y `docs/borrador_informe_tecnico_EP3.md`.
