# PagoSeguro - Dashboard Streamlit

Dashboard local para consumir la API FastAPI de PagoSeguro. Permite evaluar una transaccion, procesar un CSV mediante el endpoint batch, solicitar una explicacion SHAP y revisar un historial local de auditoria.

## Requisitos

- La API `PagoSeguro_FastAPI_SHAP` debe estar funcionando en `http://127.0.0.1:8000`.
- Debes tener la API Key configurada en el `.env` de la API.
- Python 3.10 o superior.

## Instalacion en Windows

Abre una segunda ventana de PowerShell (mantén FastAPI ejecutandose en la primera):

```powershell
cd PagoSeguro_Dashboard
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Inicia Streamlit:

```powershell
streamlit run streamlit_app.py
```

El dashboard se abrira normalmente en `http://localhost:8501`.

## Uso

1. En la barra lateral deja `http://127.0.0.1:8000` como URL base.
2. Ingresa la misma API Key configurada en el archivo `.env` de FastAPI.
3. Presiona **Comprobar conexion**. El dashboard consulta `/health` y `/model-info`.
4. En **Evaluar transaccion**, pega el JSON que ya utilizaste en Swagger. Puedes activar la explicacion SHAP.
5. En **Procesar CSV**, sube un archivo con todas las columnas de features que muestra `/model-info`. Se permite `transaction_id` opcional. La columna `Class` es opcional y solo se usa para calcular metricas locales; nunca se envia al modelo. Si aparece `Time` pero no esta en `/model-info`, se excluye.
6. En **Historial de auditoria**, consulta o descarga los registros guardados por este dashboard.

## Formato CSV

El archivo debe incluir una columna por cada feature del modelo. En la version entrenada que se esta utilizando se esperan las variables `V1` a `V28` y `Amount`; el dashboard lee la lista real de `/model-info` en vez de asumirla.

Columnas opcionales:

- `transaction_id`: identificador del registro.
- `Class`: etiqueta real (0 legitima, 1 fraude), solo para evaluacion de un dataset etiquetado.
- `Time`: se ignora si no aparece en el listado de features del modelo.

La API batch acepta hasta 100 filas por solicitud. El dashboard divide automaticamente los CSV en bloques de 100.

## Auditoria local

Los eventos se guardan en `data/audit.db` mediante SQLite. Se almacenan hora UTC, id de transaccion, origen, operacion, probabilidades, decision, umbral, variables del modelo y, cuando se solicita, explicacion SHAP. No se almacena la API Key.

**Limitacion:** SQLite es solo persistencia local para desarrollo y demostracion. Todavia no es el worker asincrono Kafka/MSK ni el repositorio distribuido DynamoDB/PostgreSQL propuesto en el Caso 3. La siguiente etapa sera desacoplar los eventos y desplegar el almacenamiento de auditoria en AWS.

## Flujo actual

```text
Usuario
  |
  v
Dashboard Streamlit (:8501)
  |
  | HTTP + X-API-Key
  v
FastAPI (:8000)
  |
  +--> XGBoost + calibrador --> decision
  |
  +--> SHAP (solo cuando se solicita explicacion)
  |
  v
Dashboard --> SQLite local (historial de auditoria)
```

## Importante

- La API y el dashboard son una implementacion local de desarrollo.
- Todavia no demuestra latencia menor o igual a 100 ms, disponibilidad 99,99%, 1.000 TPS ni una infraestructura desplegada en SageMaker/EKS.
- No ingreses datos reales de tarjetas ni datos personales. El dataset Kaggle utiliza variables anonimizadas.
- No subas `.env`, claves, bases SQLite ni datos de transacciones a GitHub.


## Auditoria distribuida Kafka + PostgreSQL

La version integrada publica el evento de prediccion a Kafka desde `/v1/predict`. El worker calcula SHAP en segundo plano y persiste el resultado en PostgreSQL. La pestaña de historial mantiene el SQLite local y agrega una seccion para consultar `/v1/audit/recent` en la API. Para ver la auditoria distribuida debes levantar la carpeta raiz `PagoSeguro_Streaming_Auditoria` con `docker compose up --build -d` y usar la misma API Key.
