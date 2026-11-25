# Despliegue AWS SageMaker

Ejecuta la guia de la seccion **10. Despliegue del modelo en AWS SageMaker** del README principal.

- `deploy.py`: empaqueta el modelo JSON, lo sube a S3, crea el modelo y endpoint real-time, espera `InService` y ejecuta una prueba de humo.
- `delete_endpoint.py`: elimina endpoint, configuracion y modelo para detener el costo de hosting.
- `requirements.txt`: boto3 y SageMaker Python SDK.
- `test_response_parser.py`: prueba el parseo de las respuestas numericas del contenedor.

El endpoint entrega la probabilidad raw de XGBoost. FastAPI aplica despues el calibrador Platt y el umbral del archivo `model_metadata.json`.
