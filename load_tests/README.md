# Pruebas de Carga con Locust

Este directorio contiene las pruebas de rendimiento para medir **latencia (p50, p95, p99)**, **throughput (RPS/TPS)** y **tasa de errores**.

## Nota sobre `RATE_LIMIT`

Para pruebas de carga locales, ajusta temporalmente `RATE_LIMIT=10000/minute` en `.env` (y reinicia la API) para que `slowapi` no devuelva HTTP 429 artificialmente durante el benchmark.

## Ejecucion headless

En PowerShell (con la API corriendo en `http://127.0.0.1:8000`):

```powershell
$env:API_KEY = "tu-api-key"
.\.venv-sagemaker\Scripts\locust.exe -f load_tests/locustfile.py --headless -u 30 -r 10 -t 30s --host http://127.0.0.1:8000 --csv load_tests/results_local --html load_tests/report_local.html
```

Luego consulta `http://127.0.0.1:8000/metrics` para ver el resumen Prometheus acumulado en la API.

