from __future__ import annotations
import argparse, json, os, time
from pathlib import Path
import boto3

p = argparse.ArgumentParser(description="Elimina endpoint, endpoint config y modelo SageMaker de PagoSeguro.")
p.add_argument("--info", default="sagemaker/deployment_info.json")
p.add_argument("--region", default=os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION"))
p.add_argument("--delete-s3", action="store_true", help="Tambien borra el modelo de S3 y trata de borrar el bucket si queda vacio.")
a = p.parse_args()
info = json.loads(Path(a.info).read_text(encoding="utf-8"))
region = a.region or info["region"]
sm = boto3.client("sagemaker", region_name=region)
s3 = boto3.client("s3", region_name=region)
endpoint = info["endpoint_name"]
try:
    sm.delete_endpoint(EndpointName=endpoint)
    print("Solicitud de eliminacion del endpoint enviada.")
except sm.exceptions.ResourceNotFound:
    print("El endpoint ya no existe.")
while True:
    try:
        state = sm.describe_endpoint(EndpointName=endpoint)["EndpointStatus"]
        print("Estado:", state)
        time.sleep(20)
    except sm.exceptions.ResourceNotFound:
        break
for key in ("endpoint_config_name", "model_name"):
    try:
        (sm.delete_endpoint_config if key == "endpoint_config_name" else sm.delete_model)(**({"EndpointConfigName": info[key]} if key == "endpoint_config_name" else {"ModelName": info[key]}))
        print("Eliminado:", info[key])
    except sm.exceptions.ResourceNotFound:
        pass
if a.delete_s3:
    bucket = info["bucket"]
    key = info["model_s3_uri"].split(f"{bucket}/", 1)[-1]
    try:
        s3.delete_object(Bucket=bucket, Key=key)
        s3.delete_bucket(Bucket=bucket)
        print("Objeto y bucket S3 eliminados (el bucket debia estar vacio).")
    except Exception as exc:
        print("No se pudo borrar S3 automaticamente:", exc)
print("Recursos SageMaker eliminados. Revisa la consola AWS para confirmar que no queden endpoints.")
