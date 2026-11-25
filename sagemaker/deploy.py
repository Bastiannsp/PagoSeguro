from __future__ import annotations

import argparse
import json
import os
import shutil
import tarfile
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path



def parse_args():
    parser = argparse.ArgumentParser(
        description="Empaqueta el modelo XGBoost de PagoSeguro y crea un endpoint real-time en SageMaker."
    )
    parser.add_argument("--artifacts-dir", default="api/artifacts", help="Carpeta con JSON de XGBoost y metadata.")
    parser.add_argument("--role-arn", default=os.getenv("SAGEMAKER_EXECUTION_ROLE_ARN"), help="ARN del rol IAM de ejecucion SageMaker.")
    parser.add_argument("--region", default=os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION"))
    parser.add_argument("--bucket", default=os.getenv("SAGEMAKER_BUCKET"), help="Bucket S3 existente; si se omite se crea uno.")
    parser.add_argument("--endpoint-name", default=None, help="Nombre opcional. Si se omite se genera uno unico.")
    parser.add_argument("--instance-type", default="ml.m5.large", help="Instancia para prueba inicial; tiene costo mientras el endpoint exista.")
    parser.add_argument("--instance-count", type=int, default=1)
    parser.add_argument("--xgboost-image-version", default="3.2-0", help="Version soportada del contenedor SageMaker XGBoost.")
    parser.add_argument("--wait-seconds", type=int, default=1800)
    return parser.parse_args()


def ensure_bucket(s3, bucket: str, region: str):
    try:
        if region == "us-east-1":
            s3.create_bucket(Bucket=bucket)
        else:
            s3.create_bucket(
                Bucket=bucket,
                CreateBucketConfiguration={"LocationConstraint": region},
            )
        print(f"Bucket S3 creado: {bucket}")
    except s3.exceptions.BucketAlreadyOwnedByYou:
        print(f"Se usara el bucket existente: {bucket}")
    except s3.exceptions.BucketAlreadyExists as exc:
        raise RuntimeError(
            f"El nombre de bucket S3 ya pertenece a otra cuenta: {bucket}. Ejecuta otra vez para generar otro."
        ) from exc


def parse_probability(payload: str) -> float:
    text = payload.strip()
    try:
        value = json.loads(text)
        while isinstance(value, list) and value:
            value = value[0]
        if isinstance(value, dict):
            for key in ("prediction", "predictions", "probability", "score"):
                if key in value:
                    value = value[key]
                    while isinstance(value, list) and value:
                        value = value[0]
                    break
        if isinstance(value, (int, float)):
            result = float(value)
        else:
            result = float(text.split(",")[0].strip())
    except Exception:
        result = float(text.split(",")[0].strip())
    if not 0 <= result <= 1:
        raise ValueError(f"El contenedor devolvio un valor fuera de [0,1]: {text}")
    return result


def wait_endpoint(sm, endpoint_name: str, max_seconds: int):
    started = time.time()
    while time.time() - started < max_seconds:
        result = sm.describe_endpoint(EndpointName=endpoint_name)
        status = result["EndpointStatus"]
        print(f"Estado endpoint {endpoint_name}: {status}")
        if status == "InService":
            return
        if status in {"Failed", "OutOfService"}:
            raise RuntimeError(
                f"SageMaker no pudo dejar el endpoint operativo. Motivo: {result.get('FailureReason', 'No informado')}"
            )
        time.sleep(30)
    raise TimeoutError(f"El endpoint no quedo InService dentro de {max_seconds} segundos.")


def main():
    args = parse_args()
    if not args.region:
        raise SystemExit("Falta region. Configura AWS_DEFAULT_REGION o usa --region, por ejemplo us-east-1.")
    if not args.role_arn:
        raise SystemExit("Falta --role-arn o SAGEMAKER_EXECUTION_ROLE_ARN. En AWS Academy suele existir un rol de laboratorio; confirma su ARN en IAM.")
    if args.instance_count < 1:
        raise SystemExit("--instance-count debe ser al menos 1.")

    artifacts = Path(args.artifacts_dir).resolve()
    model_json = artifacts / "xgboost_fraud_model.json"
    metadata_path = artifacts / "model_metadata.json"
    if not model_json.is_file() or not metadata_path.is_file():
        raise SystemExit(
            f"No encuentro {model_json} y/o {metadata_path}. Copia los artefactos desde Colab a api/artifacts/."
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    features = metadata.get("features")
    if not features:
        raise SystemExit("model_metadata.json no contiene la lista 'features'.")
    print("Modelo del proyecto:", metadata.get("model_type", "XGBoost"))
    print("Version registrada al entrenar:", metadata.get("xgboost_version", "no informada"))
    print("Cantidad de variables:", len(features))
    print("Umbral se mantiene en FastAPI:", metadata.get("threshold"))

    import boto3
    from sagemaker.core import image_uris

    session = boto3.Session(region_name=args.region)
    sts = session.client("sts")
    identity = sts.get_caller_identity()
    account_id = identity["Account"]
    s3 = session.client("s3")
    sm = session.client("sagemaker")
    runtime = session.client("sagemaker-runtime")

    bucket = args.bucket or f"pagoseguro-model-{account_id}-{args.region}-{uuid.uuid4().hex[:8]}"
    ensure_bucket(s3, bucket, args.region)

    suffix = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:4]
    endpoint_name = args.endpoint_name or f"pagoseguro-fraud-{suffix}"
    model_name = f"{endpoint_name}-model"
    config_name = f"{endpoint_name}-config"
    s3_key = f"pagoseguro/{endpoint_name}/model.tar.gz"

    with tempfile.TemporaryDirectory(prefix="pagoseguro-sm-") as tmp:
        tmp_path = Path(tmp)
        # El contenedor oficial busca el artefacto con este nombre. El contenido sigue siendo JSON nativo XGBoost.
        model_file = tmp_path / "xgboost-model"
        shutil.copy2(model_json, model_file)
        archive_path = tmp_path / "model.tar.gz"
        with tarfile.open(archive_path, "w:gz") as archive:
            archive.add(model_file, arcname="xgboost-model")
        s3.upload_file(str(archive_path), bucket, s3_key)

    model_data_url = f"s3://{bucket}/{s3_key}"
    image_uri = image_uris.retrieve(
        framework="xgboost",
        region=args.region,
        version=args.xgboost_image_version,
    )
    print("Imagen XGBoost SageMaker:", image_uri)
    print("Modelo en S3:", model_data_url)

    sm.create_model(
        ModelName=model_name,
        ExecutionRoleArn=args.role_arn,
        PrimaryContainer={
            "Image": image_uri,
            "ModelDataUrl": model_data_url,
            "Environment": {"SAGEMAKER_CONTAINER_LOG_LEVEL": "20"},
        },
        EnableNetworkIsolation=True,
        Tags=[{"Key": "Project", "Value": "PagoSeguro"}, {"Key": "Course", "Value": "ITY1102-EP3"}],
    )
    sm.create_endpoint_config(
        EndpointConfigName=config_name,
        ProductionVariants=[
            {
                "VariantName": "AllTraffic",
                "ModelName": model_name,
                "InitialInstanceCount": args.instance_count,
                "InstanceType": args.instance_type,
                "InitialVariantWeight": 1.0,
            }
        ],
        Tags=[{"Key": "Project", "Value": "PagoSeguro"}],
    )
    sm.create_endpoint(
        EndpointName=endpoint_name,
        EndpointConfigName=config_name,
        Tags=[{"Key": "Project", "Value": "PagoSeguro"}],
    )
    wait_endpoint(sm, endpoint_name, args.wait_seconds)

    # Smoke test: vector cero (no representa una transaccion real, solo verifica el contrato del endpoint).
    sample_values = [0.0 for _ in features]
    if "Amount" in features:
        sample_values[features.index("Amount")] = 1.0
    csv_body = ",".join(format(value, ".17g") for value in sample_values)
    response = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="text/csv",
        Accept="text/csv",
        Body=csv_body.encode("utf-8"),
    )
    response_text = response["Body"].read().decode("utf-8")
    raw_probability = parse_probability(response_text)
    print("Smoke test SageMaker OK. Probabilidad XGBoost sin calibrar:", raw_probability)

    info = {
        "project": "PagoSeguro",
        "endpoint_name": endpoint_name,
        "model_name": model_name,
        "endpoint_config_name": config_name,
        "region": args.region,
        "bucket": bucket,
        "model_s3_uri": model_data_url,
        "image_uri": image_uri,
        "instance_type": args.instance_type,
        "instance_count": args.instance_count,
        "xgboost_image_version": args.xgboost_image_version,
        "training_xgboost_version": metadata.get("xgboost_version"),
        "features": features,
        "threshold": metadata.get("threshold"),
        "smoke_test_raw_probability": raw_probability,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    output_path = Path(__file__).resolve().parent / "deployment_info.json"
    output_path.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nEndpoint desplegado correctamente:", endpoint_name)
    print("Datos de despliegue guardados en:", output_path)
    print("\nAgrega al .env de PagoSeguro:")
    print("USE_SAGEMAKER_ENDPOINT=true")
    print(f"SAGEMAKER_ENDPOINT_NAME={endpoint_name}")
    print(f"AWS_REGION={args.region}")
    print(f"AWS_DEFAULT_REGION={args.region}")
    print("AWS_ACCESS_KEY_ID=... (credencial temporal de AWS Academy)")
    print("AWS_SECRET_ACCESS_KEY=... (credencial temporal de AWS Academy)")
    print("AWS_SESSION_TOKEN=... (credencial temporal de AWS Academy)")
    print("\nEl rol de ejecucion de SageMaker necesita leer el objeto S3. El usuario que ejecuta este script necesita iam:PassRole, S3 y permisos SageMaker.")


if __name__ == "__main__":
    main()
