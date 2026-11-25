from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import boto3

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


def parse_args():
    parser = argparse.ArgumentParser(
        description="Configura Application Auto Scaling para el endpoint SageMaker de PagoSeguro."
    )
    parser.add_argument("--info", default="sagemaker/deployment_info.json")
    parser.add_argument("--endpoint-name", default=None)
    parser.add_argument("--variant-name", default="AllTraffic")
    parser.add_argument("--region", default=os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION"))
    parser.add_argument("--min-capacity", type=int, default=1)
    parser.add_argument("--max-capacity", type=int, default=4)
    parser.add_argument("--target-invocations", type=float, default=70.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    info = {}
    info_path = Path(args.info)
    if info_path.is_file():
        info = json.loads(info_path.read_text(encoding="utf-8"))

    endpoint_name = args.endpoint_name or info.get("endpoint_name")
    region = args.region or info.get("region") or "us-east-1"
    if not endpoint_name:
        raise SystemExit("Falta --endpoint-name o sagemaker/deployment_info.json.")

    resource_id = f"endpoint/{endpoint_name}/variant/{args.variant_name}"
    aas = boto3.client("application-autoscaling", region_name=region)

    aas.register_scalable_target(
        ServiceNamespace="sagemaker",
        ResourceId=resource_id,
        ScalableDimension="sagemaker:variant:DesiredInstanceCount",
        MinCapacity=args.min_capacity,
        MaxCapacity=args.max_capacity,
    )

    policy = aas.put_scaling_policy(
        PolicyName=f"{endpoint_name}-invocations-scaling",
        ServiceNamespace="sagemaker",
        ResourceId=resource_id,
        ScalableDimension="sagemaker:variant:DesiredInstanceCount",
        PolicyType="TargetTrackingScaling",
        TargetTrackingScalingPolicyConfiguration={
            "TargetValue": args.target_invocations,
            "PredefinedMetricSpecification": {
                "PredefinedMetricType": "SageMakerVariantInvocationsPerInstance"
            },
            "ScaleInCooldown": 300,
            "ScaleOutCooldown": 60,
        },
    )

    print("Auto-Scaling configurado correctamente:")
    print("  ResourceId:", resource_id)
    print(f"  Capacidad: {args.min_capacity} a {args.max_capacity} instancias")
    print("  Politica ARN:", policy["PolicyARN"])


if __name__ == "__main__":
    main()

