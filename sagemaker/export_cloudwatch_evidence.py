from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import boto3

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


def parse_args():
    parser = argparse.ArgumentParser(
        description="Exporta metricas y logs recientes de CloudWatch del endpoint SageMaker."
    )
    parser.add_argument("--info", default="sagemaker/deployment_info.json")
    parser.add_argument("--endpoint-name", default=None)
    parser.add_argument("--region", default=os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION"))
    parser.add_argument("--hours", type=int, default=6)
    parser.add_argument("--output-json", default="docs/cloudwatch_evidence.json")
    return parser.parse_args()


def fetch_metric(cw, endpoint_name: str, metric_name: str, stat: str, start: datetime, end: datetime):
    response = cw.get_metric_statistics(
        Namespace="AWS/SageMaker",
        MetricName=metric_name,
        Dimensions=[
            {"Name": "EndpointName", "Value": endpoint_name},
            {"Name": "VariantName", "Value": "AllTraffic"},
        ],
        StartTime=start,
        EndTime=end,
        Period=300,
        Statistics=[stat],
    )
    datapoints = sorted(response.get("Datapoints", []), key=lambda x: x["Timestamp"])
    return [
        {"timestamp": dp["Timestamp"].isoformat(), "value": dp[stat], "unit": dp.get("Unit")}
        for dp in datapoints
    ]


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

    cw = boto3.client("cloudwatch", region_name=region)
    logs = boto3.client("logs", region_name=region)
    end = datetime.now(timezone.utc)
    start = end - timedelta(hours=args.hours)

    metrics = {
        "Invocations": fetch_metric(cw, endpoint_name, "Invocations", "Sum", start, end),
        "ModelLatency_us": fetch_metric(cw, endpoint_name, "ModelLatency", "Average", start, end),
        "OverheadLatency_us": fetch_metric(cw, endpoint_name, "OverheadLatency", "Average", start, end),
        "Invocation4XXErrors": fetch_metric(cw, endpoint_name, "Invocation4XXErrors", "Sum", start, end),
        "Invocation5XXErrors": fetch_metric(cw, endpoint_name, "Invocation5XXErrors", "Sum", start, end),
    }

    log_group = f"/aws/sagemaker/Endpoints/{endpoint_name}"
    recent_logs = []
    try:
        streams = logs.describe_log_streams(
            logGroupName=log_group, orderBy="LastEventTime", descending=True, limit=3
        ).get("logStreams", [])
        for stream in streams:
            events = logs.get_log_events(
                logGroupName=log_group, logStreamName=stream["logStreamName"], limit=20
            ).get("events", [])
            recent_logs.extend(
                {"stream": stream["logStreamName"], "message": ev["message"].strip()}
                for ev in events
            )
    except Exception as exc:
        recent_logs.append({"warning": f"No se pudieron leer logs de {log_group}: {exc}"})

    payload = {
        "endpoint_name": endpoint_name,
        "region": region,
        "window_utc": {"start": start.isoformat(), "end": end.isoformat()},
        "metrics": metrics,
        "recent_logs": recent_logs[-30:],
    }
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Evidencia CloudWatch exportada en:", out_path)


if __name__ == "__main__":
    main()

