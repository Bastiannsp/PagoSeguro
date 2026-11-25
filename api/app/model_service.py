import json
import math
import os
import re

import boto3
from pathlib import Path
from typing import Any

import joblib
import xgboost as xgb
import numpy as np
import pandas as pd


# Orden de columnas original del dataset Credit Card Fraud Detection.
# El metadata del entrenamiento tiene prioridad sobre esta lista.
DEFAULT_FEATURES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]


class FraudModelService:
    """Carga el modelo, el calibrador, el umbral y genera explicaciones SHAP."""

    def __init__(self, artifacts_dir: str | Path = "artifacts") -> None:
        self.artifacts_dir = Path(artifacts_dir)
        model_path = self.artifacts_dir / "xgboost_fraud_model.json"
        calibrator_path = self.artifacts_dir / "calibration_model.joblib"
        metadata_path = self.artifacts_dir / "model_metadata.json"

        required_files = [model_path, calibrator_path, metadata_path]
        missing_files = [str(path) for path in required_files if not path.is_file()]
        if missing_files:
            raise FileNotFoundError(
                "Faltan artefactos del modelo: " + ", ".join(missing_files)
                + ". Copia los tres archivos exportados desde Colab a la carpeta artifacts/."
            )

        with metadata_path.open("r", encoding="utf-8") as file:
            self.metadata: dict[str, Any] = json.load(file)

        try:
            self.model = xgb.XGBClassifier()
            self.model.load_model(str(model_path))
        except Exception as exc:
            raise RuntimeError(
                "No se pudo cargar xgboost_fraud_model.json con XGBoost. "
                "Vuelve a exportar el JSON desde Colab y revisa que la version "
                "de xgboost sea compatible. Detalle: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        try:
            self.calibrator = joblib.load(calibrator_path)
        except Exception as exc:
            raise RuntimeError(
                "No se pudo cargar calibration_model.joblib. "
                "Vuelve a exportar ese archivo desde Colab. Detalle: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        threshold = self.metadata.get("threshold")
        if threshold is None:
            raise ValueError("model_metadata.json no contiene el campo 'threshold'.")
        self.threshold = float(threshold)
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("El umbral del metadata debe estar entre 0 y 1.")

        features = self.metadata.get("features")
        if not features:
            features = getattr(self.model, "feature_names_in_", None)
        if features is None or len(features) == 0:
            features = DEFAULT_FEATURES
        self.features = [str(feature) for feature in features]

        model_feature_count = getattr(self.model, "n_features_in_", None)
        if model_feature_count is not None and int(model_feature_count) != len(self.features):
            raise ValueError(
                "La cantidad de variables del metadata no coincide con el modelo: "
                f"metadata={len(self.features)}, modelo={model_feature_count}."
            )

        # SageMaker es opcional: en modo local se conserva el comportamiento validado.
        self.use_sagemaker = os.getenv("USE_SAGEMAKER_ENDPOINT", "false").strip().lower() in {"1", "true", "yes", "si", "sí"}
        self.sagemaker_endpoint_name = os.getenv("SAGEMAKER_ENDPOINT_NAME", "").strip()
        self.aws_region = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or None
        self.sagemaker_runtime = None
        if self.use_sagemaker:
            if not self.sagemaker_endpoint_name:
                raise ValueError("USE_SAGEMAKER_ENDPOINT=true requiere SAGEMAKER_ENDPOINT_NAME.")
            try:
                self.sagemaker_runtime = boto3.client("sagemaker-runtime", region_name=self.aws_region)
            except Exception as exc:
                raise RuntimeError(f"No se pudo configurar boto3 para SageMaker: {exc}") from exc

        self.inference_backend = "sagemaker" if self.use_sagemaker else "local"

        # Se inicializa solo cuando se solicita una explicacion.
        # Asi SHAP no agrega tiempo al endpoint normal de prediccion.
        # SHAP utiliza la copia local del mismo artefacto que fue subido a SageMaker.
        self._shap_explainer = None

    def _prepare_row(self, features: dict[str, Any]) -> pd.DataFrame:
        """Valida las variables y las ordena como espera el modelo."""
        expected = set(self.features)
        received = set(features.keys())
        missing = [name for name in self.features if name not in received]
        extra = sorted(received - expected)

        if missing or extra:
            details = {}
            if missing:
                details["missing_features"] = missing
            if extra:
                details["unknown_features"] = extra
            raise ValueError(json.dumps(details, ensure_ascii=False))

        values: list[float] = []
        for name in self.features:
            value = features[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"La variable '{name}' debe ser numerica.")
            numeric_value = float(value)
            if not math.isfinite(numeric_value):
                raise ValueError(f"La variable '{name}' debe ser un numero finito.")
            values.append(numeric_value)

        return pd.DataFrame([values], columns=self.features)

    def _calibrate_probability(self, raw_probability: float) -> float:
        if self.calibrator is None:
            return raw_probability

        clipped = np.clip(raw_probability, 1e-6, 1 - 1e-6)
        logit_value = np.log(clipped / (1 - clipped))
        return float(self.calibrator.predict_proba(np.array([[logit_value]]))[0, 1])

    @staticmethod
    def _parse_sagemaker_probability(response_body: bytes | str) -> float:
        """Convierte la respuesta CSV/texto del contenedor XGBoost a probabilidad."""
        text = response_body.decode("utf-8") if isinstance(response_body, bytes) else str(response_body)
        text = text.strip()
        if not text:
            raise RuntimeError("SageMaker devolvio una respuesta vacia.")

        value = None
        try:
            parsed = json.loads(text)
            while isinstance(parsed, (list, tuple)) and parsed:
                parsed = parsed[0]
            if isinstance(parsed, dict):
                for key in ("probability", "prediction", "predictions", "score"):
                    if key in parsed:
                        parsed = parsed[key]
                        while isinstance(parsed, (list, tuple)) and parsed:
                            parsed = parsed[0]
                        break
            if isinstance(parsed, (int, float)) and not isinstance(parsed, bool):
                value = float(parsed)
        except (json.JSONDecodeError, TypeError, ValueError):
            value = None

        if value is None:
            match = re.search(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?", text)
            if match:
                value = float(match.group(0))

        if value is None or not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise RuntimeError(f"Respuesta de SageMaker no reconocida como probabilidad: {text[:200]}")
        return float(value)

    def _predict_raw_sagemaker(self, row: pd.DataFrame) -> float:
        if self.sagemaker_runtime is None or not self.sagemaker_endpoint_name:
            raise RuntimeError("El endpoint de SageMaker no esta configurado.")
        csv_body = ",".join(format(float(value), ".17g") for value in row.iloc[0].tolist())
        try:
            response = self.sagemaker_runtime.invoke_endpoint(
                EndpointName=self.sagemaker_endpoint_name,
                ContentType="text/csv",
                Accept="text/csv",
                Body=csv_body.encode("utf-8"),
            )
            return self._parse_sagemaker_probability(response["Body"].read())
        except Exception as exc:
            raise RuntimeError(
                f"No se pudo invocar el endpoint SageMaker '{self.sagemaker_endpoint_name}': "
                f"{type(exc).__name__}: {exc}"
            ) from exc

    def predict(self, transaction_id: str | None, features: dict[str, Any]) -> dict[str, Any]:
        row = self._prepare_row(features)
        if self.use_sagemaker:
            raw_probability = self._predict_raw_sagemaker(row)
        else:
            raw_probability = float(self.model.predict_proba(row)[0, 1])
        calibrated_probability = self._calibrate_probability(raw_probability)

        predicted_class = int(calibrated_probability >= self.threshold)
        decision = "REVISAR_O_RECHAZAR" if predicted_class == 1 else "AUTORIZAR"

        return {
            "transaction_id": transaction_id,
            "fraud_probability": float(calibrated_probability),
            "raw_fraud_probability": raw_probability,
            "predicted_class": predicted_class,
            "decision": decision,
            "threshold": self.threshold,
            "inference_backend": self.inference_backend,
        }

    def explain(
        self,
        transaction_id: str | None,
        features: dict[str, Any],
        top_n: int = 10,
    ) -> dict[str, Any]:
        """Explica una prediccion con TreeSHAP sobre la salida raw de XGBoost.

        Para XGBoost binario, la salida raw corresponde a log-odds (margen),
        no a la probabilidad calibrada que se usa para tomar la decision.
        """
        row = self._prepare_row(features)
        prediction = self.predict(transaction_id, features)

        try:
            import shap

            if self._shap_explainer is None:
                self._shap_explainer = shap.TreeExplainer(
                    self.model,
                    feature_perturbation="tree_path_dependent",
                    model_output="raw",
                )

            shap_result = self._shap_explainer.shap_values(row, check_additivity=True)
        except Exception as exc:
            raise RuntimeError(
                "No se pudo generar la explicacion SHAP. "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        # Compatibilidad con versiones de SHAP que devuelven lista por clase
        # y versiones que devuelven un array con dimension de salida.
        if isinstance(shap_result, list):
            shap_array = np.asarray(shap_result[-1], dtype=float)
        else:
            shap_array = np.asarray(shap_result, dtype=float)

        if shap_array.ndim == 3:
            shap_array = shap_array[0, :, -1]
        elif shap_array.ndim == 2:
            shap_array = shap_array[0]
        elif shap_array.ndim != 1:
            raise RuntimeError(
                f"Formato SHAP inesperado: dimensiones {shap_array.shape}."
            )

        expected_value = np.asarray(self._shap_explainer.expected_value, dtype=float).reshape(-1)
        base_value = float(expected_value[-1] if expected_value.size > 1 else expected_value[0])
        raw_model_output = float(self.model.predict(row, output_margin=True)[0])
        sum_shap_values = float(np.sum(shap_array))

        contributions = []
        for feature_name, feature_value, shap_value in zip(
            self.features, row.iloc[0].tolist(), shap_array.tolist()
        ):
            if shap_value > 0:
                impact = "aumenta_riesgo"
            elif shap_value < 0:
                impact = "reduce_riesgo"
            else:
                impact = "sin_impacto"

            contributions.append(
                {
                    "feature": feature_name,
                    "value": float(feature_value),
                    "shap_value": float(shap_value),
                    "impact": impact,
                }
            )

        contributions.sort(key=lambda item: abs(item["shap_value"]), reverse=True)
        top_features = contributions[: max(1, min(int(top_n), len(contributions)))]
        additivity_error = abs((base_value + sum_shap_values) - raw_model_output)

        return {
            "prediction": prediction,
            "explanation": {
                "method": "SHAP TreeExplainer",
                "model_output": "raw_log_odds",
                "base_value": base_value,
                "raw_model_output": raw_model_output,
                "sum_shap_values": sum_shap_values,
                "additivity_error": float(additivity_error),
                "top_features": top_features,
                "note": (
                    "Los valores SHAP explican la salida raw (log-odds) de XGBoost sin calibrar. "
                    "Los aportes positivos aumentan la salida del modelo hacia la clase fraude; "
                    "los negativos la reducen. La decision final sigue usando la probabilidad "
                    "calibrada y el umbral configurado."
                ),
            },
        }
