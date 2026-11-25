from typing import Union

from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt

NumberValue = Union[StrictFloat, StrictInt]


class PredictionRequest(BaseModel):
    """Solicitud de prediccion para una transaccion."""

    model_config = ConfigDict(extra="forbid")

    transaction_id: str | None = Field(default=None, max_length=100)
    features: dict[str, NumberValue]


class PredictionResponse(BaseModel):
    transaction_id: str | None
    fraud_probability: float
    raw_fraud_probability: float
    predicted_class: int
    decision: str
    threshold: float
    audit_event_status: str = "disabled"
    inference_backend: str = "local"


class ExplainRequest(PredictionRequest):
    """Solicitud de prediccion con explicabilidad local SHAP."""

    top_n: int = Field(default=10, ge=1, le=29)


class FeatureContribution(BaseModel):
    feature: str
    value: float
    shap_value: float
    impact: str


class ShapExplanation(BaseModel):
    method: str
    model_output: str
    base_value: float
    raw_model_output: float
    sum_shap_values: float
    additivity_error: float
    top_features: list[FeatureContribution]
    note: str


class ExplainResponse(BaseModel):
    prediction: PredictionResponse
    explanation: ShapExplanation


class BatchPredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    transactions: list[PredictionRequest] = Field(min_length=1, max_length=100)


class BatchPredictionResponse(BaseModel):
    total: int
    results: list[PredictionResponse]
