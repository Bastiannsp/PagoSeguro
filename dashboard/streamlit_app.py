from __future__ import annotations

import json
import os
from typing import Any

import pandas as pd
import requests
import streamlit as st

from audit_store import init_db, save_event, load_recent_events

st.set_page_config(
    page_title="PagoSeguro | Deteccion de fraude",
    page_icon="🛡️",
    layout="wide",
)

DB_PATH = os.getenv("AUDIT_DB_PATH", "data/audit.db")
init_db(DB_PATH)


def api_headers(api_key: str) -> dict[str, str]:
    return {"X-API-Key": api_key, "Content-Type": "application/json"}


def api_get(base_url: str, path: str, timeout: int = 15, api_key: str | None = None) -> Any:
    headers = api_headers(api_key) if api_key else {}
    response = requests.get(f"{base_url.rstrip('/')}{path}", headers=headers, timeout=timeout)
    response.raise_for_status()
    return response.json()


def api_post(
    base_url: str,
    path: str,
    api_key: str,
    payload: dict[str, Any],
    timeout: int = 60,
) -> Any:
    response = requests.post(
        f"{base_url.rstrip('/')}{path}",
        headers=api_headers(api_key),
        json=payload,
        timeout=timeout,
    )
    if not response.ok:
        try:
            detail = response.json()
        except ValueError:
            detail = response.text
        raise RuntimeError(f"HTTP {response.status_code}: {detail}")
    return response.json()


def normalize_request(raw_text: str) -> dict[str, Any]:
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON invalido: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError("El contenido debe ser un objeto JSON.")

    # Acepta tanto el cuerpo completo como un objeto que contenga solo las features.
    if "features" not in data:
        data = {"transaction_id": "TX-MANUAL-001", "features": data}

    if not isinstance(data.get("features"), dict):
        raise ValueError("El campo 'features' debe ser un objeto JSON.")

    data.setdefault("transaction_id", "TX-MANUAL-001")
    return data


def show_prediction(result: dict[str, Any]) -> None:
    probability = float(result.get("fraud_probability", 0.0))
    threshold = float(result.get("threshold", 0.0))
    decision = result.get("decision", "SIN DECISION")
    predicted_class = int(result.get("predicted_class", 0))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Probabilidad de fraude", f"{probability * 100:.4f}%")
    c2.metric("Umbral configurado", f"{threshold * 100:.4f}%")
    c3.metric("Clase predicha", str(predicted_class))
    c4.metric("Decision", decision)

    if predicted_class == 1:
        st.warning("La transaccion supero el umbral y requiere revision o rechazo.")
    else:
        st.success("La transaccion quedo bajo el umbral y fue autorizada por el modelo.")

    event_status = result.get("audit_event_status", "disabled")
    if event_status == "queued":
        st.info("Evento de auditoria encolado en Kafka. El worker generara SHAP y guardara el registro en PostgreSQL en segundo plano.")
    elif event_status == "unavailable":
        st.warning("No se pudo encolar el evento de auditoria. La prediccion se completo, pero revisa la conexion con Kafka.")
    else:
        st.caption("Publicacion asincrona de auditoria: " + str(event_status))

    with st.expander("Ver respuesta completa de la API"):
        st.json(result)


def shap_dataframe(explanation: dict[str, Any]) -> pd.DataFrame:
    rows = explanation.get("top_features", [])
    if not rows:
        return pd.DataFrame(columns=["variable", "aporte SHAP", "impacto", "valor"])
    frame = pd.DataFrame(rows)
    frame = frame.rename(
        columns={"feature": "variable", "shap_value": "aporte SHAP", "impact": "impacto", "value": "valor"}
    )
    frame["magnitud"] = frame["aporte SHAP"].abs()
    return frame.sort_values("magnitud", ascending=True)


def save_prediction_event(
    *,
    transaction_id: str,
    source: str,
    operation: str,
    features: dict[str, Any],
    prediction: dict[str, Any],
    explanation: dict[str, Any] | None = None,
) -> None:
    save_event(
        db_path=DB_PATH,
        transaction_id=transaction_id,
        source=source,
        operation=operation,
        features=features,
        prediction=prediction,
        explanation=explanation,
    )


st.title("🛡️ PagoSeguro")
st.caption("Panel de evaluacion de transacciones y consulta de explicabilidad del modelo")

with st.sidebar:
    st.header("Conexion con la API")
    base_url = st.text_input(
        "URL base de FastAPI",
        value=os.getenv("PAGOSEGURO_API_URL", "http://127.0.0.1:8000"),
    ).strip().rstrip("/")
    api_key = st.text_input(
        "API Key",
        value=os.getenv("API_KEY", ""),
        type="password",
        help="Se envia mediante el encabezado X-API-Key. No se guarda en el historial.",
    )
    if st.button("Comprobar conexion", use_container_width=True):
        try:
            health = api_get(base_url, "/health")
            model_info = api_get(base_url, "/model-info")
            st.session_state["api_health"] = health
            st.session_state["model_info"] = model_info
            st.success("API y modelo disponibles.")
        except Exception as exc:
            st.error(f"No se pudo conectar: {exc}")

    if "api_health" in st.session_state and "model_info" in st.session_state:
        st.divider()
        st.caption("Estado detectado")
        st.json(st.session_state["api_health"])
        info = st.session_state["model_info"]
        st.metric("Variables del modelo", info.get("features_count", "-"))
        st.metric("Umbral", f"{float(info.get('threshold', 0)):.8f}")
        st.caption(f"Modelo: {info.get('model_type', 'No informado')}")

if not base_url:
    st.error("Debes indicar la URL base de la API.")
    st.stop()

# Intento silencioso de obtener la metadata para conocer el orden/columnas esperadas.
try:
    model_info = api_get(base_url, "/model-info")
    expected_features = [str(item) for item in model_info.get("features", [])]
except Exception:
    model_info = None
    expected_features = []

if not expected_features:
    st.info("Inicia FastAPI y comprueba la conexion desde el panel lateral para cargar las variables del modelo.")
    st.stop()

st.caption(f"Modelo: {model_info.get('model_type', 'XGBoost')} · Variables esperadas: {len(expected_features)} · Umbral: {float(model_info.get('threshold', 0)):.8f}")

tab_single, tab_csv, tab_history = st.tabs(
    ["Evaluar transaccion", "Procesar CSV", "Historial de auditoria"]
)

with tab_single:
    st.subheader("Prediccion individual")
    st.write(
        "Pega el JSON completo de una solicitud o solamente el objeto `features`. "
        "Las variables deben coincidir con las que aparecen en `/model-info`."
    )
    default_features = {name: 0.0 for name in expected_features}
    default_payload = {"transaction_id": "TX-MANUAL-001", "features": default_features}
    raw_payload = st.text_area(
        "Datos de la transaccion (JSON)",
        value=json.dumps(default_payload, indent=2),
        height=360,
    )
    explain = st.checkbox("Generar tambien explicacion SHAP", value=True)
    top_n = st.slider("Cantidad de variables SHAP", min_value=3, max_value=min(20, len(expected_features)), value=min(10, len(expected_features)))

    if st.button("Evaluar transaccion", type="primary", use_container_width=True):
        if not api_key:
            st.error("Ingresa la API Key en el panel lateral.")
        else:
            try:
                payload = normalize_request(raw_payload)
                # La prediccion publica el evento a Kafka; SHAP se consulta por separado.
                prediction = api_post(base_url, "/v1/predict", api_key, payload)
                shap_result = None
                if explain:
                    request_body = {**payload, "top_n": top_n}
                    explanation_response = api_post(base_url, "/v1/explain", api_key, request_body)
                    shap_result = explanation_response.get("explanation", {})
                save_prediction_event(
                    transaction_id=str(payload.get("transaction_id") or "TX-SIN-ID"),
                    source="dashboard_manual",
                    operation="explain" if explain else "predict",
                    features=payload["features"],
                    prediction=prediction,
                    explanation=shap_result,
                )
                st.session_state["last_prediction"] = prediction
                if shap_result is not None:
                    st.session_state["last_explanation"] = shap_result
                else:
                    st.session_state.pop("last_explanation", None)
            except Exception as exc:
                st.error(str(exc))

    if "last_prediction" in st.session_state:
        st.divider()
        show_prediction(st.session_state["last_prediction"])
    if "last_explanation" in st.session_state:
        st.subheader("Explicacion SHAP")
        shap_df = shap_dataframe(st.session_state["last_explanation"])
        if not shap_df.empty:
            st.bar_chart(shap_df.set_index("variable")["aporte SHAP"])
            st.dataframe(
                shap_df[["variable", "valor", "aporte SHAP", "impacto"]].sort_values(
                    "aporte SHAP", key=lambda col: col.abs(), ascending=False
                ),
                use_container_width=True,
                hide_index=True,
            )
        st.caption(st.session_state["last_explanation"].get("note", ""))
        with st.expander("Detalles tecnicos SHAP"):
            st.json(st.session_state["last_explanation"])

with tab_csv:
    st.subheader("Procesamiento por lotes")
    st.write(
        "Carga un CSV con una columna por variable del modelo. Se permite una columna "
        "`transaction_id` opcional y una columna `Class` opcional para comparar resultados "
        "en un conjunto de prueba. `Class` nunca se envia al modelo."
    )
    st.info("La API procesa hasta 100 transacciones por solicitud. El dashboard divide automaticamente archivos mayores en bloques de 100.")
    uploaded = st.file_uploader("Seleccionar archivo CSV", type=["csv"], key="csv_upload")

    if uploaded is not None:
        try:
            csv_df = pd.read_csv(uploaded)
            st.write(f"Filas: {len(csv_df)} · Columnas: {len(csv_df.columns)}")
            st.dataframe(csv_df.head(8), use_container_width=True)
            missing = [feature for feature in expected_features if feature not in csv_df.columns]
            allowed_extra = {"transaction_id", "Class", "Time"}
            unexpected = [col for col in csv_df.columns if col not in expected_features and col not in allowed_extra]
            if missing:
                st.error(f"Faltan variables requeridas: {missing}")
            if unexpected:
                st.warning(f"Se ignoraran columnas no utilizadas por el modelo: {unexpected}")
            if "Time" in csv_df.columns and "Time" not in expected_features:
                st.warning("El CSV contiene `Time`, pero esta variable no forma parte del modelo entrenado; se excluira.")
            if "Class" in csv_df.columns:
                st.caption("La columna Class se usara solo para calcular metricas de evaluacion local; no se enviara a FastAPI.")

            max_rows = st.number_input("Maximo de filas a procesar", min_value=1, max_value=10000, value=min(len(csv_df), 500), step=100)
            selected_df = csv_df.head(int(max_rows)).copy()
            if st.button("Procesar transacciones", type="primary", disabled=bool(missing) or len(selected_df) == 0):
                if not api_key:
                    st.error("Ingresa la API Key en el panel lateral.")
                else:
                    results: list[dict[str, Any]] = []
                    features_list: list[dict[str, float]] = []
                    progress = st.progress(0.0)
                    status = st.empty()
                    total = len(selected_df)
                    try:
                        for start in range(0, total, 100):
                            chunk = selected_df.iloc[start : start + 100]
                            transactions = []
                            for local_idx, (_, row) in enumerate(chunk.iterrows()):
                                absolute_idx = start + local_idx
                                txid = str(row.get("transaction_id", f"CSV-{absolute_idx + 1:06d}"))
                                features = {name: float(row[name]) for name in expected_features}
                                transactions.append({"transaction_id": txid, "features": features})
                                features_list.append(features)
                            batch_result = api_post(
                                base_url,
                                "/v1/predict/batch",
                                api_key,
                                {"transactions": transactions},
                                timeout=120,
                            )
                            chunk_results = batch_result.get("results", [])
                            results.extend(chunk_results)
                            for tx, pred in zip(transactions, chunk_results):
                                save_prediction_event(
                                    transaction_id=str(tx["transaction_id"]),
                                    source="dashboard_csv",
                                    operation="predict",
                                    features=tx["features"],
                                    prediction=pred,
                                )
                            progress.progress(min((start + len(chunk)) / total, 1.0))
                            status.write(f"Procesadas {len(results)} de {total} transacciones...")

                        result_df = pd.DataFrame(results)
                        st.session_state["csv_results"] = result_df
                        st.session_state["csv_source"] = selected_df.reset_index(drop=True)
                        status.success(f"Procesamiento terminado: {len(results)} transacciones.")
                    except Exception as exc:
                        st.error(f"Error durante el procesamiento: {exc}")
        except Exception as exc:
            st.error(f"No se pudo leer el archivo CSV: {exc}")

    if "csv_results" in st.session_state:
        result_df = st.session_state["csv_results"].copy()
        source_df = st.session_state.get("csv_source", pd.DataFrame())
        st.divider()
        st.subheader("Resultados del procesamiento")
        if not result_df.empty:
            counts = result_df["decision"].value_counts().rename_axis("decision").reset_index(name="cantidad")
            m1, m2, m3 = st.columns(3)
            m1.metric("Transacciones evaluadas", len(result_df))
            m2.metric("Autorizadas", int((result_df["predicted_class"] == 0).sum()))
            m3.metric("Marcadas", int((result_df["predicted_class"] == 1).sum()))
            st.bar_chart(counts.set_index("decision"))
            st.dataframe(result_df, use_container_width=True, hide_index=True)

            if "Class" in source_df.columns:
                try:
                    y_true = pd.to_numeric(source_df["Class"], errors="raise").astype(int)
                    y_pred = result_df["predicted_class"].astype(int)
                    tp = int(((y_true == 1) & (y_pred == 1)).sum())
                    tn = int(((y_true == 0) & (y_pred == 0)).sum())
                    fp = int(((y_true == 0) & (y_pred == 1)).sum())
                    fn = int(((y_true == 1) & (y_pred == 0)).sum())
                    precision = tp / (tp + fp) if tp + fp else 0.0
                    recall = tp / (tp + fn) if tp + fn else 0.0
                    accuracy = (tp + tn) / len(y_true) if len(y_true) else 0.0
                    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
                    st.subheader("Metricas contra la etiqueta real Class")
                    a, b, c, d = st.columns(4)
                    a.metric("Accuracy", f"{accuracy:.4f}")
                    b.metric("Precision", f"{precision:.4f}")
                    c.metric("Recall", f"{recall:.4f}")
                    d.metric("F1-score", f"{f1:.4f}")
                    st.write({"TP": tp, "TN": tn, "FP": fp, "FN": fn})
                except Exception as exc:
                    st.warning(f"No fue posible calcular metricas con Class: {exc}")

            csv_bytes = result_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Descargar resultados CSV",
                data=csv_bytes,
                file_name="pagoseguro_resultados.csv",
                mime="text/csv",
            )

            row_to_explain = st.number_input(
                "Numero de fila para explicar con SHAP (1 = primera fila)",
                min_value=1,
                max_value=len(source_df),
                value=1,
                step=1,
            )
            if st.button("Generar explicacion SHAP de esta fila"):
                try:
                    idx = int(row_to_explain) - 1
                    source_row = source_df.iloc[idx]
                    txid = str(source_row.get("transaction_id", f"CSV-{idx + 1:06d}"))
                    features = {name: float(source_row[name]) for name in expected_features}
                    explanation_response = api_post(
                        base_url,
                        "/v1/explain",
                        api_key,
                        {"transaction_id": txid, "features": features, "top_n": 10},
                    )
                    explanation = explanation_response.get("explanation", {})
                    prediction = explanation_response.get("prediction", {})
                    save_prediction_event(
                        transaction_id=txid,
                        source="dashboard_csv",
                        operation="explain",
                        features=features,
                        prediction=prediction,
                        explanation=explanation,
                    )
                    st.session_state["csv_explanation"] = explanation
                    st.session_state["csv_explanation_prediction"] = prediction
                except Exception as exc:
                    st.error(f"No se pudo generar SHAP: {exc}")

            if "csv_explanation" in st.session_state:
                st.subheader("Explicacion SHAP de la fila seleccionada")
                shap_df = shap_dataframe(st.session_state["csv_explanation"])
                if not shap_df.empty:
                    st.bar_chart(shap_df.set_index("variable")["aporte SHAP"])
                    st.dataframe(shap_df[["variable", "valor", "aporte SHAP", "impacto"]], use_container_width=True, hide_index=True)
                st.caption(st.session_state["csv_explanation"].get("note", ""))

with tab_history:
    st.subheader("Historial local de auditoria")
    st.caption("Historial local del dashboard (SQLite). Mas abajo se muestra tambien la auditoria distribuida procesada por Kafka y PostgreSQL.")
    col_a, col_b = st.columns([1, 4])
    with col_a:
        limit = st.selectbox("Registros", [25, 50, 100, 250], index=1)
    if st.button("Actualizar historial"):
        st.rerun()
    history = load_recent_events(DB_PATH, limit=int(limit))
    if history.empty:
        st.info("Todavia no hay predicciones guardadas desde este dashboard.")
    else:
        h1, h2, h3 = st.columns(3)
        h1.metric("Registros", len(history))
        h2.metric("Autorizadas", int((history["decision"] == "AUTORIZAR").sum()))
        h3.metric("Marcadas", int((history["decision"] != "AUTORIZAR").sum()))
        st.dataframe(history, use_container_width=True, hide_index=True)
        st.download_button(
            "Descargar historial CSV",
            data=history.to_csv(index=False).encode("utf-8"),
            file_name="pagoseguro_auditoria_local.csv",
            mime="text/csv",
        )

with tab_history:
    st.divider()
    st.subheader("Auditoria distribuida Kafka + PostgreSQL")
    st.caption("Estos registros los genera el worker asincrono despues de consumir eventos desde Kafka y calcular SHAP.")
    if st.button("Actualizar auditoria distribuida"):
        st.rerun()
    if not api_key:
        st.info("Ingresa la API Key en el panel lateral para consultar la auditoria remota.")
    else:
        try:
            remote = api_get(base_url, "/v1/audit/recent?limit=100", api_key=api_key, timeout=20)
            remote_df = pd.DataFrame(remote.get("events", []))
            if remote_df.empty:
                st.info("Todavia no hay eventos procesados por el worker. Realiza una prediccion desde /v1/predict y espera unos segundos.")
            else:
                r1, r2, r3 = st.columns(3)
                r1.metric("Eventos almacenados", len(remote_df))
                r2.metric("Completados", int((remote_df["processing_status"] == "completed").sum()))
                r3.metric("Con error", int((remote_df["processing_status"] == "failed").sum()))
                columns_to_show = [
                    col for col in ["processed_at", "transaction_id", "fraud_probability", "decision", "processing_status"]
                    if col in remote_df.columns
                ]
                st.dataframe(remote_df[columns_to_show], use_container_width=True, hide_index=True)
                with st.expander("Ver datos completos de auditoria"):
                    st.dataframe(remote_df, use_container_width=True, hide_index=True)
        except Exception as exc:
            st.warning(f"Auditoria distribuida no disponible: {exc}. Verifica que Docker Compose tenga Kafka, PostgreSQL, API y worker ejecutandose.")

st.divider()
st.caption("Proyecto academico PagoSeguro · El dashboard es una interfaz local. No sustituye un sistema de pagos productivo ni demuestra por si solo los SLA del caso.")
