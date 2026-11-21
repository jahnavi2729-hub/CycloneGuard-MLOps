import io
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse


PROJECT_DIR = Path(__file__).resolve().parent
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from inference import InferenceEngine


MAX_UPLOAD_BYTES = 50 * 1024 * 1024

app = FastAPI(
    title="CycloneGuard Anomaly Inference API",
    version="1.0.0",
)

try:
    inference_engine = InferenceEngine()
    startup_error = None
except Exception as exc:
    inference_engine = None
    startup_error = f"{type(exc).__name__}: {exc}"


@app.get("/health")
def health() -> dict[str, Any]:
    if inference_engine is None:
        raise HTTPException(
            status_code=503,
            detail=f"Inference engine unavailable: {startup_error}",
        )

    return {
        "status": "healthy",
        "model_count": len(inference_engine.models),
        "feature_count": len(inference_engine.feature_columns),
        "sensor_count": len(inference_engine.sensor_columns),
        "anomaly_threshold": inference_engine.anomaly_threshold,
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    download: bool = False,
) -> Any:
    if inference_engine is None:
        raise HTTPException(
            status_code=503,
            detail=f"Inference engine unavailable: {startup_error}",
        )

    filename = file.filename or ""
    if not filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Uploaded file must have a .csv filename.",
        )

    content_type = (file.content_type or "").lower()
    allowed_types = {
        "text/csv",
        "application/csv",
        "application/vnd.ms-excel",
        "application/octet-stream",
    }
    if content_type and content_type not in allowed_types:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported content type '{content_type}'. Upload a CSV file.",
        )

    try:
        csv_bytes = await file.read(MAX_UPLOAD_BYTES + 1)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Could not read uploaded file: {type(exc).__name__}",
        ) from exc

    if not csv_bytes:
        raise HTTPException(
            status_code=400,
            detail="Uploaded CSV file is empty.",
        )
    if len(csv_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"CSV exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB upload limit.",
        )

    try:
        csv_text = csv_bytes.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise HTTPException(
            status_code=400,
            detail="CSV must be UTF-8 encoded.",
        ) from exc

    try:
        input_df = pd.read_csv(io.StringIO(csv_text))
    except pd.errors.EmptyDataError as exc:
        raise HTTPException(
            status_code=400,
            detail="Uploaded CSV contains no data.",
        ) from exc
    except pd.errors.ParserError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid CSV format: {exc}",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Could not parse CSV: {type(exc).__name__}: {exc}",
        ) from exc

    if input_df.empty:
        raise HTTPException(
            status_code=400,
            detail="Uploaded CSV contains no rows.",
        )

    try:
        result_df = inference_engine.predict(input_df)
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {type(exc).__name__}: {exc}",
        ) from exc

    if "time_dt" in result_df.columns:
        time_values = pd.to_datetime(
            result_df["time_dt"],
            errors="coerce",
        )
        result_df["time_dt"] = time_values.map(
            lambda value: value.isoformat() if not pd.isna(value) else None
        )

    anomaly_count = int(result_df["is_anomaly"].sum()) if "is_anomaly" in result_df else 0
    row_count = len(result_df)

    if download:
        csv_buffer = io.StringIO()
        result_df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)
        return StreamingResponse(
            iter([csv_buffer.getvalue()]),
            media_type="text/csv",
            headers={
                "Content-Disposition": 'attachment; filename="cycloneguard_results.csv"',
            },
        )

    anomaly_df = result_df[result_df["is_anomaly"]].copy() if "is_anomaly" in result_df else result_df.iloc[0:0].copy()
    anomaly_records = anomaly_df.where(
        pd.notna(anomaly_df),
        None,
    ).to_dict(orient="records")

    score_series = pd.to_numeric(
        result_df["anomaly_score"],
        errors="coerce",
    )
    anomaly_rate = (
        (anomaly_count / row_count * 100)
        if row_count
        else 0.0
    )

    return {
        "summary": {
            "row_count": row_count,
            "anomaly_count": anomaly_count,
            "anomaly_rate_percent": round(float(anomaly_rate), 4),
            "anomaly_threshold": inference_engine.anomaly_threshold,
            "anomaly_score_min": round(float(score_series.min()), 6) if not score_series.empty else None,
            "anomaly_score_max": round(float(score_series.max()), 6) if not score_series.empty else None,
            "anomaly_score_mean": round(float(score_series.mean()), 6) if not score_series.empty else None,
        },
        "anomalous_records": anomaly_records,
    }
