import io
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles


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

FRONTEND_DIR = PROJECT_DIR / "frontend"
STATIC_DIR = FRONTEND_DIR / "static"
TEMPLATE_DIR = FRONTEND_DIR / "templates"
OUTPUTS_DIR = PROJECT_DIR / "outputs" / "results"

HISTORICAL_FILES = {
    "test_scores": "lightgbm_test_scores.csv",
    "test_predictions": "lightgbm_test_predictions.csv",
    "abnormal_episodes": "lightgbm_abnormal_episodes.csv",
    "performance_metrics": "lightgbm_performance_metrics.csv",
    "feature_importance": "lightgbm_feature_importance.csv",
}

SENSOR_COLUMNS = (
    "Cyclone_Inlet_Gas_Temp",
    "Cyclone_Material_Temp",
    "Cyclone_Outlet_Gas_draft",
    "Cyclone_cone_draft",
    "Cyclone_Gas_Outlet_Temp",
    "Cyclone_Inlet_Draft",
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

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


@lru_cache(maxsize=None)
def load_historical_csv(dataset: str) -> pd.DataFrame:
    if dataset not in HISTORICAL_FILES:
        raise HTTPException(
            status_code=404,
            detail="Unknown historical dataset.",
        )

    path = OUTPUTS_DIR / HISTORICAL_FILES[dataset]
    if not path.is_file():
        raise HTTPException(
            status_code=500,
            detail=f"Historical dataset '{dataset}' is unavailable.",
        )

    try:
        return pd.read_csv(path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, OSError) as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Historical dataset '{dataset}' could not be loaded.",
        ) from exc


def validate_historical_columns(
    dataset: str,
    frame: pd.DataFrame,
    required_columns: tuple[str, ...],
) -> None:
    missing = sorted(set(required_columns).difference(frame.columns))
    if missing:
        raise HTTPException(
            status_code=500,
            detail=f"Historical dataset '{dataset}' has an invalid schema.",
        )


def validate_time_range(
    start: pd.Timestamp | None,
    end: pd.Timestamp | None,
) -> None:
    if start is not None and end is not None and start > end:
        raise HTTPException(
            status_code=422,
            detail="'start' must be earlier than or equal to 'end'.",
        )


def filter_by_time(
    frame: pd.DataFrame,
    column: str,
    start: pd.Timestamp | None,
    end: pd.Timestamp | None,
) -> pd.DataFrame:
    result = frame.copy()
    parsed = pd.to_datetime(result[column], errors="coerce")
    if start is not None:
        result = result.loc[parsed >= start]
        parsed = parsed.loc[result.index]
    if end is not None:
        result = result.loc[parsed <= end]
    return result


def json_safe_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    safe = frame.copy()
    for column in safe.columns:
        if pd.api.types.is_datetime64_any_dtype(safe[column]):
            safe[column] = safe[column].map(
                lambda value: value.isoformat()
                if not pd.isna(value)
                else None
            )

    return (
        safe.where(pd.notna(safe), None)
        .to_dict(orient="records")
    )


def paginated_records(
    frame: pd.DataFrame,
    limit: int,
    offset: int,
) -> dict[str, Any]:
    return {
        "total": len(frame),
        "limit": limit,
        "offset": offset,
        "records": json_safe_records(frame.iloc[offset : offset + limit]),
    }


@app.get("/", response_class=HTMLResponse)
def dashboard() -> HTMLResponse:
    index_path = TEMPLATE_DIR / "index.html"
    if not index_path.is_file():
        raise HTTPException(
            status_code=503,
            detail="Dashboard template is unavailable.",
        )

    try:
        return HTMLResponse(index_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise HTTPException(
            status_code=500,
            detail="Dashboard template could not be loaded.",
        ) from exc


@app.get("/api/history/summary")
def historical_summary() -> dict[str, Any]:
    try:
        scores = load_historical_csv("test_scores")
        predictions = load_historical_csv("test_predictions")
        episodes = load_historical_csv("abnormal_episodes")
        metrics = load_historical_csv("performance_metrics")

        validate_historical_columns(
            "test_scores",
            scores,
            ("time_dt", "anomaly_score", "is_anomaly"),
        )
        validate_historical_columns(
            "test_predictions",
            predictions,
            ("time_dt",),
        )
        validate_historical_columns(
            "abnormal_episodes",
            episodes,
            ("episode_id", "start_time", "end_time", "flagged_points"),
        )
        validate_historical_columns(
            "performance_metrics",
            metrics,
            ("Sensor",),
        )

        parsed_times = pd.to_datetime(scores["time_dt"], errors="coerce")
        valid_scores = pd.to_numeric(
            scores["anomaly_score"],
            errors="coerce",
        )
        anomaly_flags = scores["is_anomaly"].astype(str).str.lower() == "true"
        anomaly_count = int(anomaly_flags.sum())
        row_count = len(scores)
        score_min = valid_scores.min()
        score_max = valid_scores.max()
        score_mean = valid_scores.mean()

        return {
            "test_row_count": row_count,
            "prediction_row_count": len(predictions),
            "anomaly_count": anomaly_count,
            "anomaly_rate_percent": round(
                anomaly_count / row_count * 100,
                4,
            ) if row_count else 0.0,
            "anomaly_score_min": round(float(score_min), 6)
            if pd.notna(score_min)
            else None,
            "anomaly_score_max": round(float(score_max), 6)
            if pd.notna(score_max)
            else None,
            "anomaly_score_mean": round(float(score_mean), 6)
            if pd.notna(score_mean)
            else None,
            "historical_range": {
                "start": parsed_times.min().isoformat()
                if parsed_times.notna().any()
                else None,
                "end": parsed_times.max().isoformat()
                if parsed_times.notna().any()
                else None,
            },
            "episode_count": len(episodes),
            "flagged_episode_points": int(
                pd.to_numeric(
                    episodes["flagged_points"],
                    errors="coerce",
                ).fillna(0).sum()
            ),
            "sensor_count": len(SENSOR_COLUMNS),
            "sensors": list(SENSOR_COLUMNS),
            "model_count": len(metrics),
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Could not build historical summary.",
        ) from exc


@app.get("/api/history/scores")
def historical_scores(
    start: str | None = None,
    end: str | None = None,
    anomalies_only: bool = False,
    limit: int = Query(default=500, ge=1, le=5000),
    offset: int = Query(default=0, ge=0),
) -> dict[str, Any]:
    scores = load_historical_csv("test_scores")
    validate_historical_columns(
        "test_scores",
        scores,
        ("time_dt", "anomaly_score", "is_anomaly"),
    )

    start_dt = pd.to_datetime(start, errors="coerce") if start else None
    end_dt = pd.to_datetime(end, errors="coerce") if end else None
    if start and pd.isna(start_dt):
        raise HTTPException(status_code=422, detail="Invalid 'start' timestamp.")
    if end and pd.isna(end_dt):
        raise HTTPException(status_code=422, detail="Invalid 'end' timestamp.")
    validate_time_range(start_dt, end_dt)

    filtered = filter_by_time(scores, "time_dt", start_dt, end_dt)
    if anomalies_only:
        filtered = filtered[
            filtered["is_anomaly"].astype(str).str.lower() == "true"
        ]

    return paginated_records(filtered, limit, offset)


@app.get("/api/history/predictions")
def historical_predictions(
    start: str | None = None,
    end: str | None = None,
    sensor: str | None = None,
    limit: int = Query(default=500, ge=1, le=5000),
    offset: int = Query(default=0, ge=0),
) -> dict[str, Any]:
    if sensor is not None and sensor not in SENSOR_COLUMNS:
        raise HTTPException(
            status_code=422,
            detail="Invalid sensor name.",
        )

    predictions = load_historical_csv("test_predictions")
    validate_historical_columns(
        "test_predictions",
        predictions,
        ("time_dt",),
    )

    start_dt = pd.to_datetime(start, errors="coerce") if start else None
    end_dt = pd.to_datetime(end, errors="coerce") if end else None
    if start and pd.isna(start_dt):
        raise HTTPException(status_code=422, detail="Invalid 'start' timestamp.")
    if end and pd.isna(end_dt):
        raise HTTPException(status_code=422, detail="Invalid 'end' timestamp.")
    validate_time_range(start_dt, end_dt)

    filtered = filter_by_time(
        predictions,
        "time_dt",
        start_dt,
        end_dt,
    )

    if sensor is not None:
        required = [
            "time_dt",
            f"{sensor}_actual",
            f"{sensor}_pred",
            f"{sensor}_residual",
        ]
        validate_historical_columns(
            "test_predictions",
            filtered,
            tuple(required),
        )
        filtered = filtered[required]

    return paginated_records(filtered, limit, offset)


@app.get("/api/history/episodes")
def historical_episodes(
    limit: int = Query(default=500, ge=1, le=5000),
    offset: int = Query(default=0, ge=0),
) -> dict[str, Any]:
    episodes = load_historical_csv("abnormal_episodes")
    validate_historical_columns(
        "abnormal_episodes",
        episodes,
        (
            "episode_id",
            "start_time",
            "end_time",
            "duration_minutes",
            "flagged_points",
            "max_anomaly_score",
            "mean_anomaly_score",
            "primary_sensor_contributor",
        ),
    )
    return paginated_records(episodes, limit, offset)


@app.get("/api/history/performance")
def historical_performance(
    limit: int = Query(default=500, ge=1, le=5000),
    offset: int = Query(default=0, ge=0),
) -> dict[str, Any]:
    metrics = load_historical_csv("performance_metrics")
    validate_historical_columns(
        "performance_metrics",
        metrics,
        (
            "Sensor",
            "Train RMSE",
            "Test RMSE",
            "Train MAE",
            "Test MAE",
            "Train R2",
            "Test R2",
        ),
    )
    return paginated_records(metrics, limit, offset)
