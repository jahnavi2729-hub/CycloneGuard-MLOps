import argparse
import json
from pathlib import Path
from typing import Optional

import joblib
import mlflow
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_TRACKING_URI = f"sqlite:///{(PROJECT_DIR / 'mlflow.db').as_posix()}"
DEFAULT_ARTIFACT_DIR = PROJECT_DIR / "artifacts"
DEFAULT_METRICS_PATH = (
    PROJECT_DIR / "outputs" / "results" / "lightgbm_performance_metrics.csv"
)


def track_existing_experiment(
    tracking_uri: str = DEFAULT_TRACKING_URI,
    artifacts_dir: Optional[Path] = None,
    metrics_path: Optional[Path] = None,
) -> str:
    artifact_directory = (
        artifacts_dir.resolve()
        if artifacts_dir is not None
        else DEFAULT_ARTIFACT_DIR.resolve()
    )
    performance_path = (
        metrics_path.resolve()
        if metrics_path is not None
        else DEFAULT_METRICS_PATH.resolve()
    )
    metadata_path = artifact_directory / "lightgbm_deployment_metadata.json"

    if not artifact_directory.is_dir():
        raise FileNotFoundError(
            f"Artifacts directory does not exist: {artifact_directory}"
        )
    if not performance_path.is_file():
        raise FileNotFoundError(
            f"Performance metrics CSV does not exist: {performance_path}"
        )
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Deployment metadata does not exist: {metadata_path}"
        )

    with metadata_path.open("r", encoding="utf-8") as metadata_file:
        metadata = json.load(metadata_file)

    sensor_columns = list(metadata["sensor_columns"])
    feature_columns = list(metadata["feature_columns"])
    lgbm_params = dict(metadata["lgbm_params"])

    if len(sensor_columns) != 6:
        raise ValueError(
            f"Expected 6 sensors, found {len(sensor_columns)}"
        )
    if len(feature_columns) != 53:
        raise ValueError(
            f"Expected 53 features, found {len(feature_columns)}"
        )

    metrics_df = pd.read_csv(performance_path)
    required_metric_columns = [
        "Sensor",
        "Train RMSE",
        "Test RMSE",
        "Train MAE",
        "Test MAE",
        "Train R2",
        "Test R2",
    ]
    missing_metrics = [
        column
        for column in required_metric_columns
        if column not in metrics_df.columns
    ]
    if missing_metrics:
        raise ValueError(
            "Performance metrics CSV is missing columns: "
            + ", ".join(missing_metrics)
        )

    model_paths = {}
    for sensor in sensor_columns:
        model_path = artifact_directory / f"lightgbm_{sensor}.joblib"
        if not model_path.is_file():
            raise FileNotFoundError(
                f"Missing saved model for {sensor}: {model_path}"
            )
        joblib.load(model_path)
        model_paths[sensor] = model_path

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("CycloneGuard_LightGBM")

    with mlflow.start_run(run_name="existing_exported_models") as run:
        mlflow.log_params(lgbm_params)
        mlflow.log_param("feature_count", len(feature_columns))
        mlflow.log_param("sensor_count", len(sensor_columns))
        mlflow.log_param("training_fraction", metadata["training_fraction"])
        mlflow.log_param(
            "feature_columns",
            json.dumps(feature_columns),
        )
        mlflow.log_param(
            "residual_standardization_ddof",
            metadata["residual_standardization"]["ddof"],
        )
        mlflow.log_param(
            "anomaly_percentile",
            metadata["anomaly_threshold"]["training_percentile"],
        )
        mlflow.log_param(
            "anomaly_threshold",
            metadata["anomaly_threshold"]["value"],
        )

        mlflow.set_tags(
            {
                "project": "CycloneGuard",
                "model_type": "LightGBM",
                "training_status": "already_trained",
                "retraining_performed": "false",
                "data_split": "chronological_80_20",
            }
        )

        for _, row in metrics_df.iterrows():
            sensor = str(row["Sensor"])
            if sensor not in sensor_columns:
                continue
            metric_prefix = sensor
            for source_name, metric_name in [
                ("Train RMSE", "train_rmse"),
                ("Test RMSE", "test_rmse"),
                ("Train MAE", "train_mae"),
                ("Test MAE", "test_mae"),
                ("Train R2", "train_r2"),
                ("Test R2", "test_r2"),
            ]:
                mlflow.log_metric(
                    f"{metric_prefix}.{metric_name}",
                    float(row[source_name]),
                )

        for metric_column in [
            "Train RMSE",
            "Test RMSE",
            "Train MAE",
            "Test MAE",
            "Train R2",
            "Test R2",
        ]:
            mlflow.log_metric(
                f"macro.{metric_column.lower().replace(' ', '_')}",
                float(metrics_df[metric_column].mean()),
            )

        mlflow.log_artifact(str(metadata_path), artifact_path="deployment")
        mlflow.log_artifact(str(performance_path), artifact_path="evaluation")

        for sensor, model_path in model_paths.items():
            mlflow.log_artifact(
                str(model_path),
                artifact_path=f"models/{sensor}",
            )

        residual_metadata = metadata["residual_standardization"]
        for sensor in sensor_columns:
            mlflow.log_metric(
                f"{sensor}.residual_mean",
                float(residual_metadata["mean_by_sensor"][sensor]),
            )
            mlflow.log_metric(
                f"{sensor}.residual_std",
                float(residual_metadata["std_by_sensor"][sensor]),
            )

        mlflow.log_artifact(
            str(Path(__file__).resolve()),
            artifact_path="tracking",
        )

        return run.info.run_id


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Log the existing CycloneGuard experiment to MLflow."
    )
    parser.add_argument(
        "--tracking-uri",
        default=DEFAULT_TRACKING_URI,
        help="MLflow tracking URI.",
    )
    parser.add_argument(
        "--artifacts-dir",
        default=None,
        help="Directory containing the six joblib models and metadata.",
    )
    parser.add_argument(
        "--metrics-path",
        default=None,
        help="Existing lightgbm_performance_metrics.csv path.",
    )
    args = parser.parse_args()

    run_id = track_existing_experiment(
        tracking_uri=args.tracking_uri,
        artifacts_dir=(
            Path(args.artifacts_dir).resolve()
            if args.artifacts_dir
            else None
        ),
        metrics_path=(
            Path(args.metrics_path).resolve()
            if args.metrics_path
            else None
        ),
    )
    print(f"Logged existing CycloneGuard run: {run_id}")


if __name__ == "__main__":
    main()
