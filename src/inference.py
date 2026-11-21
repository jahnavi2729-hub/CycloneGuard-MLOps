import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Union

import joblib
import numpy as np
import pandas as pd


class InferenceEngine:
    def __init__(
        self,
        metadata_path: Optional[Union[str, Path]] = None,
        artifacts_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        project_dir = Path(__file__).resolve().parent.parent
        default_artifacts_dir = project_dir / "artifacts"

        self.artifacts_dir = (
            Path(artifacts_dir).expanduser().resolve()
            if artifacts_dir is not None
            else default_artifacts_dir.resolve()
        )
        self.metadata_path = (
            Path(metadata_path).expanduser().resolve()
            if metadata_path is not None
            else self.artifacts_dir / "lightgbm_deployment_metadata.json"
        )

        with self.metadata_path.open("r", encoding="utf-8") as metadata_file:
            self.metadata = json.load(metadata_file)

        self.sensor_columns = list(self.metadata["sensor_columns"])
        self.feature_columns = list(self.metadata["feature_columns"])
        if len(self.feature_columns) != 53:
            raise ValueError(
                f"Expected 53 metadata features, found {len(self.feature_columns)}"
            )
        if len(self.sensor_columns) != 6:
            raise ValueError(
                f"Expected 6 sensor columns, found {len(self.sensor_columns)}"
            )

        residual_metadata = self.metadata["residual_standardization"]
        if residual_metadata["ddof"] != 1:
            raise ValueError("The exported residual statistics must use ddof=1")

        self.residual_means = pd.Series(
            residual_metadata["mean_by_sensor"], dtype=float
        ).reindex(self.sensor_columns)
        self.residual_stds = pd.Series(
            residual_metadata["std_by_sensor"], dtype=float
        ).reindex(self.sensor_columns)
        self.anomaly_threshold = float(
            self.metadata["anomaly_threshold"]["value"]
        )

        if self.residual_means.isna().any() or self.residual_stds.isna().any():
            raise ValueError("Residual statistics are missing sensor values")
        if (self.residual_stds <= 0).any():
            raise ValueError("Residual standard deviations must be positive")

        self.models = {}
        for sensor in self.sensor_columns:
            model_path_value = self.metadata["model_paths"][sensor]
            model_path = Path(model_path_value).expanduser()
            if not model_path.is_absolute():
                model_path = (self.metadata_path.parent / model_path).resolve()

            if not model_path.exists():
                raise FileNotFoundError(
                    f"Missing exported model for {sensor}: {model_path}"
                )

            model = joblib.load(model_path)
            model_feature_names = getattr(model, "feature_name_", None)
            if model_feature_names is not None:
                model_feature_names = list(model_feature_names)
                if model_feature_names != self.feature_columns:
                    raise ValueError(
                        f"Model feature order mismatch for sensor {sensor}"
                    )

            self.models[sensor] = model

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        required_columns = {"time", *self.sensor_columns}
        missing_columns = sorted(required_columns.difference(df.columns))
        if missing_columns:
            raise ValueError(
                "CSV input is missing required columns: "
                + ", ".join(missing_columns)
            )

        engineered = df.copy()
        engineered["time_dt"] = pd.to_datetime(
            engineered["time"],
            format="mixed",
            dayfirst=False,
            errors="coerce",
        )
        for column in self.sensor_columns:
            engineered[column] = pd.to_numeric(
                engineered[column],
                errors="coerce",
            )

        engineered = (
            engineered.sort_values("time_dt")
            .dropna(subset=[*self.sensor_columns, "time_dt"])
            .reset_index(drop=True)
        )
        if engineered.empty:
            return engineered

        engineered["time_diff"] = engineered["time_dt"].diff()
        engineered["segment_id"] = (
            engineered["time_diff"] != pd.Timedelta(minutes=5)
        ).cumsum()

        engineered["hour"] = engineered["time_dt"].dt.hour
        engineered["day_of_week_num"] = engineered["time_dt"].dt.dayofweek
        engineered["day_of_month"] = engineered["time_dt"].dt.day
        engineered["week_of_year"] = (
            engineered["time_dt"].dt.isocalendar().week.astype(int)
        )
        engineered["month"] = engineered["time_dt"].dt.month

        engineered["hour_sin"] = (
            np.sin(2 * np.pi * engineered["hour"] / 24.0)
        )
        engineered["hour_cos"] = (
            np.cos(2 * np.pi * engineered["hour"] / 24.0)
        )
        engineered["dow_sin"] = (
            np.sin(2 * np.pi * engineered["day_of_week_num"] / 7.0)
        )
        engineered["dow_cos"] = (
            np.cos(2 * np.pi * engineered["day_of_week_num"] / 7.0)
        )
        engineered["month_sin"] = (
            np.sin(2 * np.pi * engineered["month"] / 12.0)
        )
        engineered["month_cos"] = (
            np.cos(2 * np.pi * engineered["month"] / 12.0)
        )

        for lag in [1, 2, 3, 4]:
            for column in self.sensor_columns:
                engineered[f"{column}_lag{lag}"] = (
                    engineered.groupby("segment_id")[column].shift(lag)
                )

        for column in self.sensor_columns:
            engineered[f"{column}_diff1"] = (
                engineered.groupby("segment_id")[column]
                .shift(1)
                .diff(1)
            )
            engineered[f"{column}_roll3_mean"] = (
                engineered.groupby("segment_id")[column]
                .transform(
                    lambda values: values.shift(1)
                    .rolling(3, min_periods=1)
                    .mean()
                )
            )
            engineered[f"{column}_roll3_std"] = (
                engineered.groupby("segment_id")[column]
                .transform(
                    lambda values: values.shift(1)
                    .rolling(3, min_periods=1)
                    .std()
                )
                .fillna(0)
            )

        engineered = engineered.dropna(
            subset=self.feature_columns
        ).reset_index(drop=True)
        return engineered

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        engineered = self.engineer_features(df)
        if engineered.empty:
            return pd.DataFrame(
                columns=[
                    "time_dt",
                    *[
                        f"{sensor}_{suffix}"
                        for sensor in self.sensor_columns
                        for suffix in ("actual", "pred", "residual", "z")
                    ],
                    "anomaly_score",
                    "is_anomaly",
                ]
            )

        feature_matrix = engineered[self.feature_columns].copy()
        predictions = {}
        residuals = {}

        for sensor in self.sensor_columns:
            sensor_predictions = self.models[sensor].predict(feature_matrix)
            predictions[sensor] = sensor_predictions
            residuals[sensor] = (
                engineered[sensor].to_numpy(dtype=float) - sensor_predictions
            )

        residual_frame = pd.DataFrame(
            residuals,
            index=engineered.index,
        )
        standardized = (
            residual_frame[self.sensor_columns] - self.residual_means
        ) / self.residual_stds
        anomaly_scores = np.linalg.norm(
            standardized[self.sensor_columns].to_numpy(),
            axis=1,
        )

        result = pd.DataFrame(
            {
                "time_dt": engineered["time_dt"].to_numpy(),
                "anomaly_score": anomaly_scores,
                "is_anomaly": anomaly_scores > self.anomaly_threshold,
            }
        )

        for sensor in self.sensor_columns:
            result[f"{sensor}_actual"] = (
                engineered[sensor].to_numpy(dtype=float)
            )
            result[f"{sensor}_pred"] = predictions[sensor]
            result[f"{sensor}_residual"] = residuals[sensor]
            result[f"{sensor}_z"] = (
                standardized[sensor].to_numpy(dtype=float)
            )

        return result

    def predict_csv(
        self,
        csv_path: Union[str, Path],
    ) -> pd.DataFrame:
        return self.predict(pd.read_csv(csv_path))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run CycloneGuard LightGBM inference on a CSV file."
    )
    parser.add_argument("csv_path", help="CSV containing time and six sensor columns")
    parser.add_argument(
        "--metadata",
        default=None,
        help="Optional path to lightgbm_deployment_metadata.json",
    )
    parser.add_argument(
        "--artifacts-dir",
        default=None,
        help="Optional directory containing exported joblib models",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional CSV path for inference results",
    )
    args = parser.parse_args()

    engine = InferenceEngine(
        metadata_path=args.metadata,
        artifacts_dir=args.artifacts_dir,
    )
    result = engine.predict_csv(args.csv_path)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_path, index=False)
    else:
        result.to_csv(sys.stdout, index=False)


if __name__ == "__main__":
    main()
