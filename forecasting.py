"""Data validation, demonstration data, and forecasting utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


REQUIRED_COLUMNS = {"timestamp", "area", "demand_mw"}
OPTIONAL_NUMERIC_COLUMNS = {
    "outage_hours",
    "temperature_c",
    "wind_speed_kmh",
    "precipitation_mm",
    "customers_affected",
}
WEATHER_COLUMNS = ("temperature_c", "wind_speed_kmh", "precipitation_mm")


@dataclass(frozen=True)
class DemandForecastResult:
    forecasts: pd.DataFrame
    metrics: pd.DataFrame


@dataclass(frozen=True)
class OutageForecastResult:
    forecasts: pd.DataFrame
    metrics: dict[str, float]


def validate_data(data: pd.DataFrame) -> pd.DataFrame:
    """Normalize and validate a user-supplied area/time-series table."""
    if data.empty:
        raise ValueError("The uploaded file has no data rows.")

    frame = data.copy()
    frame.columns = [str(column).strip().lower() for column in frame.columns]
    if "outage" in frame.columns and "outage_hours" not in frame.columns:
        frame["outage_hours"] = pd.to_numeric(frame["outage"], errors="coerce").astype(
            "float64"
        )
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        expected = ", ".join(sorted(REQUIRED_COLUMNS))
        raise ValueError(
            f"Missing required column(s): {', '.join(sorted(missing))}. "
            f"Required columns are: {expected}."
        )

    frame["timestamp"] = pd.to_datetime(
        frame["timestamp"], errors="coerce", utc=True
    ).dt.tz_convert(None)
    if frame["timestamp"].isna().any():
        raise ValueError("Some timestamp values could not be parsed.")

    frame["area"] = frame["area"].astype("string").str.strip()
    if frame["area"].isna().any() or frame["area"].eq("").any():
        raise ValueError("Area values must not be blank.")

    numeric_columns = {"demand_mw"} | (
        OPTIONAL_NUMERIC_COLUMNS & set(frame.columns)
    )
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if frame[column].isna().any():
            raise ValueError(f"Column '{column}' must contain valid numeric values.")
        if not np.isfinite(frame[column].to_numpy(dtype=float)).all():
            raise ValueError(f"Column '{column}' contains a non-finite value.")
        if (frame[column] < 0).any():
            raise ValueError(f"Column '{column}' cannot contain negative values.")

    if frame.duplicated(["area", "timestamp"]).any():
        raise ValueError("Each area must have at most one row per timestamp.")

    frame = frame.sort_values(["area", "timestamp"]).reset_index(drop=True)
    for area, group in frame.groupby("area", sort=False):
        differences = group["timestamp"].diff().dropna()
        if not differences.empty:
            cadence = differences.mode().iloc[0]
            if not differences.eq(cadence).all():
                raise ValueError(
                    f"Area '{area}' has missing or irregular timestamps. "
                    "Provide a regular time series so lag features match their "
                    "intended time intervals."
                )
    return frame


def generate_demo_data(days: int = 120, seed: int = 42) -> pd.DataFrame:
    """Create labeled synthetic hourly data for demonstrating the application."""
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range(
        end=pd.Timestamp.now(tz="UTC").floor("h").tz_localize(None),
        periods=days * 24,
        freq="h",
    )
    areas = {
        "North": (1.0, 15.0),
        "South": (1.12, 27.0),
        "East": (0.92, 22.0),
        "West": (1.06, 19.0),
        "Central": (1.18, 24.0),
    }
    rows: list[dict[str, object]] = []

    for area_index, (area, (demand_scale, typical_temp)) in enumerate(areas.items()):
        day_of_year = timestamps.dayofyear.to_numpy()
        hour = timestamps.hour.to_numpy()
        seasonal_temperature = 8 * np.sin(2 * np.pi * (day_of_year - 100) / 365)
        temperature = (
            typical_temp + seasonal_temperature + rng.normal(0, 3, len(timestamps))
        )
        wind = np.maximum(0, rng.gamma(2.0, 7.0, len(timestamps)))
        storm_days = rng.random(days) < 0.055
        storm = np.repeat(storm_days, 24)
        wind = wind + storm * rng.uniform(20, 45, len(timestamps))
        precipitation = np.where(
            storm,
            rng.gamma(2.0, 2.5, len(timestamps)),
            rng.gamma(0.5, 0.25, len(timestamps)),
        )

        daily_pattern = 70 * np.sin(2 * np.pi * (hour - 7) / 24)
        evening_peak = 95 * np.exp(-((hour - 19) / 3.2) ** 2)
        weekday_effect = np.where(timestamps.dayofweek.to_numpy() < 5, 55, -25)
        demand = (
            (420 + daily_pattern + evening_peak + weekday_effect)
            * demand_scale
            + 2.2 * np.maximum(temperature - 20, 0)
            + 1.5 * np.maximum(12 - temperature, 0)
            + rng.normal(0, 13, len(timestamps))
        )
        demand = np.maximum(demand, 100)

        outage_probability = np.where(
            storm,
            0.075 + np.minimum(wind / 1100, 0.075),
            0.0015,
        )
        outage_hours = np.where(
            rng.random(len(timestamps)) < outage_probability,
            rng.integers(1, 5, len(timestamps)),
            0,
        )
        customers_affected = np.where(
            outage_hours > 0,
            rng.integers(15, 1800, len(timestamps)),
            0,
        )

        rows.extend(
            {
                "timestamp": timestamp,
                "area": area,
                "demand_mw": round(float(load), 2),
                "outage_hours": int(outage),
                "temperature_c": round(float(temp), 2),
                "wind_speed_kmh": round(float(wind_speed), 2),
                "precipitation_mm": round(float(rain), 2),
                "customers_affected": int(customers),
            }
            for timestamp, load, outage, temp, wind_speed, rain, customers in zip(
                timestamps,
                demand,
                outage_hours,
                temperature,
                wind,
                precipitation,
                customers_affected,
                strict=True,
            )
        )

    return validate_data(pd.DataFrame(rows))


def _periods_per_day(timestamps: pd.Series) -> int:
    differences = timestamps.sort_values().diff().dropna()
    if differences.empty:
        raise ValueError("Each area needs multiple historical timestamps.")
    median_seconds = differences.median().total_seconds()
    if median_seconds <= 0:
        raise ValueError("Timestamps must be strictly increasing within each area.")
    return max(1, int(round(24 * 60 * 60 / median_seconds)))


def _demand_features(group: pd.DataFrame, periods_per_day: int) -> pd.DataFrame:
    frame = group.copy()
    frame["hour"] = frame["timestamp"].dt.hour
    frame["day_of_week"] = frame["timestamp"].dt.dayofweek
    frame["month"] = frame["timestamp"].dt.month
    frame["lag_1"] = frame["demand_mw"].shift(1)
    frame["lag_day"] = frame["demand_mw"].shift(periods_per_day)
    frame["lag_week"] = frame["demand_mw"].shift(periods_per_day * 7)
    frame["rolling_day_mean"] = (
        frame["demand_mw"].shift(1).rolling(periods_per_day, min_periods=1).mean()
    )
    return frame.dropna(
        subset=["lag_1", "lag_day", "lag_week", "rolling_day_mean"]
    )


def forecast_demand(
    data: pd.DataFrame, horizon: int = 24, areas: list[str] | None = None
) -> DemandForecastResult:
    """Fit per-area random forests and recursively forecast demand."""
    if horizon < 1:
        raise ValueError("Forecast horizon must be at least one time step.")

    selected_areas = areas or sorted(data["area"].astype(str).unique().tolist())
    forecast_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    feature_columns = [
        "hour",
        "day_of_week",
        "month",
        "lag_1",
        "lag_day",
        "lag_week",
        "rolling_day_mean",
    ]

    for area in selected_areas:
        group = data.loc[data["area"].astype(str) == area].sort_values("timestamp")
        if group.empty:
            continue
        periods_per_day = _periods_per_day(group["timestamp"])
        features = _demand_features(group, periods_per_day)
        minimum_rows = max(30, periods_per_day * 7 + 1)
        if len(features) < minimum_rows:
            raise ValueError(
                f"Area '{area}' needs at least {minimum_rows} usable records after "
                "the weekly lag is created. Upload more historical data."
            )

        split = max(int(len(features) * 0.8), 1)
        if len(features) - split < 1:
            raise ValueError(f"Area '{area}' has too little data for a time holdout.")
        model = RandomForestRegressor(
            n_estimators=120,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(
            features.iloc[:split][feature_columns],
            features.iloc[:split]["demand_mw"],
        )
        validation = features.iloc[split:]
        validation_predictions = model.predict(validation[feature_columns])
        naive_predictions = validation["lag_day"].to_numpy()
        mae = float(mean_absolute_error(validation["demand_mw"], validation_predictions))
        rmse = float(
            np.sqrt(mean_squared_error(validation["demand_mw"], validation_predictions))
        )
        naive_mae = float(
            mean_absolute_error(validation["demand_mw"], naive_predictions)
        )
        residuals = validation["demand_mw"].to_numpy() - validation_predictions
        lower_error, upper_error = np.quantile(residuals, [0.05, 0.95])

        model.fit(features[feature_columns], features["demand_mw"])
        history = group["demand_mw"].astype(float).tolist()
        last_timestamp = group["timestamp"].iloc[-1]
        cadence = group["timestamp"].diff().dropna().median()
        for step in range(1, horizon + 1):
            timestamp = last_timestamp + step * cadence
            row = {
                "hour": timestamp.hour,
                "day_of_week": timestamp.dayofweek,
                "month": timestamp.month,
                "lag_1": history[-1],
                "lag_day": history[-periods_per_day],
                "lag_week": history[-periods_per_day * 7],
                "rolling_day_mean": float(
                    np.mean(history[-periods_per_day:])
                ),
            }
            prediction = max(0.0, float(model.predict(pd.DataFrame([row]))[0]))
            forecast_rows.append(
                {
                    "timestamp": timestamp,
                    "area": area,
                    "demand_mw": prediction,
                    "lower_mw": max(0.0, prediction + float(lower_error)),
                    "upper_mw": max(0.0, prediction + float(upper_error)),
                }
            )
            history.append(prediction)

        metric_rows.append(
            {
                "area": area,
                "validation_mae_mw": mae,
                "validation_rmse_mw": rmse,
                "seasonal_naive_mae_mw": naive_mae,
                "holdout_steps": len(validation),
            }
        )

    if not forecast_rows:
        raise ValueError("No forecasts could be generated for the selected areas.")
    return DemandForecastResult(
        forecasts=pd.DataFrame(forecast_rows),
        metrics=pd.DataFrame(metric_rows),
    )


def forecast_outage_risk(
    data: pd.DataFrame, horizon: int = 24, areas: list[str] | None = None
) -> OutageForecastResult | None:
    """Forecast calendar-conditioned outage probability when labels suffice."""
    if "outage_hours" not in data.columns:
        return None

    selected_areas = areas or sorted(data["area"].astype(str).unique().tolist())
    frame = data.loc[data["area"].astype(str).isin(selected_areas)].copy()
    frame["outage"] = (frame["outage_hours"] > 0).astype(int)
    frame["hour"] = frame["timestamp"].dt.hour
    frame["day_of_week"] = frame["timestamp"].dt.dayofweek
    frame["month"] = frame["timestamp"].dt.month
    split_time = frame["timestamp"].quantile(0.8)
    training = frame.loc[frame["timestamp"] < split_time]
    validation = frame.loc[frame["timestamp"] >= split_time]
    if (
        training["outage"].nunique() < 2
        or int(training["outage"].sum()) < 10
        or int((training["outage"] == 0).sum()) < 20
    ):
        return None

    categorical = ["area"]
    numeric = ["hour", "day_of_week", "month"]
    preprocessor = ColumnTransformer(
        [
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("numeric", StandardScaler(), numeric),
        ]
    )
    model = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )
    model.fit(training[categorical + numeric], training["outage"])

    metrics: dict[str, float] = {}
    if validation["outage"].nunique() == 2:
        probabilities = model.predict_proba(validation[categorical + numeric])[:, 1]
        metrics = {
            "average_precision": float(
                average_precision_score(validation["outage"], probabilities)
            ),
            "brier_score": float(
                brier_score_loss(validation["outage"], probabilities)
            ),
            "baseline_outage_rate": float(training["outage"].mean()),
        }

    forecast_rows: list[dict[str, object]] = []
    for area in selected_areas:
        area_data = frame.loc[frame["area"].astype(str) == area].sort_values(
            "timestamp"
        )
        if area_data.empty:
            continue
        cadence = area_data["timestamp"].diff().dropna().median()
        if pd.isna(cadence):
            continue
        last_timestamp = area_data["timestamp"].iloc[-1]
        for step in range(1, horizon + 1):
            timestamp = last_timestamp + step * cadence
            future_row = pd.DataFrame(
                [
                    {
                        "area": area,
                        "hour": timestamp.hour,
                        "day_of_week": timestamp.dayofweek,
                        "month": timestamp.month,
                    }
                ]
            )
            probability = float(
                model.predict_proba(future_row[categorical + numeric])[0, 1]
            )
            forecast_rows.append(
                {
                    "timestamp": timestamp,
                    "area": area,
                    "outage_probability": probability,
                }
            )
    if not forecast_rows:
        return None
    return OutageForecastResult(pd.DataFrame(forecast_rows), metrics)
