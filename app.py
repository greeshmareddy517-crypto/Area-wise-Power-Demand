from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256

import pandas as pd
import streamlit as st

from forecasting import (
    forecast_demand,
    forecast_outage_risk,
    generate_demo_data,
    validate_data,
)


st.set_page_config(
    page_title="Area-Wise Power Forecasting",
    page_icon="⚡",
    layout="wide",
)


@st.cache_data
def get_demo_data() -> pd.DataFrame:
    return generate_demo_data()


@st.cache_data
def read_uploaded_csv(contents: bytes) -> pd.DataFrame:
    return validate_data(pd.read_csv(contents))


def make_report(
    data: pd.DataFrame,
    demand_metrics: pd.DataFrame,
    demand_forecasts: pd.DataFrame,
    outage_forecasts: pd.DataFrame | None,
) -> bytes:
    """Create a portable CSV forecast report with validation context."""
    report = demand_forecasts.copy()
    if outage_forecasts is not None:
        report = report.merge(
            outage_forecasts,
            on=["timestamp", "area"],
            how="left",
        )
    report["demand_validation_mae_mw"] = report["area"].map(
        demand_metrics.set_index("area")["validation_mae_mw"]
    )
    report["data_through_utc"] = data["timestamp"].max().isoformat()
    report["report_generated_utc"] = datetime.now(timezone.utc).isoformat()
    return report.to_csv(index=False).encode("utf-8")


st.title("⚡ Area-Wise Power Demand & Outage Forecasting")
st.caption(
    "Explore historical area-level data, generate short-term forecasts, "
    "and export a transparent planning report."
)

with st.sidebar:
    st.header("Data")
    data_source = st.radio("Choose data source", ["Synthetic demo", "Upload CSV"])
    uploaded_file = None
    if data_source == "Upload CSV":
        uploaded_file = st.file_uploader("Upload historical CSV", type=["csv"])
        st.caption(
            "Required: timestamp, area, demand_mw. Optional: outage_hours, "
            "temperature_c, wind_speed_kmh, precipitation_mm, customers_affected."
        )
        st.download_button(
            "Download example CSV",
            data=get_demo_data().head(240).to_csv(index=False).encode("utf-8"),
            file_name="power_history_example.csv",
            mime="text/csv",
        )
    else:
        st.info("Synthetic demonstration records are not real grid measurements.")

if data_source == "Upload CSV" and uploaded_file is None:
    st.info("Upload a CSV file to explore your historical data.")
    st.stop()

try:
    if data_source == "Synthetic demo":
        history = get_demo_data()
        data_key = "synthetic-demo"
    else:
        uploaded_contents = uploaded_file.getvalue()
        history = read_uploaded_csv(uploaded_contents)
        data_key = sha256(uploaded_contents).hexdigest()
except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as error:
    st.error(f"Could not load the selected data: {error}")
    st.stop()

if st.session_state.get("data_key") != data_key:
    for key in (
        "demand_forecasts",
        "demand_metrics",
        "outage_forecasts",
        "outage_metrics",
        "forecast_areas",
        "forecast_horizon",
    ):
        st.session_state.pop(key, None)
    st.session_state["data_key"] = data_key

areas = sorted(history["area"].astype(str).unique().tolist())
data_is_demo = data_source == "Synthetic demo"
if data_is_demo:
    st.warning(
        "DEMO MODE: all displayed observations and forecasts are synthetic. "
        "Do not use them for real operational decisions."
    )

if len(areas) == 0:
    st.error("No valid areas were found in the selected data.")
    st.stop()

period_start = history["timestamp"].min()
period_end = history["timestamp"].max()
area_count = history["area"].nunique()
total_outages = (
    int((history["outage_hours"] > 0).sum())
    if "outage_hours" in history.columns
    else None
)

summary_columns = st.columns(4)
summary_columns[0].metric("Areas", f"{area_count:,}")
summary_columns[1].metric("Historical records", f"{len(history):,}")
summary_columns[2].metric("History starts (UTC)", period_start.strftime("%Y-%m-%d"))
summary_columns[3].metric(
    "Outage records",
    f"{total_outages:,}" if total_outages is not None else "Not provided",
)
st.caption(
    f"Data covers {period_start:%Y-%m-%d %H:%M} to "
    f"{period_end:%Y-%m-%d %H:%M} UTC. Demand is shown in MW."
)

left, right = st.columns([2, 1])
with left:
    st.subheader("Historical demand by area")
    selected_history_areas = st.multiselect(
        "Areas to display", areas, default=areas[: min(4, len(areas))]
    )
    historical = history.loc[history["area"].astype(str).isin(selected_history_areas)]
    if historical.empty:
        st.info("Select at least one area to display its history.")
    else:
        daily_demand = (
            historical.groupby(
                ["area", pd.Grouper(key="timestamp", freq="D")], as_index=False
            )["demand_mw"]
            .mean()
        )
        st.line_chart(
            daily_demand,
            x="timestamp",
            y="demand_mw",
            color="area",
            y_label="Daily average demand (MW)",
        )

with right:
    st.subheader("Historical area summary")
    summary = (
        history.groupby("area", as_index=False)
        .agg(
            average_demand_mw=("demand_mw", "mean"),
            peak_demand_mw=("demand_mw", "max"),
            demand_records=("demand_mw", "size"),
        )
        .sort_values("average_demand_mw", ascending=False)
    )
    if "outage_hours" in history.columns:
        outage_summary = history.groupby("area", as_index=False).agg(
            outage_hours=("outage_hours", "sum"),
            outage_records=("outage_hours", lambda values: int((values > 0).sum())),
        )
        summary = summary.merge(outage_summary, on="area", how="left")
    st.dataframe(
        summary.round(2),
        hide_index=True,
        width="stretch",
    )

st.subheader("Generate a forecast")
st.caption(
    "Demand models use calendar features and historical lags. Forecast intervals "
    "are empirical residual ranges from a chronological holdout, not guarantees."
)
forecast_area_options = ["All areas"] + areas
with st.form("forecast_settings"):
    selected_forecast_area = st.selectbox("Forecast area", forecast_area_options)
    horizon = st.slider(
        "Forecast horizon (time steps)",
        min_value=6,
        max_value=48,
        value=24,
        help="The interval matches the regular sampling cadence in your data.",
    )
    generate = st.form_submit_button("Generate forecasts", type="primary")

if generate:
    chosen_areas = (
        areas if selected_forecast_area == "All areas" else [selected_forecast_area]
    )
    try:
        with st.spinner("Fitting models and preparing forecasts..."):
            demand_result = forecast_demand(history, horizon, chosen_areas)
            outage_result = forecast_outage_risk(history, horizon, chosen_areas)
        st.session_state["demand_forecasts"] = demand_result.forecasts
        st.session_state["demand_metrics"] = demand_result.metrics
        st.session_state["outage_forecasts"] = (
            outage_result.forecasts if outage_result is not None else None
        )
        st.session_state["outage_metrics"] = (
            outage_result.metrics if outage_result is not None else None
        )
        st.session_state["forecast_areas"] = chosen_areas
        st.session_state["forecast_horizon"] = horizon
    except ValueError as error:
        st.error(f"Forecast could not be generated: {error}")

demand_forecasts = st.session_state.get("demand_forecasts")
demand_metrics = st.session_state.get("demand_metrics")
outage_forecasts = st.session_state.get("outage_forecasts")
outage_metrics = st.session_state.get("outage_metrics")

if demand_forecasts is not None and demand_metrics is not None:
    st.subheader("Demand forecast")
    forecast_areas = st.session_state["forecast_areas"]
    actual = history.loc[history["area"].astype(str).isin(forecast_areas)]
    recent_actual = actual.groupby("area", group_keys=False).tail(
        min(st.session_state["forecast_horizon"] * 2, len(actual))
    )
    chart_frames = [
        recent_actual[["timestamp", "area", "demand_mw"]].assign(
            series=recent_actual["area"].astype(str) + " · actual"
        )
    ]
    for column, label in (
        ("demand_mw", "forecast"),
        ("lower_mw", "lower range"),
        ("upper_mw", "upper range"),
    ):
        chart_frames.append(
            demand_forecasts[["timestamp", "area", column]]
            .rename(columns={column: "demand_mw"})
            .assign(
                series=demand_forecasts["area"].astype(str) + f" · {label}"
            )
        )
    chart = pd.concat(chart_frames, ignore_index=True)
    st.line_chart(
        chart,
        x="timestamp",
        y="demand_mw",
        color="series",
        y_label="Demand (MW)",
    )
    st.caption(
        "For future steps, lagged demand values are recursively replaced by "
        "predictions. The intervals summarize historical holdout residuals."
    )

    st.markdown("**Chronological holdout performance**")
    st.dataframe(
        demand_metrics.round(3),
        hide_index=True,
        width="stretch",
    )
    if (demand_metrics["validation_mae_mw"] > demand_metrics["seasonal_naive_mae_mw"]).any():
        st.info(
            "The model does not beat the seasonal-naive baseline in every area. "
            "Use the holdout metrics when interpreting these forecasts."
        )

    st.subheader("Outage risk")
    if outage_forecasts is None:
        if "outage_hours" not in history.columns:
            st.info(
                "Outage forecasting is unavailable because this CSV has no "
                "'outage_hours' labels."
            )
        else:
            st.warning(
                "There are too few positive outage records in the training period "
                "to fit a meaningful risk model. No outage prediction is shown."
            )
    else:
        st.caption(
            "This is a calendar-conditioned historical risk estimate. The model "
            "does not use future weather forecasts or utility network condition."
        )
        threshold = st.slider(
            "Demo alert threshold",
            min_value=0.01,
            max_value=0.50,
            value=0.10,
            step=0.01,
            format="%.2f",
        )
        risk_summary = (
            outage_forecasts.groupby("area", as_index=False)
            .agg(
                maximum_outage_probability=("outage_probability", "max"),
                mean_outage_probability=("outage_probability", "mean"),
            )
            .sort_values("maximum_outage_probability", ascending=False)
        )
        st.bar_chart(
            risk_summary.set_index("area")[
                ["maximum_outage_probability", "mean_outage_probability"]
            ],
            y_label="Estimated probability",
        )
        alerts = risk_summary.loc[
            risk_summary["maximum_outage_probability"] >= threshold
        ]
        if alerts.empty:
            st.success(
                "No area crosses the selected demo threshold for this forecast."
            )
        else:
            alert_areas = ", ".join(alerts["area"].astype(str))
            st.warning(
                f"Demo risk signal for: {alert_areas}. This is not an operational "
                "outage warning."
            )
        if outage_metrics:
            st.markdown("**Chronological outage holdout metrics**")
            st.dataframe(
                pd.DataFrame([outage_metrics]).round(4),
                hide_index=True,
                width="stretch",
            )
            st.caption(
                "Average precision is compared with event prevalence; lower Brier "
                "scores indicate better probability accuracy."
            )

    st.subheader("Forecast report")
    st.download_button(
        "Download forecast report (CSV)",
        data=make_report(
            history, demand_metrics, demand_forecasts, outage_forecasts
        ),
        file_name="power_forecast_report.csv",
        mime="text/csv",
    )
    st.dataframe(
        demand_forecasts.head(100).round(3),
        hide_index=True,
        width="stretch",
    )

with st.expander("Data preview and expected CSV format"):
    st.dataframe(history.head(100), hide_index=True, width="stretch")
    st.markdown(
        "Use one row per area and timestamp. Required fields are "
        "`timestamp`, `area`, and `demand_mw`. Supported optional fields are "
        "`outage_hours`, `temperature_c`, `wind_speed_kmh`, "
        "`precipitation_mm`, and `customers_affected`. Demand and outage values "
        "must be non-negative. Timestamps are interpreted as UTC if they do not "
        "include a timezone. For meaningful lag features, provide regular, "
        "chronologically ordered observations with at least one week of history."
    )

st.divider()
st.caption(
    "Decision-support prototype only. Forecast quality depends on data coverage, "
    "sampling cadence, and independent chronological validation."
)
