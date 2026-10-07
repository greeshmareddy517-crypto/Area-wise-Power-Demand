# Area-Wise Power Demand and Outage Forecasting

A Streamlit decision-support prototype for exploring historical area-level
electricity demand and forecasting demand and historical outage risk.

> **Important:** The built-in dataset is synthetic and intended only to
> demonstrate the application. Forecasts and alerts are not suitable for
> operational grid decisions.

## Current capabilities

- Explore historical demand and compare areas.
- Use the built-in synthetic hourly dataset or upload a CSV.
- Fit a per-area random forest demand model using calendar and historical-lag
  features.
- Compare demand forecasts with a seasonal-naive baseline on a chronological
  holdout; show empirical residual ranges.
- Estimate calendar-conditioned outage probability when there are enough
  historical outage labels.
- Review average-precision and Brier-score outage metrics on a chronological
  holdout.
- Download forecast results and validation context as a CSV report.
- Display threshold-based demo risk signals without representing them as live
  utility alerts.

## Run locally

The project uses Python 3.11 and the dependencies listed in
[`requirements.txt`](requirements.txt).

```powershell
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

If you do not use the included virtual environment, create and activate one
first, then install the requirements.

## CSV format

Supply one observation per area and timestamp. Required columns:

| Column | Meaning |
|---|---|
| `timestamp` | Observation time; timezone-aware values are converted to UTC. Naive values are interpreted as UTC. |
| `area` | Consistent geographic or utility-area identifier. |
| `demand_mw` | Non-negative demand measurement in megawatts. |

Optional columns:

| Column | Meaning |
|---|---|
| `outage_hours` | Non-negative outage duration for that observation. A model label is derived as `outage_hours > 0`. |
| `temperature_c` | Temperature in degrees Celsius, for historical exploration. |
| `wind_speed_kmh` | Wind speed in kilometers per hour, for historical exploration. |
| `precipitation_mm` | Precipitation in millimeters, for historical exploration. |
| `customers_affected` | Non-negative number of affected customers. |

The app rejects missing required columns, invalid numeric values, negative
measurements, invalid timestamps, and duplicate area/timestamp rows. For useful
demand lag features, supply regular, chronologically ordered observations and
at least one week of data per area. The model currently assumes a regular
sampling cadence within each area.

## Modeling and interpretation

Demand models use lagged demand, a daily/weekly seasonal lag, and calendar
features. A chronological holdout is used to report one-step MAE and RMSE next
to a seasonal-naive MAE. Multi-step predictions recursively use prior model
predictions. The displayed demand ranges are empirical 5th-to-95th-percentile
holdout residual offsets; they are not guaranteed prediction intervals.

The outage model derives a binary event label from `outage_hours`, then uses
area and calendar features in logistic regression. It deliberately does not
use weather observations as if future weather were known. Its results express
historical calendar-conditioned risk only, not weather-driven causation or a
utility-grade outage warning. If the available training period has fewer than
10 outage records or only one target class, the app declines to show a model
forecast rather than returning a fabricated result.

Evaluation is chronological, but this application is still a prototype. A
credible research evaluation should additionally use longer history, preserve
storm/event groups during validation, assess probability calibration and
false-alarm costs, and compare against appropriate area-specific baselines.

## Tests

Run the focused unit tests with:

```powershell
python -m unittest discover -s tests -v
```

## Project structure

```text
app.py                  Streamlit interface
forecasting.py          Validation, synthetic data, and modeling functions
tests/test_forecasting.py
requirements.txt        Existing application dependencies
```
