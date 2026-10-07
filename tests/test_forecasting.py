import unittest

import pandas as pd

from forecasting import (
    forecast_demand,
    forecast_outage_risk,
    generate_demo_data,
    validate_data,
)


class DataValidationTests(unittest.TestCase):
    def test_demo_data_is_hourly_and_has_multiple_areas(self) -> None:
        data = generate_demo_data(days=10)

        self.assertEqual(data["area"].nunique(), 5)
        self.assertEqual(len(data), 10 * 24 * 5)
        self.assertEqual(data.groupby("area").size().nunique(), 1)
        self.assertTrue(data["demand_mw"].min() >= 0)

    def test_validate_data_normalizes_column_names_and_timestamps(self) -> None:
        data = pd.DataFrame(
            {
                " Timestamp ": ["2025-01-01T00:00:00Z"],
                " AREA ": ["North"],
                " Demand_MW ": ["12.5"],
            }
        )

        result = validate_data(data)

        self.assertEqual(result.loc[0, "area"], "North")
        self.assertEqual(result.loc[0, "demand_mw"], 12.5)
        self.assertEqual(str(result.loc[0, "timestamp"]), "2025-01-01 00:00:00")

    def test_validate_data_rejects_duplicate_area_timestamps(self) -> None:
        duplicate = pd.DataFrame(
            {
                "timestamp": ["2025-01-01", "2025-01-01"],
                "area": ["North", "North"],
                "demand_mw": [10, 11],
            }
        )

        with self.assertRaisesRegex(ValueError, "at most one row"):
            validate_data(duplicate)

    def test_validate_data_rejects_irregular_timestamps(self) -> None:
        irregular = pd.DataFrame(
            {
                "timestamp": ["2025-01-01T00:00:00Z", "2025-01-01T02:00:00Z", "2025-01-01T03:00:00Z"],
                "area": ["North", "North", "North"],
                "demand_mw": [10, 11, 12],
            }
        )

        with self.assertRaisesRegex(ValueError, "missing or irregular"):
            validate_data(irregular)

    def test_validate_data_rejects_negative_measurements(self) -> None:
        data = pd.DataFrame(
            {
                "timestamp": ["2025-01-01"],
                "area": ["North"],
                "demand_mw": [-1],
            }
        )

        with self.assertRaisesRegex(ValueError, "cannot contain negative"):
            validate_data(data)


class ForecastTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data = generate_demo_data(days=45)

    def test_demand_forecast_has_expected_horizon_and_nonnegative_bounds(self) -> None:
        result = forecast_demand(self.data, horizon=6, areas=["North"])

        self.assertEqual(len(result.forecasts), 6)
        self.assertEqual(result.forecasts["area"].unique().tolist(), ["North"])
        self.assertTrue((result.forecasts["demand_mw"] >= 0).all())
        self.assertTrue((result.forecasts["lower_mw"] <= result.forecasts["upper_mw"]).all())
        self.assertIn("seasonal_naive_mae_mw", result.metrics.columns)

    def test_insufficient_demand_history_is_reported(self) -> None:
        short_history = self.data.loc[
            self.data["timestamp"] < self.data["timestamp"].min() + pd.Timedelta(days=5)
        ]

        with self.assertRaisesRegex(ValueError, "needs at least"):
            forecast_demand(short_history, horizon=6, areas=["North"])

    def test_outage_forecast_returns_no_model_without_labels(self) -> None:
        no_labels = self.data.drop(columns="outage_hours")

        self.assertIsNone(forecast_outage_risk(no_labels, horizon=6, areas=["North"]))

    def test_outage_forecast_requires_sufficient_positive_history(self) -> None:
        no_events = self.data.copy()
        no_events["outage_hours"] = 0

        self.assertIsNone(
            forecast_outage_risk(no_events, horizon=6, areas=["North"])
        )

    def test_outage_forecast_produces_probabilities_with_enough_labels(self) -> None:
        data = generate_demo_data(days=120)

        result = forecast_outage_risk(data, horizon=6, areas=["North"])

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(len(result.forecasts), 6)
        self.assertTrue(result.forecasts["outage_probability"].between(0, 1).all())
        self.assertIn("average_precision", result.metrics)


if __name__ == "__main__":
    unittest.main()
