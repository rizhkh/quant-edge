import os
import sys
import io
import pandas as pd
import numpy as np
from contextlib import redirect_stdout, redirect_stderr
from forecaster.analog import run_analog_forecast
from forecaster.knn import run_knn_forecast


def run_backtest(df: pd.DataFrame, config: dict) -> dict:
    """
    Walk-forward backtest: run forecasts on historical data and compare to actual prices.

    Returns:
      {
        "total_forecasts": int,
        "date_range": (start_date, end_date),
        "mae": float,
        "rmse": float,
        "mape": float,
        "directional_accuracy": float,
        "cone_hit_rate": float,
        "daily_accuracy": {day: mae},
        "summary_metrics": pd.DataFrame
      }
    """
    months_back   = config.get("BACKTEST_MONTHS", 6)
    step_days     = config.get("BACKTEST_STEP", 20)
    forecast_days = config.get("BACKTEST_FORECAST_DAYS", 30)
    start_date    = config.get("BACKTEST_START")
    end_date      = config.get("BACKTEST_END")

    if start_date and end_date:
        test_start_idx = max(0, int(df.index.searchsorted(pd.Timestamp(start_date))))
        test_end_idx   = min(len(df), int(df.index.searchsorted(pd.Timestamp(end_date), side="right")))
    else:
        test_start_idx = max(0, len(df) - months_back * 21)
        test_end_idx   = len(df)

    results = []

    # Walk forward through test period
    for i in range(test_start_idx, test_end_idx - forecast_days, step_days):
        test_date = df.index[i]

        # Split: train on data before test_date, forecast from test_date
        train_df = df.iloc[:i].copy()
        actual_future = df.iloc[i:i+forecast_days].copy()

        if len(train_df) < 100:  # Need enough training data
            continue

        try:
            # Run both forecasts on training data (suppress output)
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                analog_result = run_analog_forecast(train_df, config)
                knn_result = run_knn_forecast(train_df, config)

            # Get forecast prices
            analog_forecast = analog_result["forecast_df"]["price"].values[:forecast_days]
            knn_forecast = knn_result["forecast_cone"]["median"].values[:forecast_days]
            knn_low = knn_result["forecast_cone"]["low"].values[:forecast_days]
            knn_high = knn_result["forecast_cone"]["high"].values[:forecast_days]

            # Get actual prices
            actual_prices = actual_future["Close"].values[:forecast_days]

            # Compute errors
            analog_mae = np.mean(np.abs(analog_forecast - actual_prices))
            knn_mae = np.mean(np.abs(knn_forecast - actual_prices))

            analog_rmse = np.sqrt(np.mean((analog_forecast - actual_prices) ** 2))
            knn_rmse = np.sqrt(np.mean((knn_forecast - actual_prices) ** 2))

            analog_mape = np.mean(np.abs((actual_prices - analog_forecast) / actual_prices)) * 100
            knn_mape = np.mean(np.abs((actual_prices - knn_forecast) / actual_prices)) * 100

            # Directional accuracy (did price go up/down as forecast predicted?)
            analog_direction = 1 if analog_forecast[-1] > train_df["Close"].iloc[-1] else 0
            knn_direction = 1 if knn_forecast[-1] > train_df["Close"].iloc[-1] else 0
            actual_direction = 1 if actual_prices[-1] > train_df["Close"].iloc[-1] else 0

            analog_dir_correct = 1 if analog_direction == actual_direction else 0
            knn_dir_correct = 1 if knn_direction == actual_direction else 0

            # Cone hit rate (% of prices within bands)
            analog_hits = np.sum((actual_prices >= knn_low) & (actual_prices <= knn_high))
            cone_hit_rate = analog_hits / len(actual_prices)

            # Day-by-day errors
            day_errors = {}
            for day in [1, 5, 10, 20, 30]:
                if day <= len(actual_prices):
                    idx = day - 1
                    day_errors[f"day_{day}_analog"] = abs(analog_forecast[idx] - actual_prices[idx])
                    day_errors[f"day_{day}_knn"] = abs(knn_forecast[idx] - actual_prices[idx])

            results.append({
                "test_date": test_date,
                "forecast_start": actual_future.index[0],
                "forecast_end": actual_future.index[-1],
                "analog_mae": analog_mae,
                "knn_mae": knn_mae,
                "analog_rmse": analog_rmse,
                "knn_rmse": knn_rmse,
                "analog_mape": analog_mape,
                "knn_mape": knn_mape,
                "analog_directional": analog_dir_correct,
                "knn_directional": knn_dir_correct,
                "cone_hit_rate": cone_hit_rate,
                **day_errors
            })
        except Exception as e:
            print(f"  Warning: Backtest failed at {test_date}: {e}")
            continue

    if not results:
        print("No valid backtest results generated.")
        return {}

    results_df = pd.DataFrame(results)

    # Aggregate metrics
    overall_mae = results_df["knn_mae"].mean()
    overall_rmse = results_df["knn_rmse"].mean()
    overall_mape = results_df["knn_mape"].mean()
    overall_directional_acc = results_df["knn_directional"].mean() * 100
    overall_cone_hit = results_df["cone_hit_rate"].mean() * 100

    # Daily breakdown
    daily_accuracy = {}
    for day in [1, 5, 10, 20, 30]:
        col = f"day_{day}_knn"
        if col in results_df.columns:
            daily_accuracy[f"day_{day}"] = results_df[col].mean()

    return {
        "total_forecasts": len(results),
        "date_range": (results_df["test_date"].min(), results_df["test_date"].max()),
        "mae": overall_mae,
        "rmse": overall_rmse,
        "mape": overall_mape,
        "directional_accuracy": overall_directional_acc,
        "cone_hit_rate": overall_cone_hit,
        "daily_accuracy": daily_accuracy,
        "summary_metrics": results_df
    }
