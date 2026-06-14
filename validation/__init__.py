import pandas as pd
import numpy as np


def compare_forecasts(df: pd.DataFrame, forecast_csv_path: str = "output/forecast_results.csv") -> pd.DataFrame:
    """
    Compare forecasted prices to actual prices for any days that have passed.

    Returns DataFrame with columns:
      forecast_date, actual_date, analog_forecast, knn_forecast, actual_price,
      analog_error, knn_error, analog_pct_error, knn_pct_error
    """

    # Read forecast results
    forecast_df = pd.read_csv(forecast_csv_path)
    forecast_df['date'] = pd.to_datetime(forecast_df['date'])

    # Get historical data with dates
    hist_data = df.copy()
    hist_data = hist_data[['Close']].reset_index()
    hist_data['Date'] = pd.to_datetime(hist_data['Date'])
    hist_data = hist_data.rename(columns={'Date': 'date', 'Close': 'actual_price'})

    # Get today's date
    today = pd.Timestamp.now().normalize()

    # Find which forecast dates have passed
    forecast_df['has_passed'] = forecast_df['date'] < today
    past_forecasts = forecast_df[forecast_df['has_passed']].copy()

    if len(past_forecasts) == 0:
        print("No forecast dates have passed yet.")
        return pd.DataFrame()

    # Merge with actual prices
    comparison = past_forecasts.merge(
        hist_data,
        left_on='date',
        right_on='date',
        how='left'
    )

    # Drop rows where actual price is missing (date not in historical data)
    comparison = comparison.dropna(subset=['actual_price'])

    # Calculate errors
    comparison['analog_error'] = abs(comparison['analog_price'] - comparison['actual_price'])
    comparison['knn_error'] = abs(comparison['knn_median'] - comparison['actual_price'])

    comparison['analog_pct_error'] = (comparison['analog_error'] / comparison['actual_price'] * 100)
    comparison['knn_pct_error'] = (comparison['knn_error'] / comparison['actual_price'] * 100)

    return comparison[['day', 'date', 'analog_price', 'knn_median', 'knn_low', 'knn_high',
                       'actual_price', 'analog_error', 'knn_error', 'analog_pct_error', 'knn_pct_error']]
