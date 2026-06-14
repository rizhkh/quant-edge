import pandas as pd
import numpy as np


def compare_forecasts(df: pd.DataFrame, forecast_csv_path: str) -> pd.DataFrame:
    """
    Compare forecasted prices to actual prices for any days that have passed.

    Enriched columns beyond price error:
      - prior_close:        previous trading day's close (forecast baseline)
      - actual_open/high/low: intraday OHLC for range context
      - actual_move_pct:    actual % move from prior close
      - predicted_move_pct: k-NN median % move from prior close
      - direction_correct:  whether forecast called up/down correctly
      - band_touched:       whether intraday range overlapped knn band at any point
    """

    forecast_df = pd.read_csv(forecast_csv_path)
    forecast_df['date'] = pd.to_datetime(forecast_df['date'])

    # Build OHLC history keyed by date
    hist = df.copy()[['Open', 'High', 'Low', 'Close']].reset_index()
    hist['Date'] = pd.to_datetime(hist['Date'])
    hist = hist.rename(columns={
        'Date':  'date',
        'Open':  'actual_open',
        'High':  'actual_high',
        'Low':   'actual_low',
        'Close': 'actual_price',
    })

    # Prior close = previous row's Close (shift by 1)
    hist['prior_close'] = hist['actual_price'].shift(1)

    today = pd.Timestamp.now().normalize()

    forecast_df['has_passed'] = forecast_df['date'] <= today
    past_forecasts = forecast_df[forecast_df['has_passed']].copy()

    if len(past_forecasts) == 0:
        print("No forecast dates have passed yet.")
        return pd.DataFrame()

    # Dedupe: forecast_results accumulates snapshots over time. Keep the latest
    # snapshot per date (highest last_updated). Without this, validation
    # double-counts the same date across multiple forecast runs.
    if 'last_updated' in past_forecasts.columns:
        past_forecasts['last_updated'] = pd.to_datetime(past_forecasts['last_updated'], errors='coerce')
        past_forecasts = past_forecasts.sort_values(['date', 'last_updated'])
        past_forecasts = past_forecasts.drop_duplicates(subset=['date'], keep='last')

    comparison = past_forecasts.merge(hist, on='date', how='left')
    comparison = comparison.dropna(subset=['actual_price'])

    # Derive knn_* reference columns from the winning similarity method.
    # Triggered when knn_method is present AND knn_* columns are missing OR fully null
    # (some CSVs have knn_median as an empty placeholder column — must still derive).
    if 'knn_method' in comparison.columns:
        existing_knn_med = comparison['knn_median'] if 'knn_median' in comparison.columns else None
        needs_derive = existing_knn_med is None or existing_knn_med.isna().all()
        if needs_derive:
            for suffix in ['median', 'low', 'high', 'direction']:
                comparison[f'knn_{suffix}'] = comparison.apply(
                    lambda r: r.get(f"{r['knn_method']}_{suffix}")
                    if pd.notna(r.get('knn_method')) else None,
                    axis=1,
                )

    # Coerce derived knn columns to numeric — apply() returns object dtype when any
    # row is None (e.g. tickers with no kNN data ever), which breaks np.sign() below.
    for col in ['knn_median', 'knn_low', 'knn_high', 'analog_price']:
        if col in comparison.columns:
            comparison[col] = pd.to_numeric(comparison[col], errors='coerce')

    # If kNN data is entirely missing for this ticker, raise a clean error rather than
    # cryptic numpy "unorderable types" deeper down. Caller can surface it as a skip.
    if comparison['knn_median'].dropna().empty:
        raise ValueError(
            "No kNN data available for any past date — run `main.py knn TICKER` "
            "or `main.py run_all knn` to populate the kNN columns."
        )

    # Price errors — kNN and analog
    comparison['analog_error']     = abs(comparison['analog_price'] - comparison['actual_price'])
    comparison['knn_error']        = abs(comparison['knn_median']   - comparison['actual_price'])
    comparison['analog_pct_error'] = comparison['analog_error'] / comparison['actual_price'] * 100
    comparison['knn_pct_error']    = comparison['knn_error']    / comparison['actual_price'] * 100

    # Move % from prior close
    comparison['actual_move_pct']    = (comparison['actual_price'] - comparison['prior_close']) / comparison['prior_close'] * 100
    comparison['predicted_move_pct'] = (comparison['knn_median']   - comparison['prior_close']) / comparison['prior_close'] * 100

    # Direction: did the kNN model call up/down correctly?
    comparison['direction_correct'] = (
        np.sign(comparison['actual_move_pct']) == np.sign(comparison['predicted_move_pct'])
    )

    # kNN band touched intraday
    comparison['band_touched'] = (
        (comparison['actual_high'] >= comparison['knn_low']) &
        (comparison['actual_low']  <= comparison['knn_high'])
    )

    # Direction validation using current_close (close at forecast time) for all models
    if 'current_close' in comparison.columns:
        actual_up = comparison['actual_price'] > comparison['current_close']
        for prefix in ['analog', 'knn', 'spearman', 'pearson', 'cosine', 'euclidean', 'kendall',
                       'manhattan', 'xgb', 'lgb', 'rf', 'knn2']:
            dir_col = 'analog_direction' if prefix == 'analog' else f'{prefix}_direction'
            if dir_col in comparison.columns:
                pred_up = comparison[dir_col] == 'UP'
                comparison[f'{prefix}_direction_correct'] = (pred_up == actual_up)

    # Random Forest columns — only if present in the forecast CSV
    if 'rf_median' in comparison.columns:
        comparison['rf_error']     = abs(comparison['rf_median'] - comparison['actual_price'])
        comparison['rf_pct_error'] = comparison['rf_error'] / comparison['actual_price'] * 100
        comparison['rf_in_band']   = (
            (comparison['actual_price'] >= comparison['rf_low']) &
            (comparison['actual_price'] <= comparison['rf_high'])
        )
        comparison['rf_band_touched'] = (
            (comparison['actual_high'] >= comparison['rf_low']) &
            (comparison['actual_low']  <= comparison['rf_high'])
        )

    # LightGBM columns — only if present in the forecast CSV
    if 'lgb_median' in comparison.columns:
        comparison['lgb_error']     = abs(comparison['lgb_median'] - comparison['actual_price'])
        comparison['lgb_pct_error'] = comparison['lgb_error'] / comparison['actual_price'] * 100
        comparison['lgb_in_band']   = (
            (comparison['actual_price'] >= comparison['lgb_low']) &
            (comparison['actual_price'] <= comparison['lgb_high'])
        )
        comparison['lgb_band_touched'] = (
            (comparison['actual_high'] >= comparison['lgb_low']) &
            (comparison['actual_low']  <= comparison['lgb_high'])
        )

    # XGBoost columns — only if present in the forecast CSV
    if 'xgb_median' in comparison.columns:
        comparison['xgb_error']     = abs(comparison['xgb_median'] - comparison['actual_price'])
        comparison['xgb_pct_error'] = comparison['xgb_error'] / comparison['actual_price'] * 100
        comparison['xgb_direction_correct'] = (
            np.sign(comparison['actual_move_pct']) ==
            np.sign((comparison['xgb_median'] - comparison['prior_close']) / comparison['prior_close'] * 100)
        )
        comparison['xgb_band_touched'] = (
            (comparison['actual_high'] >= comparison['xgb_low']) &
            (comparison['actual_low']  <= comparison['xgb_high'])
        )
        comparison['xgb_in_band'] = (
            (comparison['actual_price'] >= comparison['xgb_low']) &
            (comparison['actual_price'] <= comparison['xgb_high'])
        )

    # Return all columns — base + computed + any p-band / per-method columns from the forecast CSV
    base_cols = [
        'day', 'date',
        'prior_close', 'analog_price', 'knn_median', 'knn_low', 'knn_high',
        'actual_open', 'actual_high', 'actual_low', 'actual_price',
        'analog_error', 'knn_error', 'analog_pct_error', 'knn_pct_error',
        'actual_move_pct', 'predicted_move_pct', 'direction_correct', 'band_touched',
    ]
    extra_cols = [c for c in comparison.columns if c not in base_cols]
    return comparison[base_cols + extra_cols]
