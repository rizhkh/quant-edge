import numpy as np
import pandas as pd
from features.xgb_features   import build_xgb_features
from models.xgboost_model    import _train_quantile_models


def run_xgb_backtest(df: pd.DataFrame, config: dict, params: dict = None) -> dict:
    """
    Walk-forward backtest for XGBoost.

    At each test date, trains on all data before that date, predicts forward,
    then compares against actual prices. No lookahead.

    Returns the same metric keys as run_backtest() so results are comparable.
    """
    months_back   = config.get("BACKTEST_MONTHS", 6)
    step_days     = config.get("BACKTEST_STEP", 20)
    forecast_days = config.get("BACKTEST_FORECAST_DAYS", 30)
    use_sector    = config.get("XGB_USE_SECTOR", True)
    use_vix       = config.get("XGB_USE_VIX", True)
    use_earnings  = config.get("XGB_USE_EARNINGS", True)
    sector_map    = config.get("SECTOR_MAP", {})
    ticker        = config.get("TICKER")
    lookback      = config.get("ML_TRAINING_LOOKBACK_BARS", 1500)
    start_date    = config.get("BACKTEST_START")
    end_date      = config.get("BACKTEST_END")

    if start_date and end_date:
        test_start_idx = max(0, int(df.index.searchsorted(pd.Timestamp(start_date))))
        test_end_idx   = min(len(df), int(df.index.searchsorted(pd.Timestamp(end_date), side="right")))
    else:
        test_start_idx = max(0, len(df) - months_back * 21)
        test_end_idx   = len(df)

    results = []

    for i in range(test_start_idx, test_end_idx - forecast_days, step_days):
        test_date    = df.index[i]
        train_df     = df.iloc[:i].copy()
        actual_slice = df.iloc[i : i + forecast_days].copy()

        if len(train_df) < 100:
            continue

        try:
            X_all     = build_xgb_features(train_df, ticker=ticker,
                                           use_sector=use_sector, use_vix=use_vix,
                                           use_earnings=use_earnings, sector_map=sector_map)
            X_current = X_all.iloc[[-1]].values

            if np.isnan(X_current).any():
                continue

            current_close = float(train_df["Close"].iloc[-1])
            actual_prices = actual_slice["Close"].values[:forecast_days]

            # Train one model set targeting the full forecast horizon
            fwd_return = train_df["Close"].pct_change(forecast_days).shift(-forecast_days)
            combined   = X_all.copy()
            combined["_y"] = fwd_return
            combined   = combined.dropna()
            if lookback and len(combined) > lookback:
                combined = combined.iloc[-lookback:]

            if len(combined) < 60:
                continue

            conf_bands = config.get("CONFIDENCE_BANDS", [10, 50, 90])
            quantiles  = sorted(set([(100 - p) / 100 for p in conf_bands] + [0.50]))
            models = _train_quantile_models(
                combined.drop(columns=["_y"]).values,
                combined["_y"].values,
                quantiles,
                params,
            )

            q_low  = min(quantiles)   # lowest quantile = lowest price (e.g. 10th percentile)
            q_high = max(quantiles)   # highest quantile = highest price (e.g. 80th percentile)
            xgb_low  = float(current_close * (1 + models[q_low].predict(X_current)[0]))
            xgb_med  = float(current_close * (1 + models[0.50].predict(X_current)[0]))
            xgb_high = float(current_close * (1 + models[q_high].predict(X_current)[0]))

            actual_final = float(actual_prices[-1])
            mae  = abs(xgb_med - actual_final)
            mape = abs(xgb_med - actual_final) / actual_final * 100

            pred_dir   = 1 if xgb_med > current_close else 0
            actual_dir = 1 if actual_final > current_close else 0
            dir_correct = 1 if pred_dir == actual_dir else 0

            hits = sum(1 for p in actual_prices if xgb_low <= p <= xgb_high)
            cone_hit = hits / len(actual_prices)

            # Per-day MAE at milestones
            day_errors = {}
            for day in [1, 5, 10, 20, 30]:
                if day <= len(actual_prices):
                    day_errors[f"day_{day}_mae"] = abs(xgb_med - actual_prices[day - 1])

            results.append({
                "test_date":         test_date,
                "current_close":     current_close,
                "xgb_low":           xgb_low,
                "xgb_median":        xgb_med,
                "xgb_high":          xgb_high,
                "actual_price":      actual_final,
                "mae":               mae,
                "mape":              mape,
                "directional":       dir_correct,
                "cone_hit_rate":     cone_hit,
                **day_errors,
            })

        except Exception as e:
            print(f"  Warning: XGBoost backtest failed at {test_date}: {e}")
            continue

    if not results:
        return {}

    results_df = pd.DataFrame(results)

    daily_accuracy = {}
    for day in [1, 5, 10, 20, 30]:
        col = f"day_{day}_mae"
        if col in results_df.columns:
            daily_accuracy[f"day_{day}"] = results_df[col].mean()

    return {
        "total_forecasts":      len(results),
        "date_range":           (results_df["test_date"].min(), results_df["test_date"].max()),
        "mae":                  results_df["mae"].mean(),
        "mape":                 results_df["mape"].mean(),
        "directional_accuracy": results_df["directional"].mean() * 100,
        "cone_hit_rate":        results_df["cone_hit_rate"].mean() * 100,
        "daily_accuracy":       daily_accuracy,
        "summary_metrics":      results_df,
    }
