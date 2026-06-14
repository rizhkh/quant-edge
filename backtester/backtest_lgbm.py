import numpy as np
import pandas as pd
from features.lgbm_features  import build_lgbm_features
from models.lightgbm_model   import _train_quantile_models


def run_lgbm_backtest(df: pd.DataFrame, config: dict, params: dict = None) -> dict:
    """
    Walk-forward backtest for LightGBM.
    Mirrors backtest_xgb.py structure — returns identical metric keys.
    """
    months_back   = config.get("BACKTEST_MONTHS", 6)
    step_days     = config.get("BACKTEST_STEP", 20)
    forecast_days = config.get("BACKTEST_FORECAST_DAYS", 30)
    use_vix       = config.get("LGBM_USE_VIX", True)
    use_spy       = config.get("LGBM_USE_SPY", True)
    lookback      = config.get("ML_TRAINING_LOOKBACK_BARS", 1500)
    start_date    = config.get("BACKTEST_START")
    end_date      = config.get("BACKTEST_END")

    if start_date and end_date:
        test_start_idx = max(0, int(df.index.searchsorted(pd.Timestamp(start_date))))
        test_end_idx   = min(len(df), int(df.index.searchsorted(pd.Timestamp(end_date), side="right")))
    else:
        test_start_idx = max(0, len(df) - months_back * 21)
        test_end_idx   = len(df)

    conf_bands = config.get("CONFIDENCE_BANDS", [10, 50, 90])
    quantiles  = sorted(set([(100 - p) / 100 for p in conf_bands] + [0.50]))

    # Build full feature matrix once (no lookahead — we slice per test point)
    X_full = build_lgbm_features(df, use_vix=use_vix, use_spy=use_spy)

    results = []

    for i in range(test_start_idx, test_end_idx - forecast_days, step_days):
        test_date    = df.index[i]
        train_df     = df.iloc[:i].copy()
        actual_slice = df.iloc[i: i + forecast_days].copy()

        if len(train_df) < 100:
            continue

        try:
            X_train_full = X_full.iloc[:i].copy()
            X_current    = X_full.iloc[[i - 1]]  # keep as DataFrame — matches fit feature names

            current_close = float(train_df["Close"].iloc[-1])
            actual_prices = actual_slice["Close"].values[:forecast_days]

            fwd_return = train_df["Close"].pct_change(forecast_days).shift(-forecast_days)
            combined   = X_train_full.copy()
            combined["_y"] = fwd_return
            combined   = combined.dropna()
            if lookback and len(combined) > lookback:
                combined = combined.iloc[-lookback:]

            if len(combined) < 60:
                continue

            X_tr = combined.drop(columns=["_y"])  # keep DataFrame — consistent with X_current
            y_tr = combined["_y"].values

            models = _train_quantile_models(X_tr, y_tr, quantiles, params)

            q_low  = min(quantiles)   # lowest quantile = lowest price (e.g. 10th percentile)
            q_high = max(quantiles)   # highest quantile = highest price (e.g. 80th percentile)
            lgb_low  = float(current_close * (1 + models[q_low].predict(X_current)[0]))
            lgb_med  = float(current_close * (1 + models[0.50].predict(X_current)[0]))
            lgb_high = float(current_close * (1 + models[q_high].predict(X_current)[0]))

            actual_final = float(actual_prices[-1])
            mae  = abs(lgb_med - actual_final)
            mape = abs(lgb_med - actual_final) / (actual_final + 1e-10) * 100

            dir_correct = int((lgb_med > current_close) == (actual_final > current_close))
            hits        = sum(1 for p in actual_prices if lgb_low <= p <= lgb_high)
            cone_hit    = hits / len(actual_prices)

            day_errors = {}
            for day in [1, 5, 10, 20, 30]:
                if day <= len(actual_prices):
                    day_errors[f"day_{day}_mae"] = abs(lgb_med - actual_prices[day - 1])

            results.append({
                "test_date":     test_date,
                "current_close": current_close,
                "lgb_low":       lgb_low,
                "lgb_median":    lgb_med,
                "lgb_high":      lgb_high,
                "actual_price":  actual_final,
                "mae":           mae,
                "mape":          mape,
                "directional":   dir_correct,
                "cone_hit_rate": cone_hit,
                **day_errors,
            })

        except Exception as e:
            print(f"  Warning: LightGBM backtest failed at {test_date}: {e}")
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
