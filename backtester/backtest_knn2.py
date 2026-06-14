"""
Walk-forward backtest for the enhanced knn2 forecaster.
Primary metric: Day 5 directional accuracy.
Same return dict structure as other backtests for leaderboard compatibility.
"""

import io
import numpy as np
import pandas as pd
from contextlib import redirect_stdout, redirect_stderr

from features.engineer        import compute_features
from features.feature_weights import compute_feature_weights
from forecaster.knn2          import run_knn2_forecast, _weighted_percentile
from similarity.search_enhanced import find_top_k_enhanced


def run_knn2_backtest(df: pd.DataFrame, config: dict,
                      params: dict | None = None) -> dict:
    """
    Walk-forward backtest for knn2.

    Feature weights are computed ONCE on data available at test_start_idx
    (no lookahead). They are NOT recomputed at each step for speed —
    this is a documented approximation; weights change slowly within a
    12-month window.

    Returns same dict structure as run_backtest / run_xgb_backtest so
    it plugs into the unified leaderboard.
    """
    months_back   = config.get("BACKTEST_MONTHS", 6)
    step_days     = config.get("BACKTEST_STEP", 20)
    forecast_days = config.get("BACKTEST_FORECAST_DAYS", 30)
    start_date    = config.get("BACKTEST_START")
    end_date      = config.get("BACKTEST_END")
    ticker        = config.get("TICKER", "")
    feature_cols  = config.get("FEATURE_COLS")
    lookback      = config.get("ML_TRAINING_LOOKBACK_BARS", 1500)
    bands         = config.get("CONFIDENCE_BANDS", [20, 60, 90])

    if not feature_cols:
        print("knn2 backtest: FEATURE_COLS not set in config.")
        return {}

    if start_date and end_date:
        test_start_idx = max(0, int(df.index.searchsorted(pd.Timestamp(start_date))))
        test_end_idx   = min(len(df), int(df.index.searchsorted(pd.Timestamp(end_date), side="right")))
    else:
        test_start_idx = max(0, len(df) - months_back * 21)
        test_end_idx   = len(df)

    saved = params or {}
    half_life_days = int(saved.get("half_life_days", 250))
    distance_pct   = float(saved.get("distance_threshold_pct", 20.0))
    k_max  = config.get("K", 30)
    k_min  = max(5, k_max // 4)
    min_gap = config.get("MIN_GAP", 20)
    window_len   = config["WINDOW_LEN"]
    forecast_len = config["FORECAST_LEN"]

    # Compute feature weights once on training data available at test start
    train_start_df = df.iloc[:test_start_idx].copy()
    if len(train_start_df) < 60:
        print("knn2 backtest: insufficient training data.")
        return {}

    feat_weights = compute_feature_weights(
        train_start_df, feature_cols, ticker, lookback,
        force_recompute=True,
    )

    lam = np.log(2) / half_life_days

    results = []

    for i in range(test_start_idx, test_end_idx - forecast_days, step_days):
        test_date     = df.index[i]
        train_df      = df.iloc[:i].copy()
        actual_slice  = df.iloc[i : i + forecast_days].copy()

        if len(train_df) < 100:
            continue

        actual_prices = actual_slice["Close"].values[:forecast_days]
        current_close = float(train_df["Close"].iloc[-1])

        try:
            series = train_df[feature_cols].values

            # Run ensemble search on training data only (suppress verbose output)
            from forecaster.knn2 import _ENSEMBLE_METHODS, _ENSEMBLE_WEIGHTS, _weighted_percentile

            method_votes = []
            primary_paths = None
            primary_weights = None

            for method in _ENSEMBLE_METHODS:
                matches, meta = find_top_k_enhanced(
                    series, window_len, forecast_len, method,
                    weights      = feat_weights,
                    feature_cols = feature_cols,
                    distance_pct = distance_pct,
                    k_min        = k_min,
                    k_max        = k_max,
                    min_gap      = min_gap,
                )

                if not matches:
                    method_votes.append(0.5)
                    continue

                idxs     = [m[0] for m in matches]
                scores   = np.array([m[1] for m in matches], dtype=float)
                age_bars = np.array([m[2] for m in matches], dtype=float)

                recency_w = np.exp(-lam * age_bars)
                dist_w    = np.maximum(scores, 0.0)
                combined  = dist_w * recency_w
                w_sum     = combined.sum()
                combined  = combined / w_sum if w_sum > 0 else np.ones(len(matches)) / len(matches)

                return_series = train_df["return"].values
                paths, valid_w = [], []
                for j, idx in enumerate(idxs):
                    start = idx + window_len
                    end   = start + forecast_len
                    if end > len(train_df):
                        continue
                    analog_returns = return_series[start:end]
                    path = [current_close]
                    for r in analog_returns:
                        path.append(path[-1] * (1.0 + r))
                    paths.append(path[1:])
                    valid_w.append(combined[j])

                if not paths:
                    method_votes.append(0.5)
                    continue

                paths_arr = np.array(paths)
                w_arr     = np.array(valid_w)
                w_arr     = w_arr / w_arr.sum()

                day5_col    = min(4, paths_arr.shape[1] - 1)
                day5_prices = paths_arr[:, day5_col]
                bullish     = (day5_prices > current_close).astype(float)
                vote        = float(np.dot(w_arr, bullish))
                method_votes.append(vote)

                if primary_paths is None:
                    primary_paths   = paths_arr
                    primary_weights = w_arr

            if primary_paths is None or len(primary_paths) == 0:
                continue

            # Ensemble vote
            ew = _ENSEMBLE_WEIGHTS[:len(method_votes)]
            mv = method_votes[:len(_ENSEMBLE_METHODS)]
            ensemble_vote = sum(w * v for w, v in zip(ew, mv)) / sum(ew[:len(mv)])

            # Day 5 directional accuracy (primary metric)
            if forecast_days >= 5:
                day5_pred = _weighted_percentile(primary_paths[:, 4], primary_weights, 50)
                day5_act  = actual_prices[4]
                dir_5_correct = int((day5_pred > current_close) == (day5_act > current_close))
            else:
                day5_correct = int((ensemble_vote > 0.5) == (actual_prices[-1] > current_close))
                dir_5_correct = day5_correct

            # MAE using weighted median at each day vs actual
            knn2_medians = np.array([
                _weighted_percentile(primary_paths[:, d], primary_weights, 50)
                for d in range(min(forecast_days, primary_paths.shape[1]))
            ])
            actual_trimmed = actual_prices[:len(knn2_medians)]
            mae  = float(np.mean(np.abs(knn2_medians - actual_trimmed)))
            mape = float(np.mean(np.abs((actual_trimmed - knn2_medians) / (actual_trimmed + 1e-10))) * 100)

            # Cone hit rate using p-bands
            highest_prob = max(bands)
            lowest_prob  = min(bands)
            knn2_low  = np.array([_weighted_percentile(primary_paths[:, d], primary_weights, 100 - highest_prob) for d in range(len(knn2_medians))])
            knn2_high = np.array([_weighted_percentile(primary_paths[:, d], primary_weights, 100 - lowest_prob)  for d in range(len(knn2_medians))])
            hits     = np.sum((actual_trimmed >= knn2_low) & (actual_trimmed <= knn2_high))
            cone_hit = float(hits / len(actual_trimmed))

            # Day-level MAE
            day_errors = {}
            for day in [1, 5, 10, 20, 30]:
                if day <= len(knn2_medians):
                    day_errors[f"day_{day}_mae"] = abs(knn2_medians[day - 1] - actual_trimmed[day - 1])

            results.append({
                "test_date":     test_date,
                "current_close": current_close,
                "knn2_median":   knn2_medians[-1] if len(knn2_medians) > 0 else np.nan,
                "actual_price":  actual_trimmed[-1],
                "mae":           mae,
                "mape":          mape,
                "directional":   dir_5_correct,
                "cone_hit_rate": cone_hit,
                **day_errors,
            })

        except Exception as e:
            print(f"  Warning: knn2 backtest failed at {test_date}: {e}")
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
        "total_forecasts":      len(results_df),
        "date_range":           (results_df["test_date"].min(), results_df["test_date"].max()),
        "mae":                  results_df["mae"].mean(),
        "rmse":                 float(np.sqrt((results_df["mae"] ** 2).mean())),
        "mape":                 results_df["mape"].mean(),
        "directional_accuracy": results_df["directional"].mean() * 100,
        "cone_hit_rate":        results_df["cone_hit_rate"].mean() * 100,
        "daily_accuracy":       daily_accuracy,
        "summary_metrics":      results_df,
    }
